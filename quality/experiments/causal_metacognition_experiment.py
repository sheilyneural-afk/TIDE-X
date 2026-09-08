#!/usr/bin/env python3
"""Measure internal LoRA-contribution interventions and prospective monitoring.

Uses the real standalone V67 checkpoint. The benchmark is binary MBPP code
correctness discrimination, not free-form synthesis. Negative results are final
for a run; thresholds, cases and scales cannot be changed after measurements.
"""
from __future__ import annotations
import argparse
import ast
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from quality.experiments.v66_mbpp_capability_extract import safe_python_ast, run_restricted_tests
from quality.experiments.v68_receiver_response_probe import load_model, sha_file, write_new, canonical
from quality.experiments.prospective_capability_evidence import (
    SCHEMA, Journal, choose_actions, fit_monitor, verify_run,
)
from quality.experiments.certify_capability_evidence import require


def mutated_codes(code):
    tree = ast.parse(code)
    changes = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add,
               ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE,
               ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
    nodes = list(ast.walk(tree))
    for index, node in enumerate(nodes):
        if type(node) not in changes and not (isinstance(node, ast.Constant) and type(node.value) is int):
            continue
        changed = copy.deepcopy(tree)
        target = list(ast.walk(changed))[index]
        if type(node) in changes:
            # Operator nodes have no payload; replace their class in this copy.
            target.__class__ = changes[type(node)]
        else:
            target.value += 1
        yield ast.unparse(ast.fix_missing_locations(changed)) + "\n"


def prepare_cases(tokenizer, excluded, count, seed):
    dataset = load_dataset("mbpp", download_mode="reuse_dataset_if_exists")
    rows = sorted(dataset["test"], key=lambda row: int(row["task_id"]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    cases, answers, provenance = [], {}, {}
    for row in rows:
        key = str(row["task_id"])
        if key in excluded or row["test_setup_code"].strip() or not safe_python_ast(row["code"]):
            continue
        tests = row["test_list"]
        if len(tests) < 3 or not run_restricted_tests(row["code"], tests):
            continue
        good = ast.unparse(ast.parse(row["code"])) + "\n"
        if not run_restricted_tests(good, tests):
            continue
        for index, bad in enumerate(mutated_codes(good)):
            if index >= 32:
                break
            if bad == good or not safe_python_ast(bad) or run_restricted_tests(bad, tests):
                continue
            correct = rng.randrange(2)
            choices = [good, bad] if correct == 0 else [bad, good]
            prompt = (
                "Choose the Python implementation that correctly solves the task. "
                "Answer only A or B.\nTask: " + row["text"].strip()
                + "\nA:\n" + choices[0] + "\nB:\n" + choices[1] + "\nAnswer:"
            )
            if len(tokenizer.encode(prompt, add_special_tokens=False)) > 384:
                break
            cases.append({"task_id": key, "prompt": prompt})
            answers[key] = correct
            provenance[key] = {"tests": tests, "reference": good, "mutated": bad,
                               "reference_passed": True, "mutant_passed": False}
            break
        if len(cases) == count:
            break
    require(len(cases) == count, f"insufficient unobserved eligible tasks: {len(cases)}/{count}")
    return cases, answers, provenance, dataset["test"]._fingerprint


def choice_forward(model, tokenizer, text, label_ids):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    require(encoded.input_ids.shape[1] <= 384, "token budget exceeded")
    logits = model(**encoded, use_cache=False).logits[0, -1, label_ids]
    require(bool(torch.isfinite(logits).all()), "nonfinite choice logits")
    return logits


def predict(model, tokenizer, cases, labels):
    result = {}
    with torch.no_grad():
        for index, case in enumerate(cases):
            logits = choice_forward(model, tokenizer, case["prompt"], labels)
            probabilities = torch.softmax(logits.float(), dim=-1)
            result[case["task_id"]] = {
                "choice": int(logits.argmax()), "raw_confidence": float(probabilities.max()),
                "margin": float(logits[0]-logits[1]),
                "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
            }
            if (index+1) % 8 == 0:
                print(f"predictions {index+1}/{len(cases)}", flush=True)
    return result


def adapter_factors(model, adapter_dir, sites):
    manifest = json.loads((adapter_dir/"manifest.json").read_bytes())
    require(sha_file(adapter_dir/"adapter_model.safetensors") ==
            manifest["adapter"]["files_sha256"]["adapter_model.safetensors"], "adapter hash mismatch")
    require(sha_file(adapter_dir/"adapter_config.json") ==
            manifest["adapter"]["files_sha256"]["adapter_config.json"], "adapter config hash mismatch")
    config = json.loads((adapter_dir/"adapter_config.json").read_bytes())
    require(not config.get("use_rslora",False) and not config.get("use_dora",False)
            and not config.get("rank_pattern") and not config.get("alpha_pattern"), "unsupported LoRA recipe")
    tensors = load_file(str(adapter_dir/"adapter_model.safetensors"))
    modules, factors = {}, {}
    for site in sites:
        modules[site] = model.get_submodule(site)
        prefix = "base_model.model." + site
        a, b = tensors[prefix+".lora_A.weight"], tensors[prefix+".lora_B.weight"]
        require(a.shape[0] == config["r"] and b.shape[1] == config["r"], "LoRA rank mismatch")
        factors[site] = (a, b, config["lora_alpha"]/config["r"])
    return modules, factors


def predict_interventions(model, tokenizer, cases, labels, modules, factors, plan, root):
    predictions, traces = {}, {}
    for case in cases:
        captures = {}
        handles = []
        for site, module in modules.items():
            a,b,scale = factors[site]
            def capture(_module, inputs, output, site=site, a=a, b=b, scale=scale):
                delta = ((inputs[0].detach().float() @ a.T) @ b.T) * scale
                output = output.requires_grad_(True)
                output.retain_grad()
                captures[site] = (output, delta)
                return output
            handles.append(module.register_forward_hook(capture))
        try:
            logits = choice_forward(model, tokenizer, case["prompt"], labels)
            margin = logits[0]-logits[1]
            margin.backward()
            for site in modules:
                output, delta = captures[site]
                require(output.grad is not None, "no internal derivative")
                gradient = output.grad.detach()
                # Exact Frobenius-norm matched control, fixed channel roll.
                permuted = torch.roll(delta, shifts=plan["channel_roll"], dims=-1)
                trace_id = case["task_id"] + "-" + site.replace(".", "_")
                trace_path = root/"traces"/(trace_id+".safetensors")
                save_file({"gradient": gradient.contiguous(), "learned_delta": delta.contiguous(),
                           "permuted_delta": permuted.contiguous()}, str(trace_path))
                trace_sha = sha_file(trace_path)
                traces[(case["task_id"],site)] = trace_path
                for arm,direction in [("learned_delta",delta),("permuted_delta",permuted)]:
                    derivative = float((gradient.double()*direction.double()).sum())
                    for strength in plan["strengths"]:
                        key = trace_id + "-" + arm + "-" + str(strength)
                        predictions[key] = {
                            "task_id": case["task_id"], "site":site, "arm":arm, "strength":strength,
                            "baseline_margin":float(margin.detach()),
                            "predicted_change": -strength*derivative,
                            "trace_path":str(trace_path.relative_to(root)), "trace_sha256":trace_sha,
                            "direction_norm":float(direction.double().norm()),
                        }
        finally:
            for handle in handles:
                handle.remove()
            captures.clear()
            model.zero_grad(set_to_none=True)
        print("intervention predictions committed in memory for task " + case["task_id"],flush=True)
    return predictions,traces


def observe_interventions(model, tokenizer, cases, labels, modules, predictions, traces):
    by_id = {case["task_id"]:case for case in cases}
    observations = {}
    cache_key, trace = None, None
    for index,(key,prediction) in enumerate(predictions.items()):
        pair = (prediction["task_id"],prediction["site"])
        if pair != cache_key:
            trace = load_file(str(traces[pair]))
            cache_key = pair
        direction = trace[prediction["arm"]]
        strength = prediction["strength"]
        def intervene(_module, _inputs, output):
            require(output.shape == direction.shape, "intervention state shape mismatch")
            return output - strength*direction
        handle = modules[prediction["site"]].register_forward_hook(intervene)
        try:
            with torch.no_grad():
                logits = choice_forward(model,tokenizer,by_id[pair[0]]["prompt"],labels)
            observations[key] = {
                "prediction_sha256":hashlib.sha256(canonical(prediction)).hexdigest(),
                "margin":float(logits[0]-logits[1]), "choice":int(logits.argmax()),
            }
        finally:
            handle.remove()
        if (index+1)%12==0:
            print(f"internal interventions {index+1}/{len(predictions)}",flush=True)
    return observations


def run(args):
    require(not args.root.exists(), "run exists; preserve it and choose a new explicit run")
    require(args.root.is_absolute() and not args.root.is_relative_to(ROOT), "run must be outside source")
    args.root.mkdir(parents=True,mode=0o700)
    (args.root/"traces").mkdir()
    journal = Journal(args.root/"events")
    torch.manual_seed(args.seed)
    model,tokenizer = load_model(args.model_dir,args.threads)
    labels = [tokenizer.encode(x,add_special_tokens=False) for x in [" A"," B"]]
    require(all(len(x)==1 for x in labels), "A/B verbalizers must be single tokens")
    labels = [x[0] for x in labels]
    replay = json.loads(args.previous_replay.read_bytes())
    manifest = json.loads((args.adapter_dir/"manifest.json").read_bytes())
    smoke = json.loads(args.smoke.read_bytes())
    require(sha_file(args.adapter_dir/"manifest.json")==smoke["source_v66_manifest_sha256"]
            ==replay["adapter_artifact"]["manifest_sha256"], "adapter/source manifest mismatch")
    require(sha_file(args.model_dir/"model.safetensors")==smoke["materialization"]["output_model_sha256"],
            "standalone checkpoint mismatch")
    excluded = set(str(x) for x in manifest["source_behavior"]["heldout_task_ids"])
    excluded.update(str(x) for x in replay["fresh_evaluation"]["task_ids"])
    ir = json.loads(args.ir.read_bytes())
    require(sha_file(args.ir)==manifest["capability_ir_sha256"], "IR identity mismatch")
    excluded.update(str(row["task_id"]) for row in ir["anchors"])
    cases,answers,provenance,fingerprint = prepare_cases(tokenizer,excluded,args.calibration+args.test,args.seed)
    calibration,test = cases[:args.calibration],cases[args.calibration:]
    sites = [f"model.layers.{i}.self_attn.v_proj" for i in [3,11,19]]
    modules,factors = adapter_factors(model,args.adapter_dir,sites)
    model_hashes = {p.name:sha_file(p) for p in args.model_dir.iterdir() if p.is_file()}
    source_paths = [
        Path(__file__), ROOT/"quality/experiments/prospective_capability_evidence.py",
        ROOT/"quality/experiments/capability_statistics.py",
        ROOT/"quality/experiments/certify_capability_evidence.py",
        ROOT/"quality/experiments/v66_mbpp_capability_extract.py",
        ROOT/"quality/experiments/v68_receiver_response_probe.py",
    ]
    source_hashes = {}
    for path in source_paths:
        relative = str(path.resolve().relative_to(ROOT))
        write_new(args.root/"source"/relative,path.read_bytes())
        source_hashes[relative]=sha_file(path)
    write_new(args.root/"benchmark.json",canonical({"cases":cases,"answers":answers,"verification":provenance}))
    plan = {
        "schema":SCHEMA,"seed":args.seed,"sites":sites,"strengths":[-0.5,0.5],"channel_roll":137,
        "calibration_ids":[x["task_id"] for x in calibration],
        "test_ids":[x["task_id"] for x in test],"causal_ids":[x["task_id"] for x in test[:args.causal]],
        "excluded_ids":sorted(excluded),"source_sha256":source_hashes,
        "model_files_sha256":model_hashes,"benchmark_sha256":sha_file(args.root/"benchmark.json"),
        "dataset_test_fingerprint":fingerprint,"label_ids":labels,
        "runtime":{"python":sys.version,"torch":torch.__version__},
        "criteria":{"minimum_effect_rms":0.01,"maximum_relative_prediction_rmse":0.25,
                    "minimum_sign_accuracy":0.8,"minimum_brier_improvement":0.01,
                    "minimum_auc":0.6,"minimum_selection_gain":0.05,"maximum_p":0.025},
        "scope":"binary correct-versus-mutated MBPP program discrimination on previously excluded test tasks",
        "test_outcomes_visible_to_monitor":False,
    }
    journal.append("precommit",plan)
    calibration_predictions=predict(model,tokenizer,calibration,labels)
    journal.append("calibration_predictions",calibration_predictions)
    calibration_answers={x["task_id"]:answers[x["task_id"]] for x in calibration}
    journal.append("calibration_answers",calibration_answers)
    monitor=fit_monitor(calibration_predictions,calibration_answers)
    journal.append("monitor_frozen",monitor)
    test_predictions=predict(model,tokenizer,test,labels)
    journal.append("test_predictions",test_predictions)
    actions=choose_actions(test_predictions,monitor,args.seed)
    journal.append("actions_committed",actions)
    causal_cases=test[:args.causal]
    predictions,traces=predict_interventions(model,tokenizer,causal_cases,labels,modules,factors,plan,args.root)
    journal.append("intervention_predictions",predictions)
    observations=observe_interventions(model,tokenizer,causal_cases,labels,modules,predictions,traces)
    journal.append("intervention_observations",observations)
    journal.append("test_answers",{x["task_id"]:answers[x["task_id"]] for x in test})
    restored=predict(model,tokenizer,causal_cases,labels)
    require(restored=={x["task_id"]:test_predictions[x["task_id"]] for x in causal_cases},
            "baseline not restored exactly after interventions")
    require(all(not module._forward_hooks for module in modules.values()), "hook remained installed")
    require(all(sha_file(args.model_dir/name)==digest for name,digest in model_hashes.items()),
            "checkpoint files changed during experiment")
    require(all(sha_file(ROOT/path)==digest for path,digest in source_hashes.items()), "source changed during run")
    journal.append("completion",{"weights_unchanged":True,"hooks_removed":True,
                                "baseline_restored_exactly":True,"model_files_sha256":model_hashes})
    result=verify_run(args.root)
    write_new(args.root/"receipt.json",canonical(result))
    print(json.dumps(result,indent=2),flush=True)
    return 0 if all(result[k]["status"].startswith("SUPPORTED") for k in ("mechanism","metacognition")) else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir","adapter-dir","previous-replay","smoke","ir","root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20260909)
    parser.add_argument("--calibration",type=int,default=24)
    parser.add_argument("--test",type=int,default=48)
    parser.add_argument("--causal",type=int,default=6)
    parser.add_argument("--threads",type=int,default=8)
    args=parser.parse_args()
    require(args.calibration>=20 and args.test>=40 and 3<=args.causal<=args.test, "insufficient trial sizes")
    require(1<=args.threads<=8,"threads must be 1..8")
    existed=args.root.exists()
    try:
        return run(args)
    except Exception as error:
        if not existed and args.root.exists():
            write_new(args.root/"failure.json",canonical({"complete":False,"error":str(error)}))
        raise


if __name__=="__main__":
    raise SystemExit(main())
