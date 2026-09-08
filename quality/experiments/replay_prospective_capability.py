#!/usr/bin/env python3
"""Independently replay a sealed prospective experiment using its frozen source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=8)
    args=parser.parse_args()
    first=args.root/"events/000000.json"
    plan=json.loads(first.read_bytes())["payload"]
    for relative,digest in plan["source_sha256"].items():
        path=Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source path escapes snapshot")
        if hashlib.sha256((args.root/"source"/path).read_bytes()).hexdigest()!=digest:
            raise ValueError("frozen implementation mismatch")
    sys.path.insert(0,str(args.root/"source"))
    from quality.experiments.prospective_capability_evidence import read_journal,verify_run
    from quality.experiments.causal_metacognition_experiment import predict,observe_interventions
    from quality.experiments.v68_receiver_response_probe import load_model,sha_file,write_new,canonical
    from quality.experiments.v66_mbpp_capability_extract import run_restricted_tests
    from quality.experiments.certify_capability_evidence import require
    result=verify_run(args.root)
    records,tip=read_journal(args.root/"events")
    stages={row["kind"]:row["payload"] for row in records}
    benchmark=json.loads((args.root/"benchmark.json").read_bytes())
    cases={row["task_id"]:row for row in benchmark["cases"]}
    output=args.root/"independent-replay.json"
    require(not output.exists(),"independent replay already exists")
    for name,digest in plan["model_files_sha256"].items():
        require(sha_file(args.model_dir/name)==digest,"replay model/config/tokenizer mismatch")
    # Re-execute the deterministic code tests, independently of stored pass flags.
    for key,entry in benchmark["verification"].items():
        require(run_restricted_tests(entry["reference"],entry["tests"]),"reference code failed independent replay")
        require(not run_restricted_tests(entry["mutated"],entry["tests"]),"mutant passed independent replay")
        answer=benchmark["answers"][key]
        options=[entry["reference"],entry["mutated"]] if answer==0 else [entry["mutated"],entry["reference"]]
        require(cases[key]["prompt"].endswith("\nA:\n"+options[0]+"\nB:\n"+options[1]+"\nAnswer:"), "prompt alternatives do not bind verified programs")
    model,tokenizer=load_model(args.model_dir,args.threads)
    replayed=predict(model,tokenizer,[cases[k] for k in plan["calibration_ids"]+plan["test_ids"]],plan["label_ids"])
    require(replayed=={**stages["calibration_predictions"],**stages["test_predictions"]},
            "fresh-process prediction mismatch")
    predictions=stages["intervention_predictions"]
    modules={site:model.get_submodule(site) for site in plan["sites"]}
    traces={(row["task_id"],row["site"]):args.root/row["trace_path"] for row in predictions.values()}
    observed=observe_interventions(model,tokenizer,[cases[k] for k in plan["causal_ids"]],
                                  plan["label_ids"],modules,predictions,traces)
    require(observed==stages["intervention_observations"],"fresh-process intervention mismatch")
    require(all(not m._forward_hooks for m in modules.values()),"replay hook remains")
    require(tip==read_journal(args.root/"events")[1],"journal changed during independent replay")
    final={"schema":"cerebro.tidex.prospective_independent_replay/v1","complete":True,"pass":True,
           "journal_tip_sha256":tip,"prediction_count":len(replayed),"intervention_count":len(observed),
           "program_pairs_reexecuted":len(benchmark["verification"]),
           "fresh_process_predictions_exact":True,"fresh_process_interventions_exact":True,
           "scientific_verdict":result,"scope":"independent process, same machine/model/protocol; not independent laboratory replication"}
    write_new(output,canonical(final))
    print(json.dumps(final,indent=2),flush=True)


if __name__=="__main__":main()
