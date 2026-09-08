import copy
from pathlib import Path
import tempfile
import unittest
from quality.experiments.prospective_capability_evidence import Journal, read_journal, fit_monitor, choose_actions, score_monitor
from quality.experiments.causal_metacognition_experiment import mutated_codes
from quality.experiments.v66_mbpp_capability_extract import run_restricted_tests


class ProspectiveTests(unittest.TestCase):
    def test_real_program_mutation_changes_executed_result(self):
        code="def add(a,b):\n return a+b\n"
        tests=["assert add(2,3)==5", "assert add(-1,3)==2"]
        self.assertTrue(run_restricted_tests(code,tests))
        variants=list(mutated_codes(code))
        self.assertTrue(variants)
        self.assertTrue(any(not run_restricted_tests(v,tests) for v in variants))

    def test_journal_detects_reordering_and_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"events"
            journal=Journal(path)
            journal.append("prediction",{"margin":1.0})
            journal.append("observation",{"margin":0.0})
            self.assertEqual(len(read_journal(path)[0]),2)
            first=path/"000000.json"
            first.write_text(first.read_text().replace("1.0","2.0"))
            with self.assertRaises(ValueError):read_journal(path)

    def test_actions_have_equal_budget_and_do_not_accept_answers(self):
        predictions={str(i):{"choice":i%2,"raw_confidence":0.5+i/100} for i in range(40)}
        answers={str(i):i%2 for i in range(40)}
        monitor=fit_monitor(predictions,answers)
        actions=choose_actions(predictions,monitor,123)
        self.assertEqual({len(x) for x in actions["selected"].values()},{20})
        self.assertEqual(actions,choose_actions(predictions,monitor,123))
        wrong={str(i):1-i%2 for i in range(40)}
        self.assertNotEqual(score_monitor(predictions,answers,monitor,actions)["base_accuracy"],
                            score_monitor(predictions,wrong,monitor,actions)["base_accuracy"])
        self.assertEqual(actions,choose_actions(predictions,monitor,123))

    def test_missing_calibration_outcome_fails(self):
        with self.assertRaises(ValueError):
            fit_monitor({"a":{"choice":0,"raw_confidence":0.9}},{})

    def test_unequal_action_budget_fails(self):
        predictions={str(i):{"choice":0,"raw_confidence":0.7} for i in range(40)}
        answers={str(i):i%2 for i in range(40)}
        monitor=fit_monitor(predictions,answers)
        actions=choose_actions(predictions,monitor,1)
        actions["selected"]["on"].pop()
        with self.assertRaises(ValueError):score_monitor(predictions,answers,monitor,actions)


if __name__=="__main__":unittest.main()
