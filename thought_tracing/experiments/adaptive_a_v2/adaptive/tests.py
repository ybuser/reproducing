"""Mock-only contracts; no model loading, CUDA initialization or network."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from .common import ROOT, canonical, isolation_env, read, save
isolation_env()
from .kernel import FormatError, parse_initial_list, parse_option, prune_once, softmax_scores, summary_indices
from .tracer import AdaptiveTracer
from repro import local_model
import torch


class FakeModel:
    def __init__(self, initial=None):
        self.config = {"model_id": "Qwen/Qwen2.5-7B-Instruct"}
        self.args = SimpleNamespace(model=self.config["model_id"], tracing_model=self.config["model_id"])
        self.stage = "mock"
        self.case_attempt_id = "mock-only"
        self.requests = []
        self.initial = initial or ["\n".join(f"{i}. candidate-{i}" for i in range(1, 6))]
        self.action_counter = self.state_counter = 0

    def interact(self, prompt, temperature=None, max_tokens=None, system_prompt=None, history=None):
        self.requests.append({"stage": self.stage, "prompt": prompt, "system_prompt": system_prompt, "max_tokens": max_tokens})
        if self.stage.startswith("tt.initialize"):
            return self.initial.pop(0)
        if self.stage == "tt.propagate":
            before = prompt.split("'s thoughts>\n", 1)[1].split("\n</previous prediction", 1)[0]
            return "updated " + before
        if self.stage == "tt.likelihood":
            option = ["a", "a", "e", "f", "f"][self.action_counter % 5]
            self.action_counter += 1
            return f"Mock action assessment. Answer: ({option})"
        if self.stage == "adaptive.state_only_plausibility":
            option = ["a", "f", "e"][self.state_counter % 3]
            self.state_counter += 1
            return f"Mock state assessment. Answer: ({option})"
        if self.stage == "tt.summary":
            return "Alex believes the item is in the basket."
        if self.stage == "tt.target":
            return "Alex"
        return "Alex knows only the previously observed basket."

    def batch_interact(self, prompts, temperature=None, max_tokens=None, system_prompts=None, **kwargs):
        systems = system_prompts if isinstance(system_prompts, list) else [system_prompts] * len(prompts)
        return [self.interact(p, temperature, max_tokens, s) for p, s in zip(prompts, systems)]


class Contracts(unittest.TestCase):
    def setUp(self):
        base = ROOT / "records/cpu_fixtures"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.directory = Path(self.temp.name).resolve()
        assert self.directory.is_relative_to(ROOT.resolve())
        self.old_shared = local_model.SHARED

    def tearDown(self):
        local_model.SHARED = self.old_shared
        self.temp.cleanup()

    def tracer(self, initial=None):
        fake = FakeModel(initial)
        local_model.SHARED = fake
        args = SimpleNamespace(model=fake.config["model_id"], tracing_model=fake.config["model_id"], output_dir=str(self.directory), run_id="mock-only", n_hypotheses=4, use_helper_llm=False, use_perception_only=False, input_is_chat=False, dataset="tomi", print=False, likelihood_estimate="prompting", tracer_type="tracer", example_id="mock-example")
        tracer = AdaptiveTracer(args, self.directory, {"summary_top_p": 0.9, "prune_threshold": 0.10})
        tracer.target_agent, tracer.assumption = "Alex", ""
        trajectory = [{"state": None, "action": "Alex enters the room."}, {"state": "The item moved while Alex was away.", "action": None}, {"state": "No new relevant information arrived.", "action": None}]
        perceptions = [{"state": None, "action": "Alex saw the basket."}, {"state": "Alex did not see a move.", "action": None}, {"state": "Alex received no new information.", "action": None}]
        tracer.preprocess_input = lambda text, target_agent=None: {"question": "none", "context": "mock context", "target_agent": "Alex", "trajectory": deepcopy(trajectory), "perceptions": deepcopy(perceptions), "assumption": ""}
        return tracer, fake

    def test_initial_counts_1_to_5(self):
        for n in range(1, 6):
            self.assertEqual(len(parse_initial_list("\n".join(f"{i}. item {i}" for i in range(1, n + 1)))), n)

    def test_initial_format_errors_not_truncated_or_padded(self):
        for raw in ["", "plain text", "1. ", "2. item", "1. a\n1. b", "\n".join(f"{i}. a" for i in range(1, 7)), "Intro\n1. item"]:
            with self.assertRaises(FormatError): parse_initial_list(raw)

    def test_exact_threshold_survives(self):
        value = prune_once(["a", "b", "c"], [0.10, 0.09, 0.81])
        self.assertEqual(value["kept_ids"], ["a", "c"])
        self.assertAlmostEqual(sum(value["weights_after"]), 1)
        self.assertAlmostEqual(value["weights_after"][0], 0.1 / 0.91)

    def test_minimum_one_and_stable_tie(self):
        value = prune_once(["first", "second"], [0.5, 0.5], 0.9)
        self.assertEqual(value["kept_ids"], ["first"])
        self.assertEqual(value["weights_after"], [1])
        self.assertTrue(value["minimum_one_guard"])

    def test_invalid_scores_and_lengths(self):
        for scores, n in [([float("nan")], 1), ([float("inf")], 1), ([1], 2), ([], 0)]:
            with self.assertRaises(ValueError): softmax_scores(scores, n)
        with self.assertRaises(ValueError): prune_once(["a"], [float("nan")])

    def test_softmax_replaces_prior(self):
        self.assertEqual(softmax_scores([2.5, 2.5], 2), [0.5, 0.5])

    def test_original_option_parser_and_mapping(self):
        for option in "abcdef":
            self.assertEqual(parse_option(f"reasoning Answer: ({option}) label"), option)
        with self.assertRaises(FormatError): parse_option("no valid choice")

    def test_one_repair_and_actual_count(self):
        tracer, fake = self.tracer(["invalid", "1. only one"])
        h = tracer.initialize({"state": None, "action": "Alex enters."}, {"state": None, "action": "Alex sees the room."})
        self.assertEqual(len(h.texts), 1)
        self.assertEqual(list(h.weights), [1])
        self.assertEqual(sum(r["stage"].startswith("tt.initialize") for r in fake.requests), 2)

    def test_repair_failure_terminal_after_two_calls(self):
        tracer, fake = self.tracer(["invalid", "still invalid"])
        with self.assertRaises(FormatError): tracer.initialize({"state": None, "action": "enter"}, {"state": None, "action": "view"})
        self.assertEqual(len(fake.requests), 2)

    def test_initial_prompt_scope_and_state_question(self):
        tracer, fake = self.tracer(["1. only supported belief"])
        tracer.initialize({"state": "Alex was in the room.", "action": None}, {"state": "Alex saw a basket.", "action": None})
        query = next(r["prompt"] for r in fake.requests if r["stage"] == "tt.initialize")
        self.assertIn("will be thinking", query)
        self.assertNotIn("led to the action", query)
        self.assertNotIn("The item moved", query)
        self.assertNotIn("No new relevant information arrived", query)
        self.assertEqual(sum(r["stage"] == "tt.initialize" for r in fake.requests), 1)

    def test_summary_top_p_separate_from_survivors(self):
        weights = [0.4, 0.2, 0.2, 0.1, 0.1]
        pruning = prune_once(list("abcde"), weights, 0.10)
        self.assertEqual(len(pruning["kept_ids"]), 5)
        self.assertEqual(len(summary_indices(weights, 0.9)), 4)

    def test_full_event_flow_ids_pruning_and_no_resampling(self):
        tracer, fake = self.tracer()
        with patch("tracer.resample_hypotheses_with_other_info", side_effect=AssertionError("forbidden resampling")), patch.object(tracer, "rejuvenate_hypotheses", side_effect=AssertionError("forbidden rejuvenation")):
            output = tracer.trace("mock full input")
        self.assertTrue(output)
        steps = tracer.history_snapshots
        self.assertEqual([e["count_after_pruning"] for e in steps], [2, 1, 1])
        self.assertEqual([e["evaluator_type"] for e in steps], ["action_likelihood", "state_only_plausibility", "state_only_plausibility"])
        dead = set()
        for event in steps:
            self.assertFalse(dead.intersection(event["ids_before"]))
            self.assertEqual(len(event["ids_before"]), len(event["texts_before"]))
            self.assertEqual(len(event["ids_before"]), len(event["raw_scores"]))
            self.assertEqual(len(event["ids_after"]), len(event["weights_after_pruning"]))
            self.assertAlmostEqual(sum(event["weights_after_pruning"]), 1)
            dead.update(event["deleted_ids"])
        self.assertEqual(steps[-1]["weights_after_pruning"], [1])
        self.assertEqual(steps[-1]["hypothesis_evaluations"], 1)
        propagated = [r["prompt"] for r in fake.requests if r["stage"] == "tt.propagate"]
        self.assertTrue(all("candidate-3" not in p and "candidate-4" not in p and "candidate-5" not in p for p in propagated))
        state_requests = [r for r in fake.requests if r["stage"] == "adaptive.state_only_plausibility"]
        self.assertTrue(all("actual world state" in r["system_prompt"] for r in state_requests))

    def test_past_objects_and_records_immutable(self):
        tracer, _ = self.tracer()
        tracer.trace("mock")
        first_snapshot = deepcopy(tracer.history_snapshots[0])
        first = tracer.stage_sets[0]
        first_values = (list(first.texts), first.weights.copy(), deepcopy(first.contexts))
        last = tracer.stage_sets[-1]
        last.texts[0] = "changed only last object"
        last.weights[0] = 0.123
        last.contexts[0]["action"] = "changed last context"
        self.assertEqual(tracer.history_snapshots[0], first_snapshot)
        self.assertEqual(first.texts, first_values[0])
        self.assertEqual(list(first.weights), list(first_values[1]))
        self.assertEqual(first.contexts, first_values[2])

    def test_completed_case_resume_does_not_call_again(self):
        from . import runner
        class Agent:
            def __init__(self): self.calls = 0
            def begin_case(self, directory, identity): self.case_attempt_id = "mock-case"
            def interact(self, *args, **kwargs):
                self.calls += 1
                self.last_call = {"truncated": False}
                return "basket"
        agent = Agent()
        config = {"run_version": "mock_resume", "seed": 20261020, "model_revision": "pinned", "tokenizer_revision": "pinned", "final_temperature": 0, "final_max_tokens": 256}
        identity = {"config": config, "config_sha256": "mock", "prompt_sha256": "mock", "manifest_sha256": "mock"}
        core = SimpleNamespace(seed_case=lambda value: None, model_prompt=lambda row: "story\nAnswer:", ANSWER_PROMPT="final", score_final=lambda *args: {"official_substring": True, "strict_final_only": True})
        row = {"row_id": 1, "official_tt_enabled": False, "qtype": "reality", "stratum": "reality"}
        with patch.object(runner, "ROOT", self.directory):
            runner.run_case(core, agent, identity, row)
            runner.run_case(core, agent, identity, row)
        self.assertEqual(agent.calls, 1)

    def test_comparison_incomplete_and_failed_denominators(self):
        from . import report
        adaptive_root, fixed_root = self.directory / "adaptive", self.directory / "fixed"
        rows = [{"row_id": i, "stratum": "order1_false", "qtype": "first_order_0_tom", "answer": "basket"} for i in (1, 2)]
        identity = {"config": {"run_version": "mock", "model_id": "mock", "model_revision": "mock"}}
        save(adaptive_root / "records/adaptive_frozen_identity.json", identity)
        save(adaptive_root / "manifests/adaptive_cohort.json", {"rows": rows, "fixed_manifest_sha256": "source", "selection_policy": "mock fixture only"})
        for rid in (1, 2):
            base = {"row_id": rid, "case_attempt_id": "mock", "status": "ok", "manifest_sha256": "source", "metrics": {}, "grading": {"official_substring": True, "strict_final_only": True}}
            save(fixed_root / "runs/bf16_official600_v1/official600" / f"row-{rid:04d}" / "TT/result.json", base)
        first = {"row_id": 1, "case_attempt_id": "mock", "status": "ok", "run_identity_sha256": canonical(identity), "metrics": {}, "grading": {"official_substring": True, "strict_final_only": True}}
        save(adaptive_root / "runs/mock/cohort/row-0001/result.json", first)
        with patch.object(report, "ROOT", adaptive_root), patch.object(report, "FIXED", fixed_root):
            partial = report.analyze()
            self.assertEqual(partial["status"], "incomplete")
            self.assertIsNone(partial["methods_all_attempts"]["adaptive-A"]["all"]["official_accuracy"])
            prefix = report.analyze(limit=1, output_directory=adaptive_root / "reports/prefix_draft")
            self.assertEqual(prefix["cohort_size"], 1)
            self.assertEqual(prefix["full_experiment_planned_items"], 2)
            self.assertEqual(prefix["reported_ids_in_manifest_order"], [1])
            self.assertEqual(prefix["status"], "complete")
            failure = {**first, "row_id": 2, "status": "error", "grading": {"official_substring": False, "strict_final_only": False}}
            save(adaptive_root / "runs/mock/cohort/row-0002/result.json", failure)
            final = report.analyze()
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["methods_all_attempts"]["adaptive-A"]["all"]["official_accuracy"], 0.5)
        self.assertEqual(final["paired_normal_completed_count"], 1)

    def test_prepare_freeze_in_temporary_mock_workspace(self):
        from . import prepare
        scratch = self.directory / "prepare_mock"
        for name in ["records/fixed4_run_reference/run_identity.json", "records/fixed4_run_reference/model_loaded.json", "records/fixed_process_commands.json", "manifests/smoke3.json", "official/tracer.py"]:
            target = scratch / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        save(scratch / "records/cpu_validation.json", {"passed": True, "model_inference_calls": 0})
        (scratch / "configs").mkdir()
        with patch.object(prepare, "ROOT", scratch), patch.object(prepare, "protect_fixed", lambda label: {"passed": True}):
            prepare.freeze("prefix200")
        frozen = read(scratch / "records/adaptive_frozen_identity.json")
        cohort = read(scratch / "manifests/adaptive_cohort.json")
        master = read(ROOT / "manifests/tomi_official_defaults600_v1.json")
        self.assertEqual(cohort["rows"], master["rows"][:200])
        self.assertEqual(frozen["config"]["prune_threshold"], 0.10)
        self.assertEqual(frozen["config"]["summary_top_p"], 0.9)
        self.assertEqual(frozen["config"]["resampling"], "disabled")
        self.assertEqual(frozen["config"]["dtype"], "bfloat16")
        self.assertEqual(frozen["config"]["initial_report_items"], 30)
        self.assertFalse(frozen["config"]["shutdown_after_all_work"])
        self.assertTrue((scratch / "records/adaptive_frozen_identity.json").exists())

    def test_initial_draft_barrier_and_resume(self):
        from . import runner
        identity = {"config": {"run_version": "mock_staged", "initial_report_items": 2, "cohort_size": 3}}
        scratch = self.directory / "staged"
        rows = [{"row_id": n} for n in (1, 2, 3)]
        save(scratch / "manifests/adaptive_cohort.json", {"rows": rows})
        calls = []
        def reporter(limit, output_directory):
            calls.append(limit)
            (output_directory / "COMPARISON_KO.md").write_text("mock draft", encoding="utf-8")
            save(output_directory / "comparison.json", {"cohort_size": limit})
            return {"status": "complete", "cohort_size": limit}
        with patch.object(runner, "ROOT", scratch):
            with self.assertRaises(RuntimeError): runner.ensure_initial_report(identity, reporter)
            self.assertEqual(calls, [])
            for n in (1, 2):
                save(scratch / f"runs/mock_staged/cohort/row-{n:04d}/result.json", {"run_identity_sha256": canonical(identity), "status": "ok"})
            saved = runner.ensure_initial_report(identity, reporter)
            self.assertEqual(saved["example_ids_in_order"], [1, 2])
            self.assertFalse(saved["shutdown_requested"])
            runner.ensure_initial_report(identity, reporter)
            self.assertEqual(calls, [2])


    def test_fixed_guard_distinguishes_pid_reuse_and_resume_lock(self):
        from . import common, resources
        scratch = self.directory / "process_identity"
        fixed = scratch / "fixed"
        save(scratch / "records/fixed_process_commands.json", [{"pid": 32856, "created": "/Date(1792974648799)/"}])
        instances = {32856: {"alive": True, "creation_unix": 1792974648.799}}
        def probe(pid):
            return instances.get(pid, {"alive": False, "creation_unix": None})
        with patch.object(resources, "ROOT", scratch), patch.object(common, "FIXED", fixed):
            self.assertEqual(resources.fixed_active(probe), [32856])
            instances[32856]["creation_unix"] += 3600
            self.assertEqual(resources.fixed_active(probe), [])
            instances[32856]["creation_unix"] = None
            self.assertEqual(resources.fixed_active(probe), [32856])
            instances[32856]["alive"] = False
            lock = fixed / "runs/bf16_official600_v1/worker.lock"
            save(lock, {"pid": 42})
            instances[42] = {"alive": True, "creation_unix": 2000}
            os.utime(lock, (1000, 1000))
            self.assertEqual(resources.fixed_active(probe), [])
            os.utime(lock, (3000, 3000))
            self.assertEqual(resources.fixed_active(probe), [42])

    def test_native_process_identity_is_read_only(self):
        from .resources import process_identity
        current = process_identity(os.getpid())
        self.assertTrue(current["alive"])
        self.assertGreater(current["creation_unix"], 0)
        self.assertLessEqual(current["creation_unix"], time.time() + 1)
        self.assertFalse(process_identity(0)["alive"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "model_inference_calls": 0, "cuda_initialized": torch.cuda.is_initialized(), "mock_only": True, "time_unix": time.time()}
    if receipt["cuda_initialized"]:
        receipt["passed"] = False
    save(ROOT / "records/cpu_validation.json", receipt)
    raise SystemExit(0 if receipt["passed"] else 1)
