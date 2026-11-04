"""Freeze an explicitly confirmed comparison cohort; never selects by predictions."""
import argparse
import ast
import difflib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

from .common import ROOT, FIXED, canonical, digest, protect_fixed, read, save
from .kernel import SCORE_MAPPING


def freeze(selection):
    frozen_path = ROOT / "records/adaptive_frozen_identity.json"
    if frozen_path.exists():
        raise RuntimeError("Run already frozen; use a new directory/run ID for changes")
    protect_fixed("before_freeze")
    tests = read(ROOT / "records/cpu_validation.json")
    if not tests["passed"] or tests["model_inference_calls"] != 0:
        raise RuntimeError("CPU validation is not complete")
    fixed = read(ROOT / "records/fixed4_run_reference/run_identity.json")
    fixed_manifest_path = FIXED / fixed["config"]["manifest_path"]
    assert digest(fixed_manifest_path) == fixed["manifest_sha256"]
    master = read(fixed_manifest_path)
    if selection == "prefix200":
        chosen = master["rows"][:200]
        policy = "User-confirmed first 200 rows in the existing fixed600 manifest, unchanged order; newly defined comparison subset, not a pre-existing fixed200 run"
    elif selection == "all600":
        chosen = master["rows"]
        policy = "User-confirmed extension to all 600 rows of the existing fixed600 manifest"
    else:
        provided = read(Path(selection))
        requested = provided["rows"]
        assert len(requested) == 200
        by_id = {row["row_id"]: row for row in master["rows"]}
        chosen = [by_id[row["row_id"]] for row in requested]
        assert len({r["row_id"] for r in chosen}) == 200
        for original, selected in zip(requested, chosen):
            assert all(original[key] == selected[key] for key in ("story", "question", "answer", "qtype"))
        policy = f"User-specified existing 200-row manifest; hash {digest(selection)}; same IDs/order and fixed-source inputs"
    assert len(chosen) in (200, 600)
    manifest = {"selection_policy": policy, "fixed_manifest_sha256": fixed["manifest_sha256"], "fixed_manifest_path": str(fixed_manifest_path), "rows": chosen, "model_outputs_used_for_selection": False, "author_paper_run_manifest_verified": False}
    save(ROOT / "manifests/adaptive_cohort.json", manifest)
    original_inputs = {r["row_id"]: r for r in [json.loads(line) for line in (FIXED / "manifests/tomi_official_defaults600_v1.model_inputs.jsonl").read_text(encoding="utf-8").splitlines()]}
    with (ROOT / "manifests/adaptive_cohort.model_inputs.jsonl").open("x", encoding="utf-8") as stream:
        for row in chosen:
            original = original_inputs[row["row_id"]]
            assert original["prompt"] == f"{row['story'].strip('.')}.\n\nQuestion: {row['question']}\nAnswer:"
            stream.write(json.dumps(original, ensure_ascii=False) + "\n")
    smoke_rows = read(ROOT / "manifests/smoke3.json")["rows"]
    smoke = [dict(row, official_tt_enabled=True) for row in smoke_rows if row["row_id"] in (1906, 2097)]
    assert len(smoke) == 2 and any(row["belief"] == "false" for row in smoke)
    assert not {row["row_id"] for row in smoke}.intersection(row["row_id"] for row in chosen)
    save(ROOT / "manifests/adaptive_smoke.json", {"purpose": "format/execution validation only; no accuracy-based tuning", "rows": smoke})
    tree = ast.parse((ROOT / "official/tracer.py").read_text(encoding="utf-8"))
    tracer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Tracer")
    likelihood = next(node for node in tracer.body if isinstance(node, ast.FunctionDef) and node.name == "prompt_likelihood")
    mapping = next(ast.literal_eval(node.value) for node in ast.walk(likelihood) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "score_mapping" for t in node.targets))
    assert mapping == SCORE_MAPPING
    summary = next(node for node in tracer.body if isinstance(node, ast.FunctionDef) and node.name == "weighted_average_hypotheses")
    top_p = ast.literal_eval(summary.args.defaults[0])
    assert top_p == 0.9 and "summary_top_p" not in fixed["config"]
    config = dict(fixed["config"])
    config.pop("n_hypotheses", None)
    config.update(run_version="adaptive_a_v2", method="adaptive-A", model_path=str((FIXED / fixed["config"]["model_path"]).resolve()), min_hypotheses=1, max_initial_hypotheses=5, prune_threshold=0.10, prune_comparison="strictly_less_than", resampling="disabled", rejuvenation="disabled", summary_top_p=top_p, initial_format_retry_limit=1, state_only_evaluator="plausibility_from_agent_perspective", quantization=fixed["config"].get("quantization", "none"), manifest_path="manifests/adaptive_cohort.json", manifest_sha256=digest(ROOT / "manifests/adaptive_cohort.json"), protocol_path="records/adaptive_frozen_identity.json", seed_policy="Same fixed seed derivation: SHA256(global_seed|official600|row_id|TT); method/config/prompt identity separates caches; different call/RNG consumption may change outputs", cache_policy="Independent adaptive run/config/prompt/model/case/stage/request identity; no fixed cache reuse", algorithm_changes=["variable initial count 1..5", "strict individual-weight threshold pruning after every trajectory step", "no resampling or rejuvenation", "state-only plausibility evaluation added"], compatibility_changes=["Read-only existing interpreter/site-packages; independent copied local adapter and DLLs", "Owned resource guard/checkpoints and local logs"], cohort_size=len(chosen), fixed_factual_routing="retain reality/memory TT bypass; all generated trajectories are evaluated at every step")
    config.update(initial_report_items=30, continue_after_initial_report=True, shutdown_after_all_work=False)
    config["compatibility_changes"].append("A v2: identify fixed processes by PID plus creation time and honor a current fixed resume lock; avoid pauses from reused PIDs")
    save(ROOT / "configs/adaptive_a_v2.json", config)
    sources = sorted({*ROOT.glob("adaptive/**/*.py"), *ROOT.glob("adaptive/prompts/*.txt"), *ROOT.glob("repro/*.py"), *ROOT.glob("official/**/*.py"), *ROOT.glob("official/prompt_templates/*.txt"), ROOT / "configs/adaptive_a_v2.json", ROOT / "manifests/adaptive_cohort.json", ROOT / "manifests/adaptive_smoke.json", ROOT / "manifests/adaptive_cohort.model_inputs.jsonl", ROOT / "records/fixed_process_commands.json"})
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sources}
    prompts = {name: value for name, value in hashes.items() if name.startswith("adaptive/prompts/") or name.startswith("official/prompt_templates/")}
    identity = {"config": config, "config_sha256": canonical(config), "prompt_sha256": canonical(prompts), "source_sha256": hashes, "manifest_sha256": config["manifest_sha256"], "fixed_run_identity_sha256": canonical(fixed), "fixed_reference_commit": fixed["config"]["reference_commit"], "fixed_loaded": read(ROOT / "records/fixed4_run_reference/model_loaded.json"), "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())), "python": sys.version, "gpu_concurrency_policy": "No overlap with fixed-4; wait for its processes and all 1800 results", "state_only_and_action_scores_not_calibrated": True}
    save(frozen_path, identity)
    additions = [path for path in sources if str(path.relative_to(ROOT)).replace("\\", "/").startswith(("adaptive/", "configs/adaptive_"))]
    patch_text = "".join("".join(difflib.unified_diff([], path.read_text(encoding="utf-8").splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + str(path.relative_to(ROOT)).replace("\\", "/"))) for path in additions)
    (ROOT / "records/adaptive_additions.patch").write_text(patch_text, encoding="utf-8")
    for path in sources:
        target = ROOT / "records/adaptive_source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(json.dumps({"frozen": True, "items": len(chosen), "run_identity_sha256": canonical(identity), "fixed_source_unchanged": True, "model_inference_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmed-cohort", required=True, help="prefix200, all600, or a user-specified manifest path")
    freeze(parser.parse_args().confirmed_cohort)
