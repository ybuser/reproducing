"""Resume-capable adaptive-only worker. Imports ML libraries only in execute()."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from .common import ROOT, FIXED, alive, append, call_metrics, canonical, checkpoint_complete, digest, isolation_env, lines, protect_fixed, read, save
from .resources import ResourcePause, ResourceMonitor, fixed_active, gpu_snapshot, guarded_agent_class


def load_frozen():
    identity = read(ROOT / "records/adaptive_frozen_identity.json")
    for name, expected in identity["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Frozen adaptive source/config changed: {name}; use a new run ID")
    current_packages = dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))
    if current_packages != identity["packages"]:
        raise RuntimeError("Shared read-only environment changed; refuse mixed run")
    return identity


def run_case(core, agent, identity, row, phase="cohort", tracer_class=None):
    config = identity["config"]
    run = ROOT / "runs" / config["run_version"]
    directory = run / phase / f"row-{row['row_id']:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    identity_hash = canonical(identity)
    if checkpoint_complete(directory, identity_hash):
        return read(directory / "result.json")
    # Same fixed-4 case seed, including reference condition TT; method is in cache identity.
    split = "official600" if phase == "cohort" else "smoke3"
    case_seed = int(hashlib.sha256(f"{config['seed']}|{split}|{row['row_id']}|TT".encode()).hexdigest()[:8], 16)
    core.seed_case(case_seed)
    call_identity = {"run_identity_sha256": identity_hash, "method": "adaptive-A", "config_sha256": identity["config_sha256"], "prompt_sha256": identity["prompt_sha256"], "manifest_sha256": identity["manifest_sha256"], "model_revision": config["model_revision"], "tokenizer_revision": config["tokenizer_revision"], "row_id": row["row_id"], "split": split, "condition": "adaptive-A", "case_seed": case_seed}
    agent.begin_case(directory, call_identity)
    agent.target_agent = None
    started = time.monotonic()
    result = {**call_identity, "case_attempt_id": agent.case_attempt_id, "status": "running", "qtype": row["qtype"], "stratum": row["stratum"], "blocking_format_errors": [], "tt_bypassed": not row.get("official_tt_enabled", True), "human_intermediate_review": "pending"}
    append(directory / "lifecycle.jsonl", {"event": "case_start", "time_unix": time.time(), **call_identity})
    try:
        original = core.model_prompt(row)
        trace = ""
        if not result["tt_bypassed"]:
            if tracer_class is None:
                from .tracer import AdaptiveTracer
                tracer_class = AdaptiveTracer
            args = SimpleNamespace(model=config["model_id"], tracing_model=config["model_id"], output_dir=str(directory), run_id=config["run_version"], n_hypotheses=config["max_initial_hypotheses"], use_helper_llm=False, use_perception_only=False, input_is_chat=False, dataset="tomi", print=False, likelihood_estimate="prompting", tracer_type="tracer", example_id=row["row_id"])
            tracer = tracer_class(args, directory, config)
            with (directory / "console.txt").open("a", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                trace = tracer.trace(original)
            if not trace:
                raise RuntimeError("Empty adaptive trace")
            result.update(target_agent=tracer.target_agent, initial_count=tracer.initial_count, final_count=tracer.history_snapshots[-1]["count_after_pruning"], trajectory_steps=len(tracer.history_snapshots), trace=trace)
            final_prompt = original.removesuffix("\nAnswer:") + f"\n\n{trace.strip()}\n\n{core.ANSWER_PROMPT}"
        else:
            result.update(initial_count=None, final_count=None, trajectory_steps=0, trace="", bypass_reason="Same fixed-4 reality/memory routing; no trajectory generated")
            final_prompt = original.replace("Answer:", core.ANSWER_PROMPT)
        agent.stage = "final"
        prediction = agent.interact(final_prompt, temperature=config["final_temperature"], max_tokens=config["final_max_tokens"])
        if not prediction:
            raise RuntimeError("Empty final answer")
        # Gold and candidate-location metadata are used here only, after all inference.
        result.update(status="ok", prediction=prediction, final_truncated=agent.last_call["truncated"], grading=core.score_final(prediction, row, agent.last_call["truncated"]))
    except ResourcePause:
        save(directory / "checkpoint.json", {**call_identity, "status": "paused_resource", "time_unix": time.time(), "completed_call_replay": "calls.jsonl; regenerate deterministic control flow from initial seed and restore per-call RNG"})
        append(directory / "lifecycle.jsonl", {"event": "case_resource_pause", "time_unix": time.time()})
        raise
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), grading={"official_substring": False, "strict_final_only": False, "human_review_required": True, "review_reasons": ["execution_error"]})
        if "Format" in type(exc).__name__:
            result["blocking_format_errors"].append(type(exc).__name__)
    result["elapsed_this_attempt_seconds"] = time.monotonic() - started
    result["metrics"] = call_metrics(directory)
    save(directory / "result.json", result)
    save(directory / "checkpoint.json", {**call_identity, "status": result["status"], "result_sha256": digest(directory / "result.json")})
    append(directory / "lifecycle.jsonl", {"event": "case_end", "time_unix": time.time(), "status": result["status"]})
    print(f"DONE adaptive-A {phase} row={row['row_id']} status={result['status']} calls={result['metrics']['calls']}", flush=True)
    return result


def ensure_initial_report(identity, reporter=None):
    config = identity["config"]
    limit = config["initial_report_items"]
    run = ROOT / "runs" / config["run_version"]
    marker = run / f"first{limit}_draft_ready.json"
    if marker.exists():
        saved = read(marker)
        if saved["run_identity_sha256"] != canonical(identity):
            raise RuntimeError("Draft identity mismatch")
        for name, expected in saved["report_sha256"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError("Saved draft changed; preserve and investigate")
        return saved
    rows = read(ROOT / "manifests/adaptive_cohort.json")["rows"][:limit]
    if len(rows) != limit or not all(checkpoint_complete(run / "cohort" / f"row-{row['row_id']:04d}", canonical(identity)) for row in rows):
        raise RuntimeError("First draft requires every prefix case to have a terminal result")
    if reporter is None:
        from .report import analyze
        reporter = analyze
    directory = ROOT / "reports" / f"first{limit}_draft"
    directory.mkdir(parents=True, exist_ok=True)
    summary = reporter(limit=limit, output_directory=directory)
    if summary["status"] != "complete" or summary["cohort_size"] != limit:
        raise RuntimeError("Prefix draft is incomplete; do not continue to remaining cases")
    saved = {"run_identity_sha256": canonical(identity), "method": "adaptive-A", "prefix_size": limit, "full_planned_items": config["cohort_size"], "example_ids_in_order": [row["row_id"] for row in rows], "report_path": str((directory / "COMPARISON_KO.md").relative_to(ROOT)), "report_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in directory.iterdir() if path.is_file()}, "time_unix": time.time(), "continue_without_parameter_tuning": True, "shutdown_requested": False}
    save(marker, saved)
    save(ROOT / "records/first30_draft_notice.json", saved)
    append(run / "run_lifecycle.jsonl", {"event": "initial_draft_saved_before_remaining_cases", **saved})
    print(f"FIRST_DRAFT_SAVED prefix={limit} path={saved['report_path']}; continuing remaining cases without configuration changes", flush=True)
    return saved


def execute():
    isolation_env()
    identity = load_frozen()
    protect_fixed("before_adaptive_gpu")
    if fixed_active():
        raise ResourcePause("Fixed-4 is still active; use the background queue")
    if gpu_snapshot()[0]["free_mib"] < 28 * 1024:
        raise ResourcePause("Less than 28GiB free before adaptive model loading")
    verified = read(ROOT / "records/model_verified.json")
    model_directory = Path(identity["config"]["model_path"])
    verified_files = []
    verification_started = time.monotonic()
    for entry in verified["files"]:
        path = model_directory / entry["file"]
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != entry["sha256"]:
            raise RuntimeError(f"Shared model/tokenizer file differs: {entry['file']}")
        verified_files.append(entry["file"])
    save(ROOT / "records/model_reverified_before_gpu.json", {"files": verified_files, "revision": verified["revision"], "seconds": time.monotonic() - verification_started, "passed": True})
    from repro import run as core
    config = identity["config"]
    run = ROOT / "runs" / config["run_version"]
    run.mkdir(parents=True, exist_ok=True)
    lock = run / "worker.lock"
    if lock.exists():
        if alive(read(lock)["pid"]):
            raise RuntimeError("An adaptive worker already owns this run")
        lock.rename(run / f"worker.lock.stale-{time.time_ns()}")
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_unix": time.time(), "method": "adaptive-A"}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    save(run / "run_identity.json", identity)
    append(run / "run_lifecycle.jsonl", {"event": "run_start", "time_unix": time.time(), "pid": os.getpid(), "fixed_overlap": False})
    core.torch.backends.cudnn.benchmark = False
    core.torch.backends.cudnn.deterministic = True
    core.torch.use_deterministic_algorithms(True, warn_only=True)
    final_status = "interrupted"
    try:
        with ResourceMonitor(run) as monitor:
            agent = guarded_agent_class()(config, ROOT, monitor)
            loaded = {"load_seconds": agent.load_seconds, "dtype": str(agent.model.dtype), "attention_implementation": agent.model.config._attn_implementation, "context_limit": agent.context_limit, "tokenizer_chat_template_sha256": hashlib.sha256(agent.tokenizer.chat_template.encode()).hexdigest(), "pid": os.getpid(), "time_unix": time.time(), "local_files_only": True, "model_path": config["model_path"], "quantization": config.get("quantization", "none")}
            if loaded["tokenizer_chat_template_sha256"] != identity["fixed_loaded"]["tokenizer_chat_template_sha256"]:
                raise RuntimeError("Chat template differs from fixed-4")
            save(run / "model_loaded.json", loaded)
            append(run / "model_loads.jsonl", loaded)
            agent.begin_case(run / "connectivity", {"run_identity_sha256": canonical(identity), "method": "adaptive-A", "condition": "connectivity", "model_revision": config["model_revision"]})
            core.seed_case(config["seed"])
            agent.stage = "connectivity"
            reply = agent.interact("Reply with exactly the word READY.", temperature=0, max_tokens=16)
            if not reply or agent.last_call["truncated"]:
                raise RuntimeError("Connectivity generation failed")
            smoke = read(ROOT / "manifests/adaptive_smoke.json")["rows"]
            for row in smoke:
                save(run / "progress.json", {"phase": "smoke", "pid": os.getpid(), "active_row_id": row["row_id"], "time_unix": time.time()})
                result = run_case(core, agent, identity, row, "smoke")
                if result["status"] != "ok" or result["blocking_format_errors"] or result["metrics"]["truncated_calls"]:
                    raise RuntimeError("Smoke execution/format gate failed; no accuracy-based tuning")
            smoke_events = [e for p in (run / "smoke").glob("row-*/trace_events.jsonl") for e in lines(p) if e["event"] == "adaptive_step"]
            if not any(e["evaluator_type"] == "state_only_plausibility" for e in smoke_events):
                raise RuntimeError("Smoke did not exercise state-only evaluation; preserve logs for review")
            save(run / "smoke_gate.json", {"passed": True, "model_inference_executed": True, "accuracy_used_for_gate": False, "false_belief_included": any(r["belief"] == "false" for r in smoke), "state_only_exercised": True, "case_ids": [r["row_id"] for r in smoke], "time_unix": time.time()})
            manifest = read(ROOT / "manifests/adaptive_cohort.json")
            consecutive_failures = 0
            for position, row in enumerate(manifest["rows"], 1):
                if position == config["initial_report_items"] + 1:
                    ensure_initial_report(identity)
                save(run / "progress.json", {"phase": "cohort", "pid": os.getpid(), "position_1based": position, "planned": len(manifest["rows"]), "active_row_id": row["row_id"], "time_unix": time.time()})
                result = run_case(core, agent, identity, row)
                consecutive_failures = consecutive_failures + 1 if result["status"] == "error" else 0
                save(run / "checkpoint.json", {"completed_ids": [r["row_id"] for r in manifest["rows"] if checkpoint_complete(run / "cohort" / f"row-{r['row_id']:04d}", canonical(identity))], "run_identity_sha256": canonical(identity), "planned": len(manifest["rows"]), "time_unix": time.time()})
                if consecutive_failures >= 3:
                    raise RuntimeError("Three consecutive adaptive execution failures; diagnostic stop")
                if position == config["initial_report_items"]:
                    ensure_initial_report(identity)
            final_status = "complete"
    except ResourcePause as exc:
        final_status = "waiting_resources"
        append(run / "run_lifecycle.jsonl", {"event": "resource_pause", "error": str(exc), "time_unix": time.time()})
        return 75
    except BaseException as exc:
        append(run / "run_lifecycle.jsonl", {"event": "run_error", "error_type": type(exc).__name__, "error": str(exc), "time_unix": time.time()})
        raise
    finally:
        save(run / "progress.json", {"phase": final_status, "pid": os.getpid(), "time_unix": time.time()})
        append(run / "run_lifecycle.jsonl", {"event": "run_end", "status": final_status, "time_unix": time.time()})
        if lock.exists() and read(lock)["pid"] == os.getpid():
            lock.rename(run / f"worker.lock.finished-{time.time_ns()}")
        protect_fixed("after_adaptive_worker")
        from .report import analyze
        analyze()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        try:
            raise SystemExit(execute())
        except ResourcePause as exc:
            print(str(exc), flush=True)
            raise SystemExit(75)
    identity = load_frozen()
    print(json.dumps({"validated": True, "method": "adaptive-A", "items": len(read(ROOT / "manifests/adaptive_cohort.json")["rows"]), "model_inference_executed": False, "run_identity_sha256": canonical(identity)}))


if __name__ == "__main__":
    main()
