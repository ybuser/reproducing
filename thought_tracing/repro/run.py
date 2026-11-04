from . import bootstrap  # must precede Torch and Transformers

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = bootstrap.ROOT
sys.path.insert(0, str(ROOT / "official"))
from .data import model_prompt
from .local_model import LocalTransformersAgent, append_json, canonical_hash, read_jsonl
from .scoring import score_final
from .tracing import RecordingTracer, record_resampling

ANSWER_PROMPT = "The short one-sentence answer specifying the most detailed location including both the container and the place (e.g., from A in B) without any explanation is:"


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(path.name + f".tmp-{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows scanners/readers may briefly deny FILE_SHARE_DELETE after close.
    # Preserve the complete temp file and retry only sharing/lock violations;
    # access-policy errors and persistent failures still propagate unchanged.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (32, 33) or attempt == 9:
                raise
            print(f"LOCAL_IO_RETRY file={path.name} winerror={exc.winerror} attempt={attempt + 1}", flush=True)
            time.sleep(min(0.1 * 2**attempt, 2.0))


def progress(**values):
    path = ROOT / "records/progress.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(values, last_update_unix=time.time())
    write_json(path, current)


def code_digest():
    paths = sorted([*ROOT.glob("repro/*.py"), *ROOT.glob("official/**/*.py"), *ROOT.glob("official/prompt_templates/*.txt")])
    return canonical_hash({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def seed_case(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def call_metrics(directory):
    records = read_jsonl(directory / "calls.jsonl") if (directory / "calls.jsonl").exists() else []
    ends = [r for r in records if r["event"] == "call_end"]
    good = [r for r in ends if r["status"] == "ok"]
    return {"calls": len(ends), "successful_calls": len(good), "failed_calls": len(ends) - len(good), "input_tokens": sum(r["input_tokens"] for r in good), "output_tokens": sum(r["output_tokens"] for r in good), "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in good), "call_wall_seconds": sum(r["wall_seconds"] for r in ends), "truncated_calls": sum(r["truncated"] for r in good), "peak_allocated_bytes": max((r.get("peak_allocated_bytes", 0) for r in ends), default=0), "peak_reserved_bytes": max((r.get("peak_reserved_bytes", 0) for r in ends), default=0)}


class GPUMonitor:
    def __init__(self, directory):
        self.path, self.stop = directory / "gpu_samples.jsonl", threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                append_json(self.path, {"time_unix": time.time(), "values_mib_mib_percent": result.stdout.strip(), "returncode": result.returncode})
            except Exception as exc:
                append_json(self.path, {"time_unix": time.time(), "error_type": type(exc).__name__})
            self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=8)


def run_case(agent, config, run_dir, run_identity, split, row, condition, manifest_sha, retry_failed):
    directory = run_dir / split / f"row-{row['row_id']:04d}" / condition
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if old["status"] == "ok" or not retry_failed:
            print(f"SKIP {split} row={row['row_id']} {condition} status={old['status']}", flush=True)
            return old
        history = directory / f"result.failed-{time.time_ns()}.json"
        result_path.rename(history)
    case_seed = int(hashlib.sha256(f"{config['seed']}|{split}|{row['row_id']}|{condition}".encode()).hexdigest()[:8], 16)
    seed_case(case_seed)
    identity = {"run_identity_sha256": canonical_hash(run_identity), "manifest_sha256": manifest_sha, "split": split, "row_id": row["row_id"], "condition": condition, "case_seed": case_seed, "model_revision": config["model_revision"], "tokenizer_revision": config["tokenizer_revision"]}
    agent.begin_case(directory, identity)
    agent.target_agent = None
    result = {**identity, "case_attempt_id": agent.case_attempt_id, "status": "running", "qtype": row["qtype"], "stratum": row["stratum"], "answer": row["answer"], "blocking_format_errors": [], "intermediate_semantic_review": "not_reviewed"}
    append_json(directory / "lifecycle.jsonl", {"event": "case_start", "time_unix": time.time(), **identity})
    started = time.monotonic()
    try:
        prompt = model_prompt(row)
        if condition == "Base":
            final_prompt = prompt.replace("Answer:", ANSWER_PROMPT)
        elif condition == "CoT":
            cot_prompt = prompt.replace("Answer:", "").strip() + "\nLet's think step by step.\n"
            agent.stage = "cot.reasoning"
            response = agent.interact(cot_prompt, temperature=config["cot_temperature"], max_tokens=config["cot_max_tokens"])
            result["cot"] = response
            final_prompt = cot_prompt.strip() + f"\n\n{response}\n\n{ANSWER_PROMPT}"
        elif condition == "TT":
            args = SimpleNamespace(model=config["model_id"], tracing_model=config["model_id"], output_dir=str(directory), run_id=config["run_version"], n_hypotheses=4, use_helper_llm=False, use_perception_only=False, input_is_chat=False, dataset="tomi", print=False, likelihood_estimate="prompting", tracer_type="tracer")
            tracer = RecordingTracer(args, directory)
            with (directory / "official_console.txt").open("a", encoding="utf-8") as console, contextlib.redirect_stdout(console), contextlib.redirect_stderr(console), record_resampling(tracer):
                trace = tracer.batch_trace([prompt], [True])[0]
            result["trace"] = trace
            if not trace:
                result["blocking_format_errors"].append("empty_trace")
            trace_events = read_jsonl(directory / "trace_events.jsonl")
            if any(e.get("format_error") for e in trace_events):
                result["blocking_format_errors"].append("likelihood_option_parse")
            final_prompt = prompt.removesuffix("\nAnswer:") + f"\n\n{trace.strip()}\n\n{ANSWER_PROMPT}"
        else:
            raise ValueError(condition)
        agent.stage = "final"
        prediction = agent.interact(final_prompt, temperature=config["final_temperature"], max_tokens=config["final_max_tokens"])
        result.update(status="ok", prediction=prediction, final_truncated=agent.last_call["truncated"], grading=score_final(prediction, row, agent.last_call["truncated"]))
        if not prediction:
            result["blocking_format_errors"].append("empty_final")
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), grading={"official_substring": False, "strict_final_only": False, "human_review_required": True, "review_reasons": ["execution_error"]})
    result["elapsed_this_attempt_seconds"] = time.monotonic() - started
    result["metrics"] = call_metrics(directory)
    write_json(result_path, result)
    progress(phase=split, run_version=config["run_version"], last_row_id=row["row_id"], last_condition=condition, last_case_status=result["status"], inference_executed=True, completed_case_conditions=len(list((run_dir / split).glob("row-*/*/result.json"))))
    append_json(directory / "lifecycle.jsonl", {"event": "case_end", "time_unix": time.time(), "status": result["status"], "metrics": result["metrics"]})
    print(f"DONE {split} row={row['row_id']} {condition} status={result['status']} calls={result['metrics']['calls']} tokens={result['metrics']['total_tokens']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bf16_v2.json")
    parser.add_argument("--phase", choices=["connectivity", "smoke", "pilot", "all"], default="all")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--wait-for-pilot-release", action="store_true", help="Keep the shared model loaded while smoke logs are inspected; release with runs/<version>/pilot.release")
    cli = parser.parse_args()
    config = json.loads((ROOT / cli.config).read_text(encoding="utf-8"))
    if config["model_id"] != "Qwen/Qwen2.5-7B-Instruct" or config["model_revision"] != "a09a35458c702b33eeacc393d103063234e8bc28" or config["tokenizer_revision"] != config["model_revision"]:
        raise ValueError("This reproduction permits only the explicitly pinned local Qwen model/tokenizer")
    if config.get("use_helper_llm") or config.get("use_perception_only"):
        raise ValueError("Helper APIs and algorithm ablations are outside this fixed run")
    assert config["n_hypotheses"] == 4 and config["batch_size"] == 1 and not config["algorithm_changes"]
    run_dir = ROOT / "runs" / config["run_version"]
    run_dir.mkdir(parents=True, exist_ok=True)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    identity = {"config": config, "code_sha256": code_digest(), "packages": dict(sorted(packages.items())), "python": sys.version, "platform": platform.platform(), "torch_cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    identity_path = run_dir / "run_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise RuntimeError("Code/config/environment changed; preserve results and use a new run_version")
    write_json(identity_path, identity)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    with GPUMonitor(run_dir):
        agent = LocalTransformersAgent(config, ROOT)
        write_json(run_dir / "model_loaded.json", {"load_seconds": agent.load_seconds, "dtype": str(agent.model.dtype), "attention_implementation": agent.model.config._attn_implementation, "context_limit": agent.context_limit, "tokenizer_chat_template_sha256": hashlib.sha256(agent.tokenizer.chat_template.encode()).hexdigest(), "model_generation_config": agent.model.generation_config.to_dict(), "actual_generation": "explicit per-call GenerationConfig; see calls.jsonl", "local_files_only": True, "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(), "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled()})
        seed_case(config["seed"])
        agent.begin_case(run_dir / "connectivity", {"run_identity_sha256": canonical_hash(identity), "condition": "connectivity"})
        agent.stage = "connectivity"
        reply = agent.interact("Reply with exactly the word READY.", temperature=0, max_tokens=16)
        write_json(run_dir / "connectivity/result.json", {"reply": reply, "metrics": call_metrics(run_dir / "connectivity")})
        if not reply or agent.last_call["truncated"]:
            raise RuntimeError("Short generation failed or was truncated")
        print(f"CONNECTIVITY {reply!r}; model loaded once in {agent.load_seconds:.1f}s", flush=True)
        progress(phase="connectivity_complete", inference_executed=True, run_version=config["run_version"])
        if cli.phase == "connectivity":
            return
        smoke_path = ROOT / "manifests/smoke3.json"
        smoke_sha = hashlib.sha256(smoke_path.read_bytes()).hexdigest()
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if cli.phase in {"smoke", "all"}:
            for row in smoke["rows"]:
                for condition in ["Base", "CoT", "TT"]:
                    run_case(agent, config, run_dir, identity, "smoke3", row, condition, smoke_sha, cli.retry_failed)
        smoke_results = [json.loads(p.read_text(encoding="utf-8")) for p in (run_dir / "smoke3").glob("row-*/*/result.json")]
        gate_ok = len(smoke_results) == 9 and all(r["status"] == "ok" and not r["blocking_format_errors"] for r in smoke_results)
        metrics = {name: sum(r["metrics"][name] for r in smoke_results) for name in ["calls", "input_tokens", "output_tokens", "total_tokens", "call_wall_seconds", "truncated_calls"]}
        estimate = {"gate_passed": gate_ok, "completed_case_conditions": len(smoke_results), "measured": metrics, "pilot20_linear_estimate": {k: v * 20 / 3 for k, v in metrics.items()}, "caution": "Length/action counts differ by case; linear estimate is approximate. No accuracy-based tuning. Truncation is logged without increasing official limits."}
        write_json(run_dir / "smoke_estimate.json", estimate)
        if cli.phase in {"pilot", "all"}:
            if not gate_ok:
                raise RuntimeError("Smoke execution/format gate not passed; pilot not started")
            if cli.wait_for_pilot_release:
                print("SMOKE READY: inspect smoke_estimate.json and raw logs, then create pilot.release", flush=True)
                progress(phase="smoke_review", smoke_gate_passed=True, pilot_started=False)
                while not (run_dir / "pilot.release").exists():
                    time.sleep(2)
            path = ROOT / "manifests/pilot20.json"
            manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for row in manifest["rows"]:
                for condition in ["Base", "CoT", "TT"]:
                    run_case(agent, config, run_dir, identity, "pilot20", row, condition, manifest_sha, cli.retry_failed)
    print("RUN PHASE FINISHED; raw logs and per-case results saved", flush=True)
    progress(phase="inference_phase_finished", requested_phase=cli.phase, analysis_pending=True)


if __name__ == "__main__":
    main()
