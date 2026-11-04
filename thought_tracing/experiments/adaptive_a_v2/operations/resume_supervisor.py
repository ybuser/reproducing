"""External queue supervision; no changes to frozen inference or experiment settings."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive.common import append, canonical, digest, isolation_env, protect_fixed, read, save
from adaptive.resources import process_identity
from adaptive.runner import load_frozen


def resource_query_timeout(status, launched_unix):
    return (
        status.get("time_unix", 0) >= launched_unix
        and status.get("phase") == "blocked_queue_error"
        and status.get("error_type") == "TimeoutExpired"
        and "nvidia-smi" in status.get("error", "")
    )


def active_lock(path):
    if not path.exists():
        return False
    saved = read(path)
    current = process_identity(saved["pid"])
    expected = saved.get("creation_unix", saved.get("started_unix"))
    return current["alive"] and (
        current["creation_unix"] is None
        or expected is None
        or current["creation_unix"] <= expected + 1
    )


def run():
    isolation_env()
    from operations.resume_entry import install
    install()
    lock = ROOT / "records/supervisor.lock"
    if active_lock(lock):
        raise RuntimeError("An adaptive supervisor is already active")
    if active_lock(ROOT / "records/queue.lock"):
        raise RuntimeError("A queue is already active; do not duplicate or replace it")
    if active_lock(ROOT / "runs/adaptive_a_v2/worker.lock"):
        raise RuntimeError("An adaptive worker is already active; do not duplicate it")
    if lock.exists():
        lock.rename(lock.with_name(f"supervisor.lock.stale-{time.time_ns()}"))
    started = time.time()
    saved = {"pid": os.getpid(), "started_unix": started, **process_identity(os.getpid())}
    with lock.open("x", encoding="utf-8") as stream:
        import json
        json.dump(saved, stream)
        stream.flush()
        os.fsync(stream.fileno())
    base = {"supervisor_pid": os.getpid(), "supervisor_creation_unix": saved["creation_unix"],
            "supervisor_sha256": digest(__file__), "shutdown_requested": False,
            "policy": "Restart only owned queues exiting after nvidia-smi query TimeoutExpired; preserve all inference settings and checkpoints"}

    def status(phase, **details):
        save(ROOT / "records/supervisor_status.json", {**base, "phase": phase, "time_unix": time.time(), **details})

    failures = 0
    try:
        while True:
            identity = load_frozen()
            protect_fixed("supervisor_before_queue")
            complete = ROOT / "records/all_work_complete.json"
            if complete.exists() and read(complete).get("status") == "complete":
                status("all_work_complete")
                return 0
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1000000:06d}"
            out = ROOT / "records" / f"queue.supervised.{stamp}.stdout.log"
            err = ROOT / "records" / f"queue.supervised.{stamp}.stderr.log"
            private = {"OPENAI_API_KEY", "TOGETHERAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY"}
            env = {name: os.environ[name] for name in os.environ if name not in private}
            launched = time.time()
            with out.open("xb", buffering=0) as stdout, err.open("xb", buffering=0) as stderr:
                child = subprocess.Popen([sys.executable, "-B", "-u", str(ROOT / "operations/resume_entry.py"), "queue"],
                    cwd=ROOT, env=env, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
                append(ROOT / "records/supervisor_history.jsonl", {**base, "event": "queue_started", "launcher_pid": child.pid, "time_unix": launched, "run_identity_sha256": canonical(identity)})
                while True:
                    status("queue_running", queue_launcher_pid=child.pid, stdout=str(out), stderr=str(err), restarts=failures)
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            queue = read(ROOT / "records/queue_status.json")
            append(ROOT / "records/supervisor_history.jsonl", {"event": "owned_queue_exit", "returncode": code, "queue_status": queue, "time_unix": time.time()})
            if code == 0 and queue.get("phase") == "all_work_complete":
                status("all_work_complete")
                return 0
            if not resource_query_timeout(queue, launched):
                status("blocked_non_resource_error", returncode=code, queue_status=queue)
                return 1
            failures += 1
            delay = min(60 * 2 ** min(failures - 1, 3), 300)
            status("waiting_after_resource_query_timeout", restart_delay_seconds=delay, restarts=failures, queue_status=queue)
            for _ in range(delay // 30):
                time.sleep(30)
    except Exception as exc:
        status("blocked_supervisor_error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if lock.exists() and read(lock).get("pid") == os.getpid():
            lock.rename(lock.with_name(f"supervisor.lock.finished-{time.time_ns()}"))


if __name__ == "__main__":
    raise SystemExit(run())
