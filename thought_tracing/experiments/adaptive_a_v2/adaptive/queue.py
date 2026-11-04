"""Persistent local job: fixed-4 -> smoke -> adaptive600 with first30 draft; no shutdown."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import ROOT, FIXED, alive, append, canonical, digest, isolation_env, protect_fixed, read, save
from .resources import fixed_active, gpu_snapshot, ram_snapshot


def fixed_complete():
    directory = FIXED / "runs/bf16_official600_v1"
    manifest = read(FIXED / "manifests/tomi_official_defaults600_v1.json")
    count = sum((directory / "official600" / f"row-{r['row_id']:04d}" / method / "result.json").exists() for r in manifest["rows"] for method in ("Base", "CoT", "TT"))
    progress = read(directory / "progress.json")
    return count == 1800 and progress.get("phase") == "complete", count


def run(interval=30):
    isolation_env()
    lock = ROOT / "records/queue.lock"
    if lock.exists():
        if alive(read(lock)["pid"]):
            raise RuntimeError("Adaptive queue is already active")
        lock.rename(ROOT / "records" / f"queue.lock.stale-{time.time_ns()}")
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    consecutive_free_samples = 0
    try:
        while True:
            active = fixed_active()
            completed, count = fixed_complete()
            frozen = (ROOT / "records/adaptive_frozen_identity.json").exists()
            status = {"pid": os.getpid(), "time_unix": time.time(), "fixed_active_pids": active, "fixed_completed_conditions": count, "fixed_planned_conditions": 1800, "cohort_confirmed_and_frozen": frozen, "shutdown_after_all_work": False, "adaptive_gpu_started": False}
            if not frozen:
                status["phase"] = "waiting_validation_and_freeze"
            elif active or not completed:
                status["phase"] = "waiting_fixed4_completion_and_exit"
            else:
                from .runner import load_frozen
                identity = load_frozen()
                protect_fixed("queue_before_gpu")
                snapshot = gpu_snapshot()
                memory = ram_snapshot()
                enough = snapshot and snapshot[0]["free_mib"] >= 28 * 1024 and memory["system_ram_available_bytes"] >= 24 * 2**30
                consecutive_free_samples = consecutive_free_samples + 1 if enough else 0
                status.update(phase="waiting_sustained_gpu_headroom", resource_snapshot=snapshot, ram=memory, consecutive_free_samples=consecutive_free_samples)
                if consecutive_free_samples >= 3:
                    status.update(phase="adaptive_worker_starting", adaptive_gpu_started=True)
                    save(ROOT / "records/queue_status.json", status)
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    out = ROOT / "records" / f"adaptive_worker.{stamp}.stdout.log"
                    err = ROOT / "records" / f"adaptive_worker.{stamp}.stderr.log"
                    env = os.environ.copy()
                    for secret_name in ("OPENAI_API_KEY", "TOGETHERAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY"):
                        env.pop(secret_name, None)
                    with out.open("ab", buffering=0) as stdout, err.open("ab", buffering=0) as stderr:
                        child = subprocess.Popen([sys.executable, "-B", "-u", "-m", "adaptive.runner", "--execute"], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
                        status.update(phase="adaptive_worker_running", adaptive_launcher_pid=child.pid, stdout=str(out), stderr=str(err))
                        save(ROOT / "records/queue_status.json", status)
                        while child.poll() is None:
                            status["time_unix"] = time.time()
                            worker_lock = ROOT / "runs" / identity["config"]["run_version"] / "worker.lock"
                            if worker_lock.exists():status["adaptive_worker_pid"] = read(worker_lock)["pid"]
                            save(ROOT / "records/queue_status.json", status)
                            time.sleep(interval)
                    append(ROOT / "records/queue_history.jsonl", {"event": "adaptive_worker_exit", "returncode": child.returncode, "time_unix": time.time(), "launcher_pid": child.pid})
                    if child.returncode == 75:
                        consecutive_free_samples = 0
                        save(ROOT / "records/queue_status.json", {**status, "phase": "adaptive_paused_waiting_resources", "time_unix": time.time()})
                        time.sleep(interval)
                        continue
                    if child.returncode != 0:
                        save(ROOT / "records/queue_status.json", {**status, "phase": "blocked_adaptive_error", "returncode": child.returncode, "time_unix": time.time(), "shutdown_executed": False})
                        return 1
                    from .report import analyze
                    summary = analyze()
                    if summary["status"] != "complete" or fixed_active() or not fixed_complete()[0]:
                        raise RuntimeError("Completion verification failed; no shutdown")
                    final_protection = protect_fixed("after_all_work")
                    for filename in ("REPORT.md", "summary.json"):
                        source = FIXED / "reports/bf16_official600_v1" / filename
                        shutil.copy2(source, ROOT / "reports" / f"fixed600_final_{filename}")
                    final = {"status": "complete", "time_unix": time.time(), "fixed_conditions": 1800, "adaptive_items": summary["cohort_size"], "comparison": "reports/COMPARISON_KO.md", "fixed_source_protection": final_protection, "fixed_overlap": summary["fixed_overlap_detected"], "pending_human_semantic_review": True, "shutdown_requested": False, "owned_worker_exited": True}
                    save(ROOT / "records/all_work_complete.json", final)
                    (ROOT / "reports/FINAL_STATUS_KO.md").write_text(f"# 자동 작업 완료\n\nfixed-4 600문항 1800조건과 adaptive-A {summary['cohort_size']}문항의 시도·기록·자동 비교를 마쳤다. 실패와 사람 검토 항목은 COMPARISON_KO.md 및 HUMAN_REVIEW.json에 따로 남겼다. 기존 소스 불변성 검사를 통과했다. 논문 전체 조건이나 pruning 하나의 효과를 재현했다고 주장하지 않는다.\n\n실행·재개 명령: ../README.md. 전체 비교: COMPARISON_KO.md. 사용자가 종료 요청을 취소했으므로 실험 완료 후에도 컴퓨터를 종료하지 않는다.\n", encoding="utf-8")
                    save(ROOT / "records/queue_status.json", {**status, "phase": "all_work_complete", "time_unix": time.time(), "shutdown_requested": False})
                    return 0
            save(ROOT / "records/queue_status.json", status)
            time.sleep(interval)
    except Exception as exc:
        save(ROOT / "records/queue_status.json", {"pid": os.getpid(), "phase": "blocked_queue_error", "error_type": type(exc).__name__, "error": str(exc), "time_unix": time.time(), "shutdown_executed": False})
        append(ROOT / "records/queue_history.jsonl", {"event": "queue_error", "error_type": type(exc).__name__, "error": str(exc), "time_unix": time.time()})
        return 1
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if lock.exists() and read(lock)["pid"] == os.getpid():
            lock.rename(ROOT / "records" / f"queue.lock.finished-{time.time_ns()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    raise SystemExit(run())
