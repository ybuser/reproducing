"""Versioned operations entry: confirm absent completed fixed PIDs using Windows inventory.

Frozen experiment files, model inputs, decoding, scoring and cache identities are unchanged.
Only the old fixed process instances get an additional read-only existence check.
"""
import ctypes
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive import resources
from adaptive.common import FIXED, append, canonical, digest, read

NATIVE_IDENTITY = resources.process_identity
REPORTED = set()


def windows_process_ids():
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.EnumProcesses.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
    for capacity in (4096, 8192, 16384, 32768, 65536):
        values = (ctypes.c_ulong * capacity)()
        written = ctypes.c_ulong()
        if not psapi.EnumProcesses(values, ctypes.sizeof(values), ctypes.byref(written)):
            raise ctypes.WinError(ctypes.get_last_error())
        if written.value < ctypes.sizeof(values):
            return set(values[:written.value // ctypes.sizeof(ctypes.c_ulong)])
    raise RuntimeError("Process inventory truncated; preserve conservative protection")


def resolve_completed_identity(pid, native_probe=NATIVE_IDENTITY,
                               inventory=windows_process_ids, completed=None, log=True):
    current = native_probe(pid)
    if not current["alive"] or current["creation_unix"] is not None:
        return current
    try:
        if completed is None:
            original_pids = {int(item["pid"]) for item in read(ROOT / "records/fixed_process_commands.json")}
            progress = read(FIXED / "runs/bf16_official600_v1/progress.json")
            completed = (pid in original_pids and progress.get("phase") == "complete"
                and progress.get("completed_case_conditions") == 1800
                and not (FIXED / "runs/bf16_official600_v1/worker.lock").exists())
        if not completed or pid in inventory():
            return current
        time.sleep(0.02)
        if pid in inventory():
            return current
        fresh = native_probe(pid)
        if fresh["creation_unix"] is not None:
            return fresh
        resolved = {"alive": False, "creation_unix": None}
        if log and pid not in REPORTED:
            append(ROOT / "records/completed_fixed_pid_resolution.jsonl", {
                "time_unix": time.time(), "pid": pid, "native_result": current,
                "resolution": "Completed fixed run; no current lock; absent in two complete EnumProcesses inventories",
                "operations_source_sha256": digest(__file__)})
            REPORTED.add(pid)
        return resolved
    except Exception:
        return current


def install():
    resources.process_identity = resolve_completed_identity
    from operations.cuda_admission import cuda_snapshot
    resources.gpu_snapshot = cuda_snapshot


def main():
    install()
    from adaptive.runner import load_frozen
    identity = load_frozen()
    append(ROOT / "records/runtime_sessions_20261030.jsonl", {
        "time_unix": time.time(), "entry_mode": sys.argv[1],
        "operations_version": "resume_entry_v2_cuda_wddm_v1", "operations_source_sha256": digest(__file__),
        "resource_measurement_sha256": digest(ROOT / "operations/cuda_admission.py"),
        "run_identity_sha256": canonical(identity), "inference_changes": False,
        "resource_measurement_changed": True, "admission_poll_seconds": 5})
    if sys.argv[1] == "queue":
        from adaptive import queue
        original_popen = subprocess.Popen

        def worker_popen(argv, **kwargs):
            if list(argv[1:]) != ["-B", "-u", "-m", "adaptive.runner", "--execute"]:
                raise RuntimeError("Unexpected queue child command; no process started")
            return original_popen([argv[0], "-B", "-u", str(Path(__file__).resolve()), "worker"], **kwargs)

        # Change this queue module's launch object only, not the subprocess module.
        queue.subprocess = SimpleNamespace(Popen=worker_popen, CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW)
        return queue.run(interval=5)
    if sys.argv[1] == "worker":
        from adaptive import runner
        sys.argv = [str(Path(__file__).resolve()), "--execute"]
        return runner.main()
    raise RuntimeError("Expected queue or worker entry mode")


if __name__ == "__main__":
    raise SystemExit(main())
