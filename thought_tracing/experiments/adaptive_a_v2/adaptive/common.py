"""Local records and protection checks; safe to import without ML packages."""
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
FIXED = ROOT.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    with tmp.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (32, 33) or attempt == 9:
                raise
            time.sleep(min(0.1 * 2**attempt, 2))


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def lines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        code = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def protect_fixed(label):
    before = read(ROOT / "records/fixed4_protection_before.json")
    mismatches = [name for name, expected in before["protected_sha256"].items() if not (FIXED / name).exists() or digest(FIXED / name) != expected]
    result = {"checked_at": datetime.now().astimezone().isoformat(), "label": label, "files_checked": len(before["protected_sha256"]), "mismatches": mismatches, "outputs_excluded": True, "passed": not mismatches}
    save(ROOT / "records" / f"fixed4_protection_{label}.json", result)
    if mismatches:
        raise RuntimeError("Fixed-4 source/config changed externally; preserve both runs and investigate")
    return result


def checkpoint_complete(directory, identity_sha):
    path = Path(directory) / "result.json"
    if not path.exists():
        return False
    value = read(path)
    if value["run_identity_sha256"] != identity_sha:
        raise RuntimeError("Result identity mismatch; use a new run ID")
    return value.get("status") in ("ok", "error")


def isolation_env():
    values = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "WANDB_DISABLED": "true", "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": str(ROOT / "cache/huggingface"), "HF_HUB_CACHE": str(ROOT / "cache/huggingface/hub"), "TORCH_HOME": str(ROOT / "cache/torch"), "XDG_CACHE_HOME": str(ROOT / "cache/xdg"), "TEMP": str(ROOT / "cache/tmp"), "TMP": str(ROOT / "cache/tmp")}
    for name in ("huggingface/hub", "torch", "xdg", "tmp"):
        (ROOT / "cache" / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(values)
    return values


def call_metrics(directory):
    calls = [row for row in lines(Path(directory) / "calls.jsonl") if row["event"] == "call_end"]
    values = {"calls": len(calls), "failed_calls": sum(row["status"] != "ok" for row in calls), "input_tokens": sum(row.get("input_tokens", 0) for row in calls), "output_tokens": sum(row.get("output_tokens", 0) for row in calls), "call_wall_seconds": sum(row.get("wall_seconds", 0) for row in calls), "truncated_calls": sum(bool(row.get("truncated")) for row in calls), "peak_allocated_bytes": max((row.get("peak_allocated_bytes", 0) for row in calls), default=0), "peak_reserved_bytes": max((row.get("peak_reserved_bytes", 0) for row in calls), default=0)}
    interrupted = lines(Path(directory) / "resource_interruptions.jsonl")
    values["partial_output_tokens_resource_interruptions"] = sum(row.get("output_tokens", 0) for row in interrupted)
    values["output_tokens"] += values["partial_output_tokens_resource_interruptions"]
    values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
    values["failed_generated_token_counts_may_be_unknown"] = any(row["status"] != "ok" for row in calls)
    return values
