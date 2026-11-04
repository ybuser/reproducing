"""WDDM resource measurement after a recorded 20GiB allocation/write validation.

Keep 28GiB admission and 2GiB running margins. Preserve raw NVML readings alongside
the CUDA-allocatable reading used by the existing guard. No model or decoding changes.
"""
import math
import subprocess
import time

from adaptive.common import ROOT, append, digest, read
from adaptive.resources import gpu_snapshot as NVML_SNAPSHOT

LAST_NVML = None
LAST_NVML_TIME = None
VALIDATED = False
ANNOUNCED = False


def combine_memory(row, free_bytes, total_bytes, nvml_error=None):
    if not 0 <= free_bytes <= total_bytes:
        raise RuntimeError("Invalid CUDA free/total memory values")
    if abs(total_bytes / 2**20 - row["total_mib"]) > 2:
        raise RuntimeError("CUDA/NVML device capacity mismatch; do not change admission")
    return {**row, "nvml_reported": dict(row),
        "total_mib": int(total_bytes / 2**20),
        "used_mib": math.ceil((total_bytes - free_bytes) / 2**20),
        "free_mib": int(free_bytes / 2**20),
        "measurement_source": "cuda_mem_get_info_wddm_v1",
        "cuda_free_bytes": free_bytes, "cuda_total_bytes": total_bytes,
        "nvml_query_error": nvml_error}


def cuda_snapshot():
    global LAST_NVML, LAST_NVML_TIME, VALIDATED, ANNOUNCED
    if not VALIDATED:
        validation = read(ROOT / "records/cuda_allocation_validation_20261030.json")
        if not validation.get("passed") or validation["bytes_requested"] < 20 * 2**30:
            raise RuntimeError("20GiB allocation/write validation is required")
        mode = subprocess.run(["nvidia-smi", "--query-gpu=driver_model.current", "--format=csv,noheader"],
            text=True, capture_output=True, check=True, timeout=8, creationflags=subprocess.CREATE_NO_WINDOW)
        if mode.stdout.strip() != "WDDM":
            raise RuntimeError("CUDA measurement override is scoped to this single WDDM GPU")
        VALIDATED = True
    query_error = None
    try:
        snapshot = NVML_SNAPSHOT()
        if len(snapshot) != 1:
            raise RuntimeError("Expected one physical GPU")
        LAST_NVML, LAST_NVML_TIME = snapshot[0], time.time()
    except subprocess.TimeoutExpired as exc:
        if LAST_NVML is None:
            raise
        query_error = {"type": type(exc).__name__, "message": str(exc)}
        append(ROOT / "records/runtime_resource_diagnostics.jsonl", {
            "time_unix": time.time(), "event": "nvml_query_timeout_with_cuda_check", **query_error})
    from repro import bootstrap  # existing process-local DLL setup; no installation
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one CUDA device")
    expected_uuid = read(ROOT / "records/cuda_allocation_validation_20261030.json")["before_nvml"][0]["uuid"]
    if LAST_NVML["uuid"] != expected_uuid:
        raise RuntimeError("GPU changed since allocation validation")
    free, total = torch.cuda.mem_get_info(0)
    measured = combine_memory(LAST_NVML, free, total, query_error)
    measured["nvml_last_success_unix"] = LAST_NVML_TIME
    if query_error:
        measured["utilization_percent"] = None
    if not ANNOUNCED:
        append(ROOT / "records/runtime_resource_diagnostics.jsonl", {
            "time_unix": time.time(), "event": "cuda_wddm_measurement_ready",
            "operations_sha256": digest(__file__), "measurement": measured,
            "admission_free_mib": 28672, "running_min_free_mib": 2048,
            "allocation_validation": "records/cuda_allocation_validation_20261030.json"})
        ANNOUNCED = True
    return [measured]
