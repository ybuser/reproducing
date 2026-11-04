"""Adaptive-owned admission and monitoring. Never signals a fixed-4 process."""
import ctypes
import os
import re
import subprocess
import threading
import time

from .common import ROOT, append, alive


class ResourcePause(RuntimeError):
    pass


def ram_snapshot():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in ("peak_working", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    memory, process = MemoryStatus(), ProcessMemory()
    memory.length, process.cb = ctypes.sizeof(memory), ctypes.sizeof(process)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), process.cb)
    return {"system_ram_total_bytes": memory.total, "system_ram_available_bytes": memory.available, "process_rss_bytes": process.working, "process_peak_wset_bytes": process.peak_working}


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    rows = []
    for line in result.stdout.strip().splitlines():
        index, uuid, total, used, free, utilization = [p.strip() for p in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "total_mib": int(total), "used_mib": int(used), "free_mib": int(free), "utilization_percent": int(utilization)})
    return rows


def process_identity(pid):
    """Inspect an instance, not just a reusable Windows PID; never sends signals."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return {"alive": ctypes.get_last_error() == 5, "creation_unix": None}
    try:
        exit_code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
            return {"alive": False, "creation_unix": None}
        created, exited, system, user = [ctypes.c_ulonglong() for _ in range(4)]
        known = kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user))
        return {"alive": True, "creation_unix": created.value / 10_000_000 - 11_644_473_600 if known else None}
    finally:
        kernel.CloseHandle(handle)


def fixed_active(probe=None):
    # The frozen original PID+creation time identifies the original worker.
    # A current resume lock also protects a new fixed worker with a new PID.
    from .common import FIXED, read
    probe = probe or process_identity
    references = read(ROOT / "records/fixed_process_commands.json")
    instances, active = {}, set()
    def instance(pid):
        if pid not in instances:
            instances[pid] = probe(pid)
        return instances[pid]
    for reference in references:
        pid = int(reference["pid"])
        expected = int(re.search(r"\d{10,}", reference["created"]).group()) / 1000
        current = instance(pid)
        if current["alive"] and (current["creation_unix"] is None or abs(current["creation_unix"] - expected) <= 1):
            active.add(pid)
    lock = FIXED / "runs/bf16_official600_v1/worker.lock"
    if lock.exists():
        pid = int(read(lock)["pid"])
        current = instance(pid)
        if current["alive"] and (current["creation_unix"] is None or current["creation_unix"] <= lock.stat().st_mtime + 1):
            active.add(pid)
    return sorted(active)


class ResourceMonitor:
    def __init__(self, directory):
        self.directory = directory
        self.stop_event = threading.Event()
        self.pressure = threading.Event()
        self.latest = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop_event.is_set():
            try:
                gpu = gpu_snapshot()
                memory = ram_snapshot()
                active = fixed_active()
                self.latest = {"time_unix": time.time(), "pid": os.getpid(), "gpu": gpu, **memory, "fixed_active_pids": active, "fixed_overlap": bool(active)}
                append(self.directory / "resource_samples.jsonl", self.latest)
                if not gpu or gpu[0]["free_mib"] < 2048 or memory["system_ram_available_bytes"] < 4 * 2**30 or active:
                    self.pressure.set()
            except Exception as exc:
                append(self.directory / "resource_samples.jsonl", {"time_unix": time.time(), "error_type": type(exc).__name__, "error": str(exc)})
                self.pressure.set()
            self.stop_event.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=10)


def guarded_agent_class():
    from repro.local_model import LocalTransformersAgent
    from transformers import StoppingCriteria, StoppingCriteriaList
    import torch

    class PressureStop(StoppingCriteria):
        def __init__(self, monitor):
            self.monitor = monitor

        def __call__(self, input_ids, scores, **kwargs):
            return self.monitor.pressure.is_set()

    class GuardedAgent(LocalTransformersAgent):
        def __init__(self, config, root, monitor):
            self.monitor = monitor
            super().__init__(config, root)
            original_generate = self.model.generate

            def generate(*args, **kwargs):
                criteria = list(kwargs.pop("stopping_criteria", []))
                kwargs["stopping_criteria"] = StoppingCriteriaList(criteria + [PressureStop(monitor)])
                output = original_generate(*args, **kwargs)
                if monitor.pressure.is_set():
                    input_length = kwargs["input_ids"].shape[-1]
                    tokens = output[0, input_length:].tolist()
                    append(self.case_dir / "resource_interruptions.jsonl", {"time_unix": time.time(), "stage": self.stage, "ordinal": self.ordinal, "output_tokens": len(tokens), "partial_token_ids": tokens, "raw_partial_output": self.tokenizer.decode(tokens, skip_special_tokens=True), "reason": "memory pressure or fixed overlap; adaptive-only generation stopped"})
                    raise ResourcePause("Adaptive-only generation stopped at token boundary; replay checkpoint on resume")
                return output

            self.model.generate = generate

        def batch_interact(self, prompts, *args, **kwargs):
            started = time.monotonic()
            ordinal = self.ordinal
            record = {"event": "batch_interface_start", "time_unix": time.time(), "stage": self.stage, "requested_hypothesis_slots": len(prompts), "ordinal_before": ordinal, "external_api_calls": 0}
            append(self.case_dir / "batch_events.jsonl", record)
            try:
                return super().batch_interact(prompts, *args, **kwargs)
            finally:
                append(self.case_dir / "batch_events.jsonl", {**record, "event": "batch_interface_end", "time_unix": time.time(), "ordinal_after": self.ordinal, "interact_slots_traversed_including_replays": self.ordinal - ordinal, "wall_seconds": time.monotonic() - started})

        def interact(self, prompt, temperature=None, max_tokens=None, system_prompt=None, history=None):
            if self.ordinal + 1 not in self.completed_calls:
                if self.monitor.pressure.is_set() or fixed_active():
                    raise ResourcePause("Adaptive pre-call resource guard")
                rendered = self.tokenizer.apply_chat_template(self.preprocess_input(prompt, system_prompt, history), tokenize=False, add_generation_prompt=True)
                tokens = len(self.tokenizer.encode(rendered, add_special_tokens=False))
                cap = self.config["default_max_tokens"] if max_tokens is None else max_tokens
                cfg = self.model.config
                kv_bytes = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2 * (tokens + cap)
                free, total = torch.cuda.mem_get_info()
                if free < kv_bytes + 2 * 2**30:
                    raise ResourcePause(f"Insufficient adaptive headroom for KV plus 2GiB margin: free={free}, estimated_kv={kv_bytes}")
            try:
                return super().interact(prompt, temperature, max_tokens, system_prompt, history)
            except torch.OutOfMemoryError as exc:
                raise ResourcePause("Adaptive CUDA OOM; own process exits and journal remains resumable") from exc

    return GuardedAgent
