import base64
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

SHARED = None


def canonical_hash(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def append_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    records = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON at {path}:{index}; preserve and repair explicitly") from exc
    return records


def rng_state():
    ns = np.random.get_state()
    encode = lambda value: base64.b64encode(value.cpu().numpy().tobytes()).decode("ascii")
    return {"python": random.getstate(), "numpy": [ns[0], ns[1].tolist(), *ns[2:]], "torch_cpu": encode(torch.get_rng_state()), "torch_cuda": [encode(s) for s in torch.cuda.get_rng_state_all()]}


def restore_rng(state):
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value
    decode = lambda value: torch.tensor(list(base64.b64decode(value)), dtype=torch.uint8)
    random.setstate(tuples(state["python"]))
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(decode(state["torch_cpu"]))
    torch.cuda.set_rng_state_all([decode(v) for v in state["torch_cuda"]])


def shared_local_model(model_name, **kwargs):
    if SHARED is None or model_name != SHARED.config["model_id"]:
        raise RuntimeError("Only the configured shared local model is permitted; no fallback")
    return SHARED


class LocalTransformersAgent:
    """One loaded model, sequential requests, complete local call journal."""

    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.args = SimpleNamespace(model=config["model_id"], tracing_model=config["model_id"])
        self.stage, self.case_dir, self.last_call = "unspecified", None, None
        model_path = self.root / config["model_path"]
        verified = json.loads((self.root / "records/model_verified.json").read_text(encoding="utf-8"))
        assert verified["revision"] == config["model_revision"]
        assert model_path.resolve() == Path(verified["local_path"]).resolve()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU or remote fallback")
        if config["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported; record an explicit separate FP16 configuration first")
        torch.set_num_threads(config["torch_num_threads"])
        started = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        quantization = config.get("quantization", "none")
        load_options = {}
        if quantization == "bitsandbytes_nf4":
            # Optional home handoff only; never selected by the BF16 config.
            from transformers import BitsAndBytesConfig
            load_options = {"device_map": {"": 0}, "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=getattr(torch, config["dtype"]))}
        elif quantization != "none":
            raise ValueError(f"Unsupported explicit quantization setting: {quantization}")
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False, torch_dtype=getattr(torch, config["dtype"]), attn_implementation=config["attention_implementation"], **load_options)
        if quantization == "none":
            self.model.to("cuda:0")
        self.model.eval()
        self.load_seconds = time.monotonic() - started
        self.context_limit = int(self.model.config.max_position_embeddings)
        self.eos_ids = self.model.generation_config.eos_token_id
        global SHARED
        if SHARED is not None:
            raise RuntimeError("A shared model is already loaded")
        SHARED = self

    def begin_case(self, directory, identity):
        self.case_dir = Path(directory)
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.identity, self.ordinal = identity, 0
        self.case_attempt_id = str(time.time_ns())
        self.completed_calls, self.attempts = {}, {}
        journal = self.case_dir / "calls.jsonl"
        if journal.exists():
            for record in read_jsonl(journal):
                if record["event"] == "call_start":
                    self.attempts[record["ordinal"]] = max(self.attempts.get(record["ordinal"], 0), record["attempt"])
                elif record["event"] == "call_end" and record["status"] == "ok":
                    self.completed_calls[record["ordinal"]] = record

    def preprocess_input(self, text, system_prompt=None, history=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": str(system_prompt)})
        for i, message in enumerate(history or []):
            messages.append(dict(message) if isinstance(message, dict) else {"role": "user" if i % 2 == 0 else "assistant", "content": str(message)})
        messages.append({"role": "user", "content": str(text)})
        return messages

    def postprocess_output(self, output):
        return output.strip()

    def interact(self, prompt, temperature=None, max_tokens=None, system_prompt=None, history=None):
        if self.case_dir is None:
            raise RuntimeError("begin_case must precede inference")
        self.ordinal += 1
        ordinal = self.ordinal
        temperature = self.config["default_temperature"] if temperature is None else float(temperature)
        max_tokens = self.config["default_max_tokens"] if max_tokens is None else int(max_tokens)
        messages = self.preprocess_input(prompt, system_prompt, history)
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        gen = {"max_new_tokens": max_tokens, "do_sample": temperature > 0, "temperature": temperature if temperature > 0 else 1.0, "top_p": self.config["top_p"], "top_k": self.config["top_k"], "repetition_penalty": self.config["repetition_penalty"], "num_beams": 1, "use_cache": True, "eos_token_id": self.eos_ids, "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
        request = {"identity": self.identity, "ordinal": ordinal, "stage": self.stage, "target_agent": getattr(self, "target_agent", None), "messages": messages, "rendered_prompt": rendered, "requested_temperature": temperature, "generation": gen}
        request_hash = canonical_hash(request)
        cached = self.completed_calls.get(ordinal)
        if cached is not None:
            if cached["request_sha256"] != request_hash:
                raise RuntimeError(f"Resume mismatch at call {ordinal}; use a new run version")
            restore_rng(cached["rng_after"])
            self.last_call = cached
            append_json(self.case_dir / "replay.jsonl", {"ordinal": ordinal, "request_sha256": request_hash, "stage": self.stage, "event": "replayed"})
            return self.postprocess_output(cached["raw_output"])
        attempt = self.attempts.get(ordinal, 0) + 1
        self.attempts[ordinal] = attempt
        started = time.monotonic()
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False, truncation=False)
        n_input = int(inputs["input_ids"].shape[-1])
        common = {"ordinal": ordinal, "attempt": attempt, "request_sha256": request_hash, "stage": self.stage}
        append_json(self.case_dir / "calls.jsonl", {**common, "event": "call_start", "request": request, "input_tokens": n_input, "context_limit": self.context_limit, "time_unix": time.time()})
        try:
            if n_input + max_tokens > self.context_limit:
                raise RuntimeError(f"Context overflow: {n_input} + {max_tokens} > {self.context_limit}; no truncation applied")
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            generation_start = time.monotonic()
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, generation_config=GenerationConfig(**gen))
            torch.cuda.synchronize()
            generation_seconds = time.monotonic() - generation_start
            generated = outputs[0, n_input:].tolist()
            raw = self.tokenizer.decode(generated, skip_special_tokens=True)
            eos = set(self.eos_ids if isinstance(self.eos_ids, list) else [self.eos_ids])
            at_limit = len(generated) >= max_tokens and (not generated or generated[-1] not in eos)
            record = {**common, "event": "call_end", "status": "ok", "raw_output": raw, "input_tokens": n_input, "output_tokens": len(generated), "generated_token_ids": generated, "finish_reason": "length" if at_limit else "eos", "truncated": at_limit, "generation_seconds": generation_seconds, "wall_seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "rng_after": rng_state(), "time_unix": time.time()}
            append_json(self.case_dir / "calls.jsonl", record)
            self.last_call = record
            return self.postprocess_output(raw)
        except Exception as exc:
            append_json(self.case_dir / "calls.jsonl", {**common, "event": "call_end", "status": "error", "error_type": type(exc).__name__, "error": str(exc), "input_tokens": n_input, "output_tokens": 0, "wall_seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "time_unix": time.time()})
            raise

    def batch_interact(self, prompts, temperature=None, max_tokens=None, system_prompts=None, histories=None, system_prompt=None, history=None):
        if system_prompts is not None and system_prompt is not None:
            raise ValueError("Specify only one system prompt argument")
        systems = system_prompts if system_prompts is not None else system_prompt
        systems = systems if isinstance(systems, list) else [systems] * len(prompts)
        histories = histories if histories is not None else [history] * len(prompts)
        if len(systems) != len(prompts) or len(histories) != len(prompts):
            raise ValueError("Prompt/system/history length mismatch")
        return [self.interact(p, temperature=temperature, max_tokens=max_tokens, system_prompt=s, history=h) for p, s, h in zip(prompts, systems, histories)]

    def generate(self, prompt, temperature=None, max_tokens=None):
        return self.interact(prompt, temperature=temperature, max_tokens=max_tokens)

    def batch_cot(self, prompts, temperature=None, max_tokens=None):
        return self.batch_interact([p.removesuffix("\nAnswer:") + "\nLet's think step by step." for p in prompts], temperature=temperature, max_tokens=max_tokens)
