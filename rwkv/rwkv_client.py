from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = WORKSPACE_ROOT / "third_party" / "reference"
SOTA_REFERENCE_DIR = WORKSPACE_ROOT / "third_party" / "sota_reference"
MODEL_STEM = Path("/data_temp/mnt/raid5/zjx/rwkv/full-training/rwkv7-g1e-7.2b-20260301-ctx8192")
VOCAB_PATH = REFERENCE_DIR / "rwkv_vocab_v20230424.txt"
RAPID_SAMPLING_DIR = WORKSPACE_ROOT / "third_party" / "Rapid-Sampling-main"


_rapid_sampler: Any | None = None
_rapid_sampler_lock = threading.Lock()
_raw_log_file_lock = threading.Lock()


def get_rapid_sampler() -> Any | None:
    global _rapid_sampler
    with _rapid_sampler_lock:
        if _rapid_sampler is not None:
            return _rapid_sampler
        try:
            if torch.version.hip is not None:
                sources = [
                    str(RAPID_SAMPLING_DIR / "hip" / "sampling_op.hip"),
                    str(RAPID_SAMPLING_DIR / "hip" / "sampling.hip"),
                ]
                extra_cuda_cflags = ["-fopenmp", "-ffast-math", "-O3", "-munsafe-fp-atomics"]
            else:
                sources = [
                    str(RAPID_SAMPLING_DIR / "sampling.cpp"),
                    str(RAPID_SAMPLING_DIR / "sampling.cu"),
                ]
                extra_cuda_cflags = ["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"]
            _rapid_sampler = load(
                name="rapid_sampling_rwkv",
                sources=sources,
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=True,
            )
        except Exception as exc:
            print(f"[rwkv_client] rapid sampler unavailable, falling back: {exc}", flush=True)
            _rapid_sampler = False
        return None if _rapid_sampler is False else _rapid_sampler


def normalize_messages(prompt: str | list[dict[str, Any]] | dict[str, Any]) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return str(prompt)
    parts: list[str] = []
    for msg in prompt:
        role = str(msg.get("role", "user")).strip().lower()
        if role == "assistant":
            role_name = "Assistant"
        elif role == "system":
            role_name = "System"
        else:
            role_name = "User"
        content = msg.get("content", "")
        parts.append(f"{role_name}: {content}")
    return "\n\n".join(parts)


def messages_to_instruction(prompt: str | list[dict[str, Any]] | dict[str, Any]) -> tuple[str, str]:
    if isinstance(prompt, str):
        return "Answer the user request.", prompt
    if isinstance(prompt, dict):
        return "Answer the user request.", str(prompt)

    system_parts: list[str] = []
    conversation_parts: list[str] = []
    for msg in prompt:
        role = str(msg.get("role", "user")).strip().lower()
        content = str(msg.get("content", ""))
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content}")
        else:
            conversation_parts.append(f"User: {content}")

    instruction = "\n\n".join(system_parts).strip() or "Answer the user request."
    input_text = "\n\n".join(conversation_parts).strip()
    return instruction, input_text


class RWKVEngine:
    def __init__(
        self,
        *,
        model_path: str = str(MODEL_STEM),
        vocab_path: str = str(VOCAB_PATH),
        reference_dir: str = str(REFERENCE_DIR),
        engine_module: str = "rwkv7",
        state_path: str = "",
        device: str = "cuda",
    ) -> None:
        if reference_dir not in sys.path:
            sys.path.insert(0, reference_dir)

        if engine_module == "rwkv7":
            from rwkv7 import RWKV_x070
        elif engine_module == "rwkv7_fp16":
            from rwkv7_fp16 import RWKV_x070
        else:
            raise ValueError(f"Unknown RWKV engine module: {engine_module}")
        from utils import TRIE_TOKENIZER, sample_logits

        self.sample_logits = sample_logits
        args = type("Args", (), {})()
        args.MODEL_NAME = model_path[:-4] if model_path.endswith(".pth") else model_path
        args.vocab_size = 65536
        self.model = RWKV_x070(args)
        self.tokenizer = TRIE_TOKENIZER(vocab_path)
        self.device = device
        self.state_path = state_path
        self.initial_state = self.load_initial_state(state_path) if state_path else None
        self.lock = threading.Lock()

    def load_initial_state(self, state_path: str) -> dict[str, torch.Tensor]:
        path = Path(state_path)
        if not path.exists():
            raise FileNotFoundError(f"time_state checkpoint not found: {path}")
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(path, map_location="cpu")
        if "full_state" in sd and isinstance(sd["full_state"], dict):
            sd = sd["full_state"]
        elif "time_state" in sd and isinstance(sd["time_state"], dict):
            sd = sd["time_state"]
        return {str(k): v.detach().cpu() for k, v in sd.items() if torch.is_tensor(v)}

    def generate_initial_state(self, batch_size: int):
        state = self.model.generate_zero_state(batch_size)
        if not self.initial_state:
            return state
        args = self.model.args
        for i in range(args.n_layer):
            ts = self.initial_state.get(f"blocks.{i}.att.time_state")
            if ts is not None:
                state[1][i] = ts.to(device=state[1].device, dtype=state[1].dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
            att_ts = self.initial_state.get(f"blocks.{i}.att.ts_state")
            if att_ts is not None:
                state[0][i, 0] = att_ts.to(device=state[0].device, dtype=state[0].dtype).unsqueeze(0).expand(batch_size, -1).clone()
            ffn_ts = self.initial_state.get(f"blocks.{i}.ffn.ts_state")
            if ffn_ts is not None:
                state[0][i, 1] = ffn_ts.to(device=state[0].device, dtype=state[0].dtype).unsqueeze(0).expand(batch_size, -1).clone()
        return state

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, utf8_errors="replace")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 0,
        presence_penalty: float = 1.0,
        repetition_penalty: float = 0.1,
        penalty_decay: float = 0.996,
        use_rapid_sampling: bool = True,
        prefill_chunk_size: int = 8192,
        stop: list[str] | None = None,
    ) -> tuple[str, int, int]:
        token_ids = self.encode(prompt)
        if not token_ids:
            token_ids = [0]
        stop = stop or []

        # The reference state is mutated in-place; keep each request isolated.
        # Use batched sequence prefill so long prompts do not pay one Python
        # dispatch per token.
        state = self.generate_initial_state(1)
        logits = None
        chunk_size = max(1, int(prefill_chunk_size))
        for start in range(0, len(token_ids), chunk_size):
            chunk = token_ids[start:start + chunk_size]
            logits = self.model.forward_batch([chunk], state)
            if torch.is_tensor(logits) and logits.dim() == 2:
                logits = logits[0]

        generated: list[int] = []
        text = ""
        rapid = get_rapid_sampler() if use_rapid_sampling and temperature > 0 else None
        rapid_states = rapid.setup_rand(int(time.time_ns() % (2**31 - 1)), 1) if rapid is not None else None
        rapid_penalties = (
            torch.zeros((1, int(logits.numel())), dtype=torch.float32, device=logits.device)
            if rapid is not None
            else None
        )
        stop_check_tokens = 128
        for _ in range(max_tokens):
            assert logits is not None
            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            elif rapid is not None and rapid_states is not None and rapid_penalties is not None:
                sample_logits = logits.float().contiguous().view(1, -1)
                next_id = int(
                    rapid.batch_sampling_repetition_temperature_topk_topp(
                        sample_logits,
                        rapid_penalties,
                        rapid_states,
                        float(presence_penalty),
                        float(repetition_penalty),
                        float(penalty_decay),
                        float(temperature),
                        int(top_k),
                        float(top_p),
                    )[0].item()
                )
            else:
                next_id = int(
                    self.sample_logits(
                        logits.clone(),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k),
                    )
                )
            generated.append(next_id)
            logits = self.model.forward_batch([[next_id]], state)
            if torch.is_tensor(logits) and logits.dim() == 2:
                logits = logits[0]
            if next_id == 0:
                break
            window_text = self.decode(generated[-stop_check_tokens:])
            if any(s and s in window_text for s in stop):
                break

        text = self.decode(generated)
        for s in stop:
            if s and s in text:
                text = text.split(s, 1)[0]
        return text, len(token_ids), len(generated)


def rwkv_worker_main(
    device_id: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    engine_kwargs: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ.setdefault("TMPDIR", str(WORKSPACE_ROOT / ".tmp"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(WORKSPACE_ROOT / ".torch_extensions"))
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
    os.chdir(WORKSPACE_ROOT)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    engine = RWKVEngine(**engine_kwargs)
    while True:
        item = request_queue.get()
        if item is None:
            break
        job_id, prompt, gen_kwargs = item
        try:
            text, input_tokens, output_tokens = engine.generate(prompt, **gen_kwargs)
            response_queue.put((job_id, True, text, input_tokens, output_tokens, None))
        except Exception as exc:
            response_queue.put((job_id, False, "", 0, 0, repr(exc)))


class RWKVWorkerPool:
    def __init__(self, devices: list[str], engine_kwargs: dict[str, Any]) -> None:
        if not devices:
            raise ValueError("RWKVWorkerPool requires at least one device")
        ctx = mp.get_context("spawn")
        self.response_queue: mp.Queue = ctx.Queue()
        self.request_queues: list[mp.Queue] = []
        self.processes: list[mp.Process] = []
        self.lock = threading.Lock()
        self.next_worker = 0
        self.next_job_id = 0
        self.pending: dict[int, queue.Queue] = {}

        for device in devices:
            request_queue: mp.Queue = ctx.Queue()
            process = ctx.Process(
                target=rwkv_worker_main,
                args=(device, request_queue, self.response_queue, engine_kwargs),
                daemon=True,
            )
            process.start()
            self.request_queues.append(request_queue)
            self.processes.append(process)

        self.response_thread = threading.Thread(target=self.collect_responses, daemon=True)
        self.response_thread.start()

    def collect_responses(self) -> None:
        while True:
            job_id, ok, text, input_tokens, output_tokens, error = self.response_queue.get()
            with self.lock:
                waiter = self.pending.pop(job_id, None)
            if waiter is not None:
                waiter.put((ok, text, input_tokens, output_tokens, error))

    def submit(self, prompt: str, gen_kwargs: dict[str, Any]) -> tuple[str, int, int]:
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self.lock:
            job_id = self.next_job_id
            self.next_job_id += 1
            worker_id = self.next_worker
            self.next_worker = (self.next_worker + 1) % len(self.request_queues)
            self.pending[job_id] = waiter
        self.request_queues[worker_id].put((job_id, prompt, gen_kwargs))
        ok, text, input_tokens, output_tokens, error = waiter.get()
        if not ok:
            raise RuntimeError(error)
        return text, input_tokens, output_tokens


class RWKVLocalClient(BaseLM):
    _engines: dict[str, RWKVEngine] = {}
    _pools: dict[str, RWKVWorkerPool] = {}
    _engines_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "rwkv7-g1e-7.2b",
        model_path: str = str(MODEL_STEM),
        vocab_path: str = str(VOCAB_PATH),
        reference_dir: str = str(REFERENCE_DIR),
        engine_module: str = "rwkv7",
        state_path: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 0,
        presence_penalty: float = 1.0,
        repetition_penalty: float = 0.1,
        penalty_decay: float = 0.996,
        use_rapid_sampling: bool = True,
        stop: list[str] | None = None,
        max_workers: int = 1,
        worker_devices: list[str] | str | None = None,
        prompt_format: str = "instruction",
        response_prefix: str = "",
        raw_log_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.reference_dir = reference_dir
        self.engine_module = engine_module
        self.state_path = state_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.penalty_decay = penalty_decay
        self.use_rapid_sampling = use_rapid_sampling
        self.stop = stop or ["\nUSER:", "\nUser:", "\nSYSTEM:", "\nSystem:"]
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        if isinstance(worker_devices, str):
            worker_devices = [x.strip() for x in worker_devices.split(",") if x.strip()]
        self.worker_devices = worker_devices
        self.prompt_format = prompt_format
        self.response_prefix = response_prefix
        self.raw_log_path = raw_log_path
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    @property
    def engine(self) -> RWKVEngine:
        key = (
            f"{self.model_path}|{self.vocab_path}|{self.reference_dir}|"
            f"{self.engine_module}|{self.state_path}|{os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        )
        with self._engines_lock:
            if key not in self._engines:
                self._engines[key] = RWKVEngine(
                    model_path=self.model_path,
                    vocab_path=self.vocab_path,
                    reference_dir=self.reference_dir,
                    engine_module=self.engine_module,
                    state_path=self.state_path,
                )
            return self._engines[key]

    @property
    def pool(self) -> RWKVWorkerPool | None:
        if not self.worker_devices:
            return None
        key = (
            f"{self.model_path}|{self.vocab_path}|{self.reference_dir}|"
            f"{self.engine_module}|{self.state_path}|{','.join(self.worker_devices)}"
        )
        with self._engines_lock:
            if key not in self._pools:
                self._pools[key] = RWKVWorkerPool(
                    devices=list(self.worker_devices),
                    engine_kwargs={
                        "model_path": self.model_path,
                        "vocab_path": self.vocab_path,
                        "reference_dir": self.reference_dir,
                        "engine_module": self.engine_module,
                        "state_path": self.state_path,
                    },
                )
            return self._pools[key]

    def format_prompt(self, prompt: str | list[dict[str, Any]] | dict[str, Any]) -> str:
        if self.prompt_format == "instruction":
            instruction, input_text = messages_to_instruction(prompt)
            return (
                f"Instruction: {instruction}\n\n"
                f"Input: {input_text}\n\n"
                f"Response: {self.response_prefix}"
            )
        if self.prompt_format == "qa":
            body = normalize_messages(prompt)
            return f"{body}\n\nAssistant: {self.response_prefix}"
        raise ValueError(f"Unknown RWKV prompt_format: {self.prompt_format}")

    def completion(self, prompt: str | list[dict[str, Any]] | dict[str, Any]) -> str:
        prompt_text = self.format_prompt(prompt)
        gen_kwargs = {
            "max_tokens": int(self.sampling_args.get("max_tokens", self.max_tokens)),
            "temperature": float(self.sampling_args.get("temperature", self.temperature)),
            "top_p": float(self.sampling_args.get("top_p", self.top_p)),
            "top_k": int(self.sampling_args.get("top_k", self.top_k)),
            "presence_penalty": float(self.sampling_args.get("presence_penalty", self.presence_penalty)),
            "repetition_penalty": float(self.sampling_args.get("repetition_penalty", self.repetition_penalty)),
            "penalty_decay": float(self.sampling_args.get("penalty_decay", self.penalty_decay)),
            "use_rapid_sampling": bool(self.sampling_args.get("use_rapid_sampling", self.use_rapid_sampling)),
            "prefill_chunk_size": int(self.sampling_args.get("prefill_chunk_size", 8192)),
            "stop": self.sampling_args.get("stop", self.stop),
        }
        pool = self.pool
        t0 = time.perf_counter()
        if pool is not None:
            text, in_tokens, out_tokens = pool.submit(prompt_text, gen_kwargs)
        else:
            with self.engine.lock:
                text, in_tokens, out_tokens = self.engine.generate(
                    prompt_text,
                    **gen_kwargs,
                )
        elapsed = time.perf_counter() - t0
        cleaned = self.postprocess(text)
        self.model_call_counts[self.model_name] += 1
        self.model_input_tokens[self.model_name] += in_tokens
        self.model_output_tokens[self.model_name] += out_tokens
        self.last_prompt_tokens = in_tokens
        self.last_completion_tokens = out_tokens
        self.log_raw_generation(
            prompt_text=prompt_text,
            raw_response=text,
            cleaned_response=cleaned,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            elapsed_sec=elapsed,
        )
        return cleaned

    def log_raw_generation(
        self,
        *,
        prompt_text: str,
        raw_response: str,
        cleaned_response: str,
        input_tokens: int,
        output_tokens: int,
        elapsed_sec: float,
    ) -> None:
        if not self.raw_log_path:
            return
        row = {
            "model": self.model_name,
            "elapsed_sec": elapsed_sec,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt": prompt_text,
            "raw_response": raw_response,
            "cleaned_response": cleaned_response,
        }
        path = Path(self.raw_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _raw_log_file_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def acompletion(self, prompt: str | list[dict[str, Any]] | dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.completion, prompt)

    def postprocess(self, text: str) -> str:
        text = text.replace("<|endoftext|>", "").strip()
        text = text.replace("</think>", "").strip()
        # Some RWKV completions continue a chat transcript. Keep only the first answer.
        text = re.split(r"\n(?:USER|User|SYSTEM|System):", text, maxsplit=1)[0].strip()
        return text

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=self.model_call_counts[model],
                    total_input_tokens=self.model_input_tokens[model],
                    total_output_tokens=self.model_output_tokens[model],
                )
                for model in self.model_call_counts
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
