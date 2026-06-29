from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RLM_MAIN = ROOT / "rlm-main"
SITE_PACKAGES = ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
REFERENCE_DIR = ROOT / "third_party" / "reference"
SOTA_REFERENCE_DIR = ROOT / "third_party" / "sota_reference"
for path in [str(RLM_MAIN), str(SITE_PACKAGES), str(ROOT), str(REFERENCE_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("TMPDIR", str(ROOT / ".tmp"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / ".torch_extensions"))

from rlm import RLM
import rlm.core.rlm as rlm_core
import rlm.environments.local_repl as local_repl
import rlm.utils.parsing as rlm_parsing
from utils import TRIE_TOKENIZER

from local_rlm.rwkv_client import RWKVLocalClient, VOCAB_PATH


NOISE_SENTENCE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."


def patch_rwkv_backend() -> None:
    def get_client(backend: str, backend_kwargs: dict[str, Any]) -> RWKVLocalClient:
        if backend != "rwkv-local":
            raise ValueError(f"This runner only registers rwkv-local, got {backend!r}")
        return RWKVLocalClient(**backend_kwargs)

    rlm_core.get_client = get_client

    def find_code_blocks_allow_python(text: str) -> list[str]:
        pattern = r"```(?:repl|python)\s*\n(.*?)\n```"
        return [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]

    rlm_core.find_code_blocks = find_code_blocks_allow_python
    rlm_parsing.find_code_blocks = find_code_blocks_allow_python

    @contextmanager
    def stable_cwd(self):
        yield

    local_repl.LocalREPL._temp_cwd = stable_cwd


def rand_var(rng: random.Random, used: set[str], n: int = 5) -> str:
    while True:
        value = "".join(rng.choices(string.ascii_uppercase, k=n))
        if value not in used:
            used.add(value)
            return value


def rand_key(rng: random.Random) -> str:
    adjectives = ["amber", "brisk", "crimson", "daring", "emerald", "frozen", "gentle", "hidden"]
    nouns = ["anchor", "bison", "canyon", "delta", "ember", "forest", "harbor", "island"]
    return f"{rng.choice(adjectives)}-{rng.choice(nouns)}-{rng.randint(100, 999)}"


def count_tokens(tokenizer: TRIE_TOKENIZER, text: str) -> int:
    return len(tokenizer.encode(text))


def fit_to_length(tokenizer: TRIE_TOKENIZER, prompt_without_context: str, context_lines: list[str], target: int) -> str:
    low, high = 1, max(2, len(context_lines))
    best = "\n".join(context_lines[:1])
    while low <= high:
        mid = (low + high) // 2
        context = "\n".join(context_lines[:mid])
        total = count_tokens(tokenizer, prompt_without_context.format(context=context))
        if total <= target:
            best = context
            low = mid + 1
        else:
            high = mid - 1
    return best


def make_vt_sample(tokenizer: TRIE_TOKENIZER, target_tokens: int, index: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + index * 9973 + target_tokens)
    used: set[str] = set()
    value = str(rng.randint(10000, 99999))
    vars_ = [rand_var(rng, used) for _ in range(5)]
    chain = [f"VAR {vars_[0]} = {value}."] + [f"VAR {vars_[i]} = VAR {vars_[i - 1]}." for i in range(1, len(vars_))]

    template = (
        "Memorize and track the chain of variable assignments hidden in the following text.\n\n"
        "{context}\n\n"
        f"Question: Find all variables that are assigned the value {value} in the text above. "
        "Return only the variable names separated by commas."
    )
    base_lines = [NOISE_SENTENCE for _ in range(max(100, target_tokens // 4))]
    prefix_context = fit_to_length(tokenizer, template, base_lines, target_tokens)
    prefix_lines = max(30, len(prefix_context.splitlines()))
    max_insert = min(len(base_lines) - 10, max(15, int(prefix_lines * 0.75)))
    insert_positions = sorted(rng.sample(range(10, max_insert), len(chain)))
    for offset, (pos, line) in enumerate(zip(insert_positions, chain, strict=True)):
        base_lines.insert(pos + offset, line)

    context = fit_to_length(tokenizer, template, base_lines, target_tokens)
    if not all(line in context for line in chain):
        raise RuntimeError(f"failed to retain vt chain for sample {index} at {target_tokens} tokens")
    prompt = template.format(context=context)
    return {
        "id": f"vt_{target_tokens}_{index}",
        "task": "vt",
        "target_tokens": target_tokens,
        "input": prompt,
        "context": context,
        "query": value,
        "outputs": vars_,
        "length": count_tokens(tokenizer, prompt),
    }


def make_niah_sample(tokenizer: TRIE_TOKENIZER, target_tokens: int, index: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + index * 7919 + target_tokens)
    pairs = [(rand_key(rng), str(rng.randint(1000000, 9999999))) for _ in range(4)]
    query_key, answer = rng.choice(pairs)
    needles = [f"One of the special magic numbers for {key} is: {value}." for key, value in pairs]

    template = (
        "Some special magic numbers are hidden within the following text. Make sure to memorize them.\n\n"
        "{context}\n\n"
        f"What is the special magic number for {query_key} mentioned in the provided text? "
        "Return only the number."
    )
    base_lines = [NOISE_SENTENCE for _ in range(max(100, target_tokens // 4))]
    prefix_context = fit_to_length(tokenizer, template, base_lines, target_tokens)
    prefix_lines = max(30, len(prefix_context.splitlines()))
    max_insert = min(len(base_lines) - 10, max(15, int(prefix_lines * 0.75)))
    insert_positions = sorted(rng.sample(range(10, max_insert), len(needles)))
    for offset, (pos, line) in enumerate(zip(insert_positions, needles, strict=True)):
        base_lines.insert(pos + offset, line)

    context = fit_to_length(tokenizer, template, base_lines, target_tokens)
    if not all(line in context for line in needles):
        raise RuntimeError(f"failed to retain niah needles for sample {index} at {target_tokens} tokens")
    prompt = template.format(context=context)
    return {
        "id": f"niah_multikey_{target_tokens}_{index}",
        "task": "niah_multikey",
        "target_tokens": target_tokens,
        "input": prompt,
        "context": context,
        "query": query_key,
        "outputs": [answer],
        "length": count_tokens(tokenizer, prompt),
    }


def generate_samples(
    lengths: list[int],
    tasks: list[str],
    sample_start: int,
    samples_per_task: int,
    seed: int,
) -> list[dict[str, Any]]:
    tokenizer = TRIE_TOKENIZER(str(VOCAB_PATH))
    rows: list[dict[str, Any]] = []
    for target_tokens in lengths:
        for task in tasks:
            for i in range(sample_start, sample_start + samples_per_task):
                if task == "vt":
                    rows.append(make_vt_sample(tokenizer, target_tokens, i, seed))
                elif task == "niah_multikey":
                    rows.append(make_niah_sample(tokenizer, target_tokens, i, seed))
                else:
                    raise ValueError(f"unknown task {task}")
    return rows


def parse_answer(task: str, text: str) -> list[str]:
    text = text.strip()
    if task == "vt":
        return re.findall(r"\b[A-Z]{5}\b", text.upper())
    if task == "niah_multikey":
        return re.findall(r"\b\d{7}\b", text)
    raise ValueError(task)


def is_correct(task: str, response: str, expected: list[str]) -> bool:
    parsed = parse_answer(task, response)
    if task == "vt":
        return set(parsed) == set(expected)
    return any(x in expected for x in parsed)


def run_direct(client: RWKVLocalClient, sample: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    response = client.completion(sample["input"])
    elapsed = time.perf_counter() - t0
    parsed = parse_answer(sample["task"], response)
    return {
        "mode": "direct",
        "elapsed_sec": elapsed,
        "response": response,
        "parsed": parsed,
        "ok": is_correct(sample["task"], response, sample["outputs"]),
        "usage": client.get_usage_summary().to_dict(),
    }


RLM_SYSTEM_PROMPT = '''You are a Python REPL controller for long-context synthetic QA.

Rules:
- Output exactly one complete code block labeled repl.
- `doc_text` contains the full text.
- `task`, `query`, and `expected_format` describe what to extract.
- Use Python string or regex operations on `doc_text`.
- Set `answer["content"]` to the final short answer and `answer["ready"] = True`.

Examples:
```repl
import re
assignments = re.findall(r"VAR ([A-Z]{{5}}) = (?:VAR ([A-Z]{{5}})|(\\d+))\\.", doc_text)
known = {{str(query)}}
changed = True
while changed:
    changed = False
    for lhs, rhs_var, rhs_val in assignments:
        if lhs not in known and (rhs_val in known or rhs_var in known):
            known.add(lhs)
            changed = True
hits = [lhs for lhs, _, _ in assignments if lhs in known]
answer["content"] = ", ".join(hits)
answer["ready"] = True
```
'''


def run_rlm(backend_kwargs: dict[str, Any], sample: dict[str, Any], mode_label: str = "rlm-repl") -> dict[str, Any]:
    t0 = time.perf_counter()
    custom_tools = {
        "doc_text": sample["context"],
        "task": sample["task"],
        "query": sample["query"],
        "expected_format": "comma separated variable names" if sample["task"] == "vt" else "single 7 digit number",
    }
    prompt = (
        f"Task: {sample['task']}\n"
        f"Query: {sample['query']}\n"
        f"Expected answer format: {custom_tools['expected_format']}\n"
        "Use the Python REPL to extract the answer from context."
    )
    rlm = RLM(
        backend="rwkv-local",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=2,
        custom_system_prompt=RLM_SYSTEM_PROMPT,
        custom_tools=custom_tools,
        orchestrator=False,
        verbose=False,
    )
    result = rlm.completion(sample["input"], root_prompt=prompt)
    elapsed = time.perf_counter() - t0
    parsed = parse_answer(sample["task"], result.response)
    return {
        "mode": mode_label,
        "elapsed_sec": elapsed,
        "response": result.response,
        "parsed": parsed,
        "ok": is_correct(sample["task"], result.response, sample["outputs"]),
        "usage": result.usage_summary.to_dict(),
    }


def make_base_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "task": sample["task"],
        "target_tokens": sample["target_tokens"],
        "length": sample["length"],
        "query": sample["query"],
        "expected": sample["outputs"],
    }


def run_one_job(mode: str, backend_kwargs: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    if mode == "direct":
        client = RWKVLocalClient(**backend_kwargs)
        return {**make_base_row(sample), **run_direct(client, sample)}
    if mode == "rlm":
        return {**make_base_row(sample), **run_rlm(backend_kwargs, sample)}
    if mode == "rlm_tuned":
        return {**make_base_row(sample), **run_rlm(backend_kwargs, sample, mode_label="rlm-repl-tuned")}
    raise ValueError(f"unknown mode {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,16384")
    parser.add_argument("--tasks", default="vt,niah_multikey")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--modes", default="direct,rlm")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="bench_outputs/ruler_synthetic_smoke.json")
    parser.add_argument("--raw-output", default="bench_outputs/ruler_synthetic_smoke_generations.jsonl")
    parser.add_argument("--parallelism", type=int, default=10)
    parser.add_argument("--worker-devices", default="0,1")
    parser.add_argument("--model-name", default="rwkv7-g1e-7.2b-ctx8192")
    parser.add_argument("--model-path", default="/data_temp/mnt/raid5/zjx/rwkv/full-training/rwkv7-g1e-7.2b-20260301-ctx8192")
    parser.add_argument("--reference-dir", default=str(SOTA_REFERENCE_DIR))
    parser.add_argument("--engine-module", default="rwkv7", choices=["rwkv7", "rwkv7_fp16"])
    parser.add_argument("--state-path", default="")
    args = parser.parse_args()

    patch_rwkv_backend()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    modes = {x.strip() for x in args.modes.split(",") if x.strip()}
    samples = generate_samples(lengths, tasks, args.sample_start, args.samples_per_task, args.seed)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")
    jsonl_path.write_text("", encoding="utf-8")
    raw_path = ROOT / args.raw_output
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("", encoding="utf-8")

    data_path = ROOT / "bench_data" / f"{output_path.stem}_samples.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
        encoding="utf-8",
    )

    backend_kwargs = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "reference_dir": args.reference_dir,
        "engine_module": args.engine_module,
        "max_tokens": args.max_tokens,
        "temperature": 0.3,
        "top_p": 0.4,
        "top_k": 20,
        "presence_penalty": 1.0,
        "repetition_penalty": 0.1,
        "penalty_decay": 0.996,
        "use_rapid_sampling": True,
        "max_workers": 2,
        "worker_devices": args.worker_devices,
        "prompt_format": "instruction",
        "response_prefix": "",
        "stop": ["\nUser:", "\nInstruction:", "🤖 Repl #2:"],
        "raw_log_path": str(raw_path),
    }
    tuned_backend_kwargs = dict(backend_kwargs)
    if args.state_path:
        tuned_backend_kwargs["state_path"] = args.state_path
        tuned_backend_kwargs["model_name"] = f"{args.model_name}-state"
    results: list[dict[str, Any]] = []
    jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for sample in samples:
        if "direct" in modes:
            jobs.append(("direct", backend_kwargs, sample))
        if "rlm" in modes:
            jobs.append(("rlm", backend_kwargs, sample))
        if "rlm_tuned" in modes:
            if not args.state_path:
                raise ValueError("--state-path is required for mode rlm_tuned")
            jobs.append(("rlm_tuned", tuned_backend_kwargs, sample))
    with ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as executor:
        future_to_job = {
            executor.submit(run_one_job, mode, job_backend_kwargs, sample): (mode, sample)
            for mode, job_backend_kwargs, sample in jobs
        }
        for future in as_completed(future_to_job):
            mode, sample = future_to_job[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    **make_base_row(sample),
                    "mode": "rlm-repl-tuned" if mode == "rlm_tuned" else ("rlm-repl" if mode == "rlm" else mode),
                    "elapsed_sec": 0.0,
                    "response": "",
                    "parsed": [],
                    "ok": False,
                    "error": repr(exc),
                }
            results.append(row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
