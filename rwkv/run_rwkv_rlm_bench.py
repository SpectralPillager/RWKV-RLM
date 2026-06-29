from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RLM_MAIN = ROOT / "rlm-main"
SITE_PACKAGES = ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
for path in [str(RLM_MAIN), str(SITE_PACKAGES), str(ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("TMPDIR", str(ROOT / ".tmp"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / ".torch_extensions"))

from rlm import RLM
import rlm.core.rlm as rlm_core

from local_rlm.rwkv_client import RWKVLocalClient


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


def extract_int(text: str) -> int | None:
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    if not nums:
        return None
    return int(nums[-1])


def make_needle_context(n_items: int, needle_index: int, answer: int) -> str:
    lines = []
    for i in range(n_items):
        value = (i * 7919 + 17) % 100000
        if i == needle_index:
            lines.append(f"record {i:05d}: target_code = {answer}")
        else:
            lines.append(f"record {i:05d}: filler_value = {value}")
    return "\n".join(lines)


def make_sum_context(n_items: int) -> tuple[str, int]:
    values = [((i * 37 + 11) % 997) for i in range(n_items)]
    lines = [f"item {i:05d}: value={v}" for i, v in enumerate(values)]
    return "\n".join(lines), sum(values)


def run_direct(client: RWKVLocalClient, prompt: str, expected: str | int | None) -> dict[str, Any]:
    t0 = time.perf_counter()
    response = client.completion(prompt)
    elapsed = time.perf_counter() - t0
    parsed = extract_int(response)
    ok = None
    if isinstance(expected, int):
        ok = parsed == expected
    elif isinstance(expected, str):
        ok = expected.lower() in response.lower()
    return {
        "mode": "direct",
        "elapsed_sec": elapsed,
        "response": response,
        "parsed_int": parsed,
        "expected": expected,
        "ok": ok,
        "usage": client.get_usage_summary().to_dict(),
    }


def run_rlm(
    *,
    context: str,
    root_prompt: str,
    backend_kwargs: dict[str, Any],
    max_iterations: int,
    max_depth: int,
    expected: int | str | None,
    custom_system_prompt: str | None = None,
    orchestrator: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    rlm = RLM(
        backend="rwkv-local",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_concurrent_subcalls=4,
        verbose=False,
        custom_system_prompt=custom_system_prompt,
        orchestrator=orchestrator,
    )
    result = rlm.completion(context, root_prompt=root_prompt)
    elapsed = time.perf_counter() - t0
    parsed = extract_int(result.response)
    ok = None
    if isinstance(expected, int):
        ok = parsed == expected
    elif isinstance(expected, str):
        ok = expected.lower() in result.response.lower()
    return {
        "mode": f"rlm-depth-{max_depth}",
        "elapsed_sec": elapsed,
        "response": result.response,
        "parsed_int": parsed,
        "expected": expected,
        "ok": ok,
        "usage": result.usage_summary.to_dict(),
    }


def append_result(output_path: Path, result: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_batched_probe(backend_kwargs: dict[str, Any], n_prompts: int = 4) -> dict[str, Any]:
    from rlm.core.comms_utils import LMRequest, send_lm_request_batched
    from rlm.core.lm_handler import LMHandler

    client = RWKVLocalClient(**backend_kwargs)
    prompts = [
        f"Return only the integer result of {17 + i} + {200 + i}."
        for i in range(n_prompts)
    ]
    t0 = time.perf_counter()
    with LMHandler(client=client, batch_max_concurrent=n_prompts) as handler:
        responses = send_lm_request_batched(handler.address, prompts)
    elapsed = time.perf_counter() - t0
    rows = []
    for prompt, response in zip(prompts, responses, strict=True):
        text = response.chat_completion.response if response.success else response.error
        rows.append(
            {
                "prompt": prompt,
                "success": response.success,
                "response": text,
                "parsed_int": extract_int(text or ""),
            }
        )
    return {
        "mode": "batched-llm-handler",
        "elapsed_sec": elapsed,
        "n_prompts": n_prompts,
        "rows": rows,
        "usage": client.get_usage_summary().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="", help="Comma-separated GPU IDs for worker pool.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--tasks",
        default="direct,batched,rlm",
        help="Comma-separated subset of: direct,batched,rlm.",
    )
    parser.add_argument("--output", default="bench_outputs/rwkv_rlm_bench.json")
    args = parser.parse_args()

    patch_rwkv_backend()
    backend_kwargs: dict[str, Any] = {
        "model_name": "rwkv7-g1e-7.2b-ctx8192",
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_workers": 4,
        "prompt_format": "instruction",
        "response_prefix": "<think></think>\n",
    }
    if args.devices:
        backend_kwargs["worker_devices"] = args.devices

    results: list[dict[str, Any]] = []
    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".jsonl").write_text("", encoding="utf-8")

    # Direct baseline: plain LM call over the full context.
    n_items = 80 if args.quick else 3000
    needle_idx = n_items - 37
    needle_answer = 424242
    needle_context = make_needle_context(n_items, needle_idx, needle_answer)
    direct_prompt = (
        "Find the target_code in the records. Return only the integer target_code.\n\n"
        + needle_context
    )
    if "direct" in tasks:
        result = {
            "task": "needle-direct",
            **run_direct(RWKVLocalClient(**backend_kwargs), direct_prompt, needle_answer),
        }
        results.append(result)
        append_result(output_path, result)

    # RLM/REPL task: solvable by inspecting context and using Python, not model memory.
    sum_context, expected_sum = make_sum_context(40 if args.quick else 1600)
    sum_prompt = (
        "The context contains lines like 'item N: value=V'. "
        "Compute the sum of all V values. Return only the integer sum."
    )
    tiny_rlm_prompt = '''You are a Python REPL controller.

Rules:
- Output exactly one complete code block labeled repl.
- Do not explain.
- Do not copy, redefine, or overwrite `context`.
- The variable `context` already exists and is a string.
- Use Python to compute the answer from `context`.
- To finish, set `answer["content"]` and then `answer["ready"] = True`.

Example:
```repl
import re
nums = [int(x) for x in re.findall(r"value=(\\d+)", context)]
answer["content"] = str(sum(nums))
answer["ready"] = True
```
'''
    if "rlm" in tasks:
        result = {
            "task": "sum-rlm-repl",
            **run_rlm(
                context=sum_context,
                root_prompt=sum_prompt,
                backend_kwargs=backend_kwargs,
                max_iterations=3,
                max_depth=1,
                expected=expected_sum,
                custom_system_prompt=tiny_rlm_prompt,
            ),
        }
        results.append(result)
        append_result(output_path, result)

    # Independent calls through the RLM LMHandler to confirm batched concurrency path.
    if "batched" in tasks:
        result = {"task": "batched-probe", **run_batched_probe(backend_kwargs)}
        results.append(result)
        append_result(output_path, result)

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
