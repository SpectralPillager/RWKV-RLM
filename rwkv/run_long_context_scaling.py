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


def extract_answer(text: str) -> str | None:
    match = re.search(r"ANS-[A-Z0-9-]+", text)
    return match.group(0) if match else None


def make_context(engine: RWKVEngine, target_tokens: int, position: str, answer: str) -> tuple[str, int]:
    filler = (
        "This is a neutral filler record for long context testing. "
        "It contains ordinary words and no target marker. "
    )
    lines: list[str] = []
    while True:
        lines.append(f"record {len(lines):06d}: {filler}")
        token_count = len(engine.encode("\n".join(lines)))
        if token_count >= target_tokens:
            break

    needle = (
        f"record TARGET: The secret retrieval code is {answer}. "
        f"When asked for the secret retrieval code, answer exactly {answer}."
    )
    if position == "front":
        insert_at = max(1, len(lines) // 20)
    elif position == "middle":
        insert_at = len(lines) // 2
    elif position == "back":
        insert_at = max(0, len(lines) - len(lines) // 20)
    else:
        raise ValueError(f"unknown position: {position}")
    lines.insert(insert_at, needle)
    context = "\n".join(lines)
    return context, len(engine.encode(context))


def direct_prompt(context: str) -> str:
    return (
        "Find the secret retrieval code in the text. "
        "Return exactly the code and nothing else.\n\n"
        f"{context}"
    )


RLM_SYSTEM_PROMPT = '''You are a Python REPL controller for long-context retrieval.

Rules:
- Output exactly one complete code block labeled repl.
- Do not explain.
- Do not copy, redefine, or overwrite `context`.
- The variable `context` already exists and is a string.
- Search `context` with Python to find the answer.
- To finish, set `answer["content"]` and then `answer["ready"] = True`.

Example:
```repl
import re
m = re.search(r"ANS-[A-Z0-9-]+", context)
answer["content"] = m.group(0) if m else "NOT_FOUND"
answer["ready"] = True
```
'''


def run_direct(client: RWKVLocalClient, context: str, expected: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    response = client.completion(direct_prompt(context))
    elapsed = time.perf_counter() - t0
    parsed = extract_answer(response)
    return {
        "mode": "direct",
        "elapsed_sec": elapsed,
        "response": response,
        "parsed": parsed,
        "expected": expected,
        "ok": parsed == expected or response.strip() == expected,
        "usage": client.get_usage_summary().to_dict(),
    }


def run_rlm(backend_kwargs: dict[str, Any], context: str, expected: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    rlm = RLM(
        backend="rwkv-local",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=2,
        max_concurrent_subcalls=4,
        custom_system_prompt=RLM_SYSTEM_PROMPT,
        orchestrator=False,
        verbose=False,
    )
    result = rlm.completion(
        context,
        root_prompt="Find the secret retrieval code. Return exactly the code and nothing else.",
    )
    elapsed = time.perf_counter() - t0
    parsed = extract_answer(result.response)
    return {
        "mode": "rlm-repl",
        "elapsed_sec": elapsed,
        "response": result.response,
        "parsed": parsed,
        "expected": expected,
        "ok": parsed == expected or result.response.strip() == expected,
        "usage": result.usage_summary.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="2048,4096,8192,12288")
    parser.add_argument("--positions", default="front,middle,back")
    parser.add_argument("--modes", default="direct,rlm")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", default="bench_outputs/long_context_scaling.json")
    parser.add_argument("--raw-output", default="bench_outputs/long_context_generations.jsonl")
    args = parser.parse_args()

    patch_rwkv_backend()

    backend_kwargs: dict[str, Any] = {
        "model_name": "rwkv7-g1e-7.2b-ctx8192",
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_workers": 4,
        "prompt_format": "instruction",
        "response_prefix": "<think></think>\n",
        "raw_log_path": str(ROOT / args.raw_output),
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")
    jsonl_path.write_text("", encoding="utf-8")

    client = RWKVLocalClient(**backend_kwargs)
    engine = client.engine
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    positions = [x.strip() for x in args.positions.split(",") if x.strip()]
    modes = {x.strip() for x in args.modes.split(",") if x.strip()}

    results: list[dict[str, Any]] = []
    for target_tokens in lengths:
        for position in positions:
            expected = f"ANS-{target_tokens}-{position.upper()}"
            context, actual_tokens = make_context(engine, target_tokens, position, expected)
            row_base = {
                "target_tokens": target_tokens,
                "actual_context_tokens": actual_tokens,
                "position": position,
                "expected": expected,
            }
            if "direct" in modes:
                result = {**row_base, **run_direct(client, context, expected)}
                results.append(result)
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(json.dumps(result, ensure_ascii=False), flush=True)
            if "rlm" in modes:
                result = {**row_base, **run_rlm(backend_kwargs, context, expected)}
                results.append(result)
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(json.dumps(result, ensure_ascii=False), flush=True)

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
