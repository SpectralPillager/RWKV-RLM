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


TASKS = [
    {
        "id": "answer-ready-final",
        "question": (
            "In the LocalREPL completion mechanism, what action signals that the RLM "
            "has a final answer ready?"
        ),
        "choices": {
            "A": "Returning a Python value from the last expression in the REPL block.",
            "B": "Setting answer[\"content\"] and then setting answer[\"ready\"] = True.",
            "C": "Printing a line that starts with FINAL:",
            "D": "Raising StopIteration from execute_code().",
        },
        "answer": "B",
    },
    {
        "id": "batch-subcalls",
        "question": (
            "Which function exposed in the local REPL runs multiple plain LLM prompts "
            "concurrently and returns the responses in input order?"
        ),
        "choices": {
            "A": "SHOW_VARS",
            "B": "rlm_query",
            "C": "llm_query_batched",
            "D": "format_iteration",
        },
        "answer": "C",
    },
    {
        "id": "python-block-parser",
        "question": (
            "In this benchmark runner, what fenced code block languages are accepted "
            "as executable REPL blocks?"
        ),
        "choices": {
            "A": "Only repl.",
            "B": "Only python.",
            "C": "Both repl and python.",
            "D": "Any language tag.",
        },
        "answer": "C",
    },
]


PREFERRED_FILES = [
    "rlm/environments/local_repl.py",
    "rlm/core/rlm.py",
    "rlm/core/lm_handler.py",
    "rlm/utils/parsing.py",
    "rlm/utils/prompts.py",
    "rlm/clients/base_lm.py",
    "rlm/clients/__init__.py",
    "rlm/core/comms_utils.py",
    "rlm/environments/base_env.py",
    "tests/test_lm_handler.py",
    "tests/test_local_repl.py",
    "tests/test_rlm_query.py",
    "README.md",
    "docs/architecture.md",
    "docs/getting-started.md",
]


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


def extract_choice(text: str) -> str | None:
    # Prefer explicit "Answer: X"; otherwise use the last standalone A-D.
    explicit = re.search(r"(?i)answer\s*[:\-]?\s*([ABCD])\b", text)
    if explicit:
        return explicit.group(1).upper()
    hits = re.findall(r"\b([ABCD])\b", text.upper())
    return hits[-1] if hits else None


def format_question(task: dict[str, Any]) -> str:
    choices = "\n".join(f"{k}. {v}" for k, v in task["choices"].items())
    return (
        f"{task['question']}\n\n{choices}\n\n"
        "Return only the single best option letter: A, B, C, or D."
    )


def load_repo_files() -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for rel in PREFERRED_FILES:
        path = RLM_MAIN / rel
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            text = text.encode("ascii", errors="replace").decode("ascii")
            files.extend(split_file(rel, text))
    # Add more Python/docs files deterministically for 64k contexts.
    seen = {rel for rel, _ in files}
    for path in sorted(RLM_MAIN.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(RLM_MAIN))
        if rel in seen:
            continue
        if any(part.startswith(".") for part in Path(rel).parts):
            continue
        if path.suffix not in {".py", ".md", ".toml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = text.encode("ascii", errors="replace").decode("ascii")
        files.extend(split_file(rel, text))
        seen.add(rel)
    return files


def split_file(rel: str, text: str, chunk_chars: int = 5000) -> list[tuple[str, str]]:
    if len(text) <= chunk_chars:
        return [(rel, text)]
    chunks = []
    for i in range(0, len(text), chunk_chars):
        chunks.append((f"{rel}#chunk={i // chunk_chars}", text[i : i + chunk_chars]))
    return chunks


def build_context(client: RWKVLocalClient, target_tokens: int) -> tuple[str, int]:
    sections: list[str] = []
    files = load_repo_files()
    i = 0
    while True:
        rel, text = files[i % len(files)]
        sections.append(f"\n\n===== FILE: {rel} =====\n{text}")
        context = "".join(sections)
        token_count = len(client.engine.encode(context))
        if token_count >= target_tokens:
            return context, token_count
        i += 1


def direct_prompt(context: str, task: dict[str, Any]) -> str:
    return (
        "You are answering a codebase understanding multiple-choice question. "
        "Use the repository context to answer.\n\n"
        f"Repository context:\n{context}\n\n"
        f"Question:\n{format_question(task)}"
    )


RLM_SYSTEM_PROMPT = '''You are a Python REPL controller for repository understanding.

Rules:
- Output exactly one complete code block labeled repl.
- The variable `context` already exists and contains repository files with headers like `===== FILE: path =====`.
- Use simple Python string search to extract relevant snippets from `context`.
- Then call `llm_query` once on a compact prompt containing those snippets and the multiple-choice question.
- To finish, set `answer["content"]` to the single option letter and then set `answer["ready"] = True`.
- Keep the code short.

Example:
```repl
import re
parts = [b[:5000] for b in context.split("===== FILE: ") if "answer" in b or "ready" in b or "llm_query_batched" in b]
resp = llm_query("Context:\\n" + "\\n".join(parts)[:12000] + "\\nQuestion:\\n" + root_question)
m = re.search(r"\\b([ABCD])\\b", resp.upper())
answer["content"] = m.group(1) if m else "?"
answer["ready"] = True
```
'''


def make_custom_tools(question: str) -> dict[str, Any]:
    return {"root_question": question}


def run_direct(client: RWKVLocalClient, context: str, task: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    response = client.completion(direct_prompt(context, task))
    elapsed = time.perf_counter() - t0
    parsed = extract_choice(response)
    return {
        "mode": "direct",
        "elapsed_sec": elapsed,
        "response": response,
        "parsed": parsed,
        "expected": task["answer"],
        "ok": parsed == task["answer"],
        "usage": client.get_usage_summary().to_dict(),
    }


def run_rlm(backend_kwargs: dict[str, Any], context: str, task: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    question = format_question(task)
    rlm = RLM(
        backend="rwkv-local",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=2,
        max_concurrent_subcalls=4,
        custom_system_prompt=RLM_SYSTEM_PROMPT,
        custom_tools=make_custom_tools(question),
        orchestrator=False,
        verbose=False,
    )
    result = rlm.completion(context, root_prompt=question)
    elapsed = time.perf_counter() - t0
    parsed = extract_choice(result.response)
    return {
        "mode": "rlm-repl",
        "elapsed_sec": elapsed,
        "response": result.response,
        "parsed": parsed,
        "expected": task["answer"],
        "ok": parsed == task["answer"],
        "usage": result.usage_summary.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,16384,32768,65536")
    parser.add_argument("--modes", default="direct,rlm")
    parser.add_argument("--task-ids", default=",".join(t["id"] for t in TASKS))
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output", default="bench_outputs/repo_context_bench.json")
    parser.add_argument("--raw-output", default="bench_outputs/repo_context_generations.jsonl")
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
    task_ids = {x.strip() for x in args.task_ids.split(",") if x.strip()}
    tasks = [t for t in TASKS if t["id"] in task_ids]
    modes = {x.strip() for x in args.modes.split(",") if x.strip()}
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    results: list[dict[str, Any]] = []
    for length in lengths:
        context, actual_tokens = build_context(client, length)
        for task in tasks:
            base = {
                "target_tokens": length,
                "actual_context_tokens": actual_tokens,
                "task_id": task["id"],
                "question": task["question"],
                "expected": task["answer"],
            }
            if "direct" in modes:
                row = {**base, **run_direct(client, context, task)}
                results.append(row)
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if "rlm" in modes:
                row = {**base, **run_rlm(backend_kwargs, context, task)}
                results.append(row)
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
