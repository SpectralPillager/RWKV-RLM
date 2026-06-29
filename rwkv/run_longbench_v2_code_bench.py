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
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".hf-cache" / "datasets"))

from datasets import load_dataset
from rlm import RLM
import rlm.core.rlm as rlm_core
import rlm.utils.parsing as rlm_parsing

from local_rlm.rwkv_client import RWKVLocalClient


DEFAULT_IDS = [
    "6708a096bb02136c067d1789",  # ~28k tokens, easy
    "66f3ad93821e116aacb2e29f",  # ~36k tokens, hard
    "66ecf139821e116aacb1e0e1",  # ~57k tokens, hard
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
    rlm_parsing.find_code_blocks = find_code_blocks_allow_python


def ascii_clean(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def format_question(ex: dict[str, Any]) -> str:
    return (
        f"{ex['question']}\n\n"
        f"A. {ex['choice_A']}\n"
        f"B. {ex['choice_B']}\n"
        f"C. {ex['choice_C']}\n"
        f"D. {ex['choice_D']}\n\n"
        "Return only the single best option letter: A, B, C, or D."
    )


def direct_prompt(ex: dict[str, Any]) -> str:
    return (
        "You are answering a code repository understanding multiple-choice question. "
        "Use the full repository context below.\n\n"
        f"Question:\n{format_question(ex)}\n\n"
        f"Repository context:\n{ascii_clean(ex['context'])}"
    )


def extract_choice(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        return None
    explicit = re.search(r"(?i)answer\s*[:\-]?\s*([ABCD])\b", text)
    if explicit:
        return explicit.group(1).upper()
    hits = re.findall(r"\b([ABCD])\b", text.upper())
    return hits[-1] if hits else None


RLM_SYSTEM_PROMPT = '''You are a Python REPL controller for code repository QA.

Rules:
- Output exactly one complete code block labeled repl.
- `context` contains a very large code repository and related text.
- `root_question` contains the multiple-choice question.
- A helper `retrieve_snippets(query: str) -> str` is available. Use it to get compact relevant snippets.
- Then call `llm_query` once with those snippets and the question.
- Set `answer["content"]` to one letter A/B/C/D and set `answer["ready"] = True`.

Example:
```repl
import re
snippet = retrieve_snippets(root_question)
resp = llm_query("Repository snippets:\\n" + snippet + "\\n\\nQuestion:\\n" + root_question)
m = re.search(r"\\b([ABCD])\\b", resp.upper())
answer["content"] = m.group(1) if m else "?"
answer["ready"] = True
```
'''


def run_direct(client: RWKVLocalClient, ex: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    response = client.completion(direct_prompt(ex))
    elapsed = time.perf_counter() - t0
    parsed = extract_choice(response)
    return {
        "mode": "direct",
        "elapsed_sec": elapsed,
        "response": response,
        "parsed": parsed,
        "expected": ex["answer"],
        "ok": parsed == ex["answer"],
        "usage": client.get_usage_summary().to_dict(),
    }


def make_retriever(context: str):
    cleaned_context = ascii_clean(context)

    def retrieve_snippets(query: str) -> str:
        stopwords = {
            "which", "what", "this", "that", "with", "from", "return", "only",
            "single", "best", "option", "letter", "there", "three", "based",
            "analyze", "across", "following", "correct", "regarding",
        }
        words = [
            w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
            if w.lower() not in stopwords
        ]
        # Split on common repo/document boundaries while remaining robust for zipped text dumps.
        blocks = re.split(r"(?m)^=====.*?$|^File: .*?$|^# .*?$", cleaned_context)
        scored = []
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            low = block.lower()
            score = sum(3 for w in words if w in block)
            score += sum(1 for w in words if w.lower() in low)
            if score:
                scored.append((score, i, block[:5000]))
        if not scored:
            return cleaned_context[:18000]
        snippets = [b for _, _, b in sorted(scored, reverse=True)[:8]]
        return "\n\n---SNIPPET---\n\n".join(snippets)[:22000]

    return retrieve_snippets


def run_rlm(backend_kwargs: dict[str, Any], ex: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    question = format_question(ex)
    rlm = RLM(
        backend="rwkv-local",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=2,
        custom_system_prompt=RLM_SYSTEM_PROMPT,
        custom_tools={"root_question": question, "retrieve_snippets": make_retriever(ex["context"])},
        orchestrator=False,
        verbose=False,
    )
    result = rlm.completion(ascii_clean(ex["context"]), root_prompt=question)
    elapsed = time.perf_counter() - t0
    parsed = extract_choice(result.response)
    return {
        "mode": "rlm-repl",
        "elapsed_sec": elapsed,
        "response": result.response,
        "parsed": parsed,
        "expected": ex["answer"],
        "ok": parsed == ex["answer"],
        "usage": result.usage_summary.to_dict(),
    }


def load_examples(ids: list[str]) -> list[dict[str, Any]]:
    ds = load_dataset("zai-org/LongBench-v2", split="train")
    by_id = {ex["_id"]: dict(ex) for ex in ds}
    return [by_id[i] for i in ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default=",".join(DEFAULT_IDS))
    parser.add_argument("--modes", default="rlm")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", default="bench_outputs/longbench_v2_code.json")
    parser.add_argument("--raw-output", default="bench_outputs/longbench_v2_code_generations.jsonl")
    args = parser.parse_args()

    patch_rwkv_backend()
    backend_kwargs = {
        "model_name": "rwkv7-g1e-7.2b-ctx8192",
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_workers": 4,
        "prompt_format": "instruction",
        "response_prefix": "",
        "stop": ["🤖 Repl #2:", "\nUser:", "\nInstruction:"],
        "raw_log_path": str(ROOT / args.raw_output),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")
    jsonl_path.write_text("", encoding="utf-8")
    raw_output_path = ROOT / args.raw_output
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text("", encoding="utf-8")

    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    modes = {x.strip() for x in args.modes.split(",") if x.strip()}
    examples = load_examples(ids)
    client = RWKVLocalClient(**backend_kwargs)
    results = []

    # Use cached token lengths from the earlier scan when available.
    length_rows = {}
    length_path = ROOT / "bench_data" / "longbench_v2_code_lengths.json"
    if length_path.exists():
        for row in json.loads(length_path.read_text(encoding="utf-8")):
            length_rows[row["_id"]] = row

    for ex in examples:
        meta = length_rows.get(ex["_id"], {})
        base = {
            "_id": ex["_id"],
            "domain": ex["domain"],
            "sub_domain": ex["sub_domain"],
            "difficulty": ex["difficulty"],
            "length_label": ex["length"],
            "estimated_tokens": meta.get("tokens"),
            "context_chars": len(ex["context"]),
            "question": ex["question"],
            "expected": ex["answer"],
        }
        if "direct" in modes:
            row = {**base, **run_direct(client, ex)}
            results.append(row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if "rlm" in modes:
            row = {**base, **run_rlm(backend_kwargs, ex)}
            results.append(row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
