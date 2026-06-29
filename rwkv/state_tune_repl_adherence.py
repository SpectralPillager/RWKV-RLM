from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
STATE_TUNING_DIR = ROOT / "third_party" / "state_tuning_min"
REFERENCE_DIR = ROOT / "third_party" / "reference"
MODEL_STEM = Path("/data_temp/mnt/raid5/zjx/rwkv/full-training/rwkv7-g1e-7.2b-20260301-ctx8192")
VOCAB_PATH = REFERENCE_DIR / "rwkv_vocab_v20230424.txt"


def setup_paths_and_env(ctx_len: int) -> None:
    os.environ.setdefault("TMPDIR", str(ROOT / ".tmp"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / ".torch_extensions"))
    os.environ["RWKV_HEAD_SIZE_A"] = "64"
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "fullstate"
    os.environ["RWKV_CTXLEN"] = str(ctx_len)
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"
    os.environ["RWKV_FLOAT_MODE"] = "fp32"
    for path in [str(STATE_TUNING_DIR), str(REFERENCE_DIR), str(ROOT)]:
        if path not in sys.path:
            sys.path.insert(0, path)


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
assignments = re.findall(r"VAR ([A-Z]{5}) = (?:VAR ([A-Z]{5})|(\\d+))\\.", doc_text)
known = {str(query)}
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


TARGET_REPL = '''```repl
import re
assignments = re.findall(r"VAR ([A-Z]{5}) = (?:VAR ([A-Z]{5})|(\\d+))\\.", doc_text)
known = {str(query)}
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


def rwkv_instruction_prompt(system_prompt: str, user_parts: list[str]) -> str:
    conversation = "\n\n".join(f"User: {part}" for part in user_parts)
    return f"Instruction: {system_prompt}\n\nInput: {conversation}\n\nResponse: "


def make_problem(query: str, context_chars: int, *, iteration: int = 1) -> str:
    metadata = (
        f"Answer the following: Task: vt\n"
        f"Query: {query}\n"
        "Expected answer format: comma separated variable names\n"
        "Use the Python REPL to extract the answer from context.\n\n"
        f"Your context is a str of {context_chars} total characters. "
        "Each sub-LLM call can handle roughly ~100k tokens at once."
    )
    turn = (
        "You have not interacted with the REPL environment or seen your prompt / context yet. "
        "Look at the context first; do not provide a final answer yet.\n\n"
        f"Turn {iteration}/2:"
    )
    return rwkv_instruction_prompt(RLM_SYSTEM_PROMPT, [metadata, turn])


def build_dataset(n: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        query = str(rng.randint(10000, 99999))
        context_chars = rng.randint(90_000, 130_000)
        rows.append({
            "id": f"repl_adherence_{i}",
            "prompt": make_problem(query, context_chars),
            "target": TARGET_REPL,
            "query": query,
        })
    return rows


def encode_example(tok, row: dict[str, str]) -> dict[str, Any]:
    prompt_ids = tok.encode(row["prompt"])
    target_ids = tok.encode(row["target"]) + [0]
    full = prompt_ids + target_ids
    return {
        "x": full[:-1],
        "y": full[1:],
        "mask": [0] * max(0, len(prompt_ids) - 1) + [1] * len(target_ids),
        "id": row["id"],
    }


def collate(encoded: list[dict[str, Any]], device: str, dtype: torch.dtype):
    max_len = max(len(x["x"]) for x in encoded)
    bsz = len(encoded)
    x = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    y = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    m = torch.zeros((bsz, max_len), dtype=torch.float32, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=dtype, device=device)
    for i, ex in enumerate(encoded):
        n = len(ex["x"])
        x[i, :n] = torch.tensor(ex["x"], dtype=torch.long, device=device)
        y[i, :n] = torch.tensor(ex["y"], dtype=torch.long, device=device)
        m[i, :n] = torch.tensor(ex["mask"], dtype=torch.float32, device=device)
        attention_mask[i, :n] = 1
    return x, y, m, attention_mask


def load_train_model_bf16_cpu_first(pth_path: str, ctx_len: int, grad_cp: int = 1):
    from rwkv7_trainable import RWKV7

    try:
        sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(pth_path, map_location="cpu")

    n_embd = sd["emb.weight"].shape[1]
    vocab_size = sd["emb.weight"].shape[0]
    n_layer = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    dim_ffn = sd.get("blocks.0.ffn.key.weight", torch.zeros(n_embd * 4, n_embd)).shape[0]
    args = SimpleNamespace(
        n_embd=n_embd,
        vocab_size=vocab_size,
        n_layer=n_layer,
        dim_att=n_embd,
        dim_ffn=dim_ffn,
        head_size_a=64,
        head_size_divisor=8,
        ctx_len=ctx_len,
        chunk_ctx=ctx_len,
        grad_cp=grad_cp,
        train_type="fullstate",
        peft="none",
        my_testing="x070",
    )
    model = RWKV7(args)
    model.load_state_dict(sd, strict=False)
    model.args = args
    model = model.to(torch.bfloat16)
    return model, args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_STEM))
    parser.add_argument("--out-dir", default="bench_outputs/state_tune_repl_adherence")
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=20)
    args = parser.parse_args()

    setup_paths_and_env(args.ctx_len)
    os.chdir(STATE_TUNING_DIR)

    from local_rlm import state_utils
    from utils import TRIE_TOKENIZER

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = build_dataset(args.train_size, args.seed)
    (out_dir / "train_data.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in train_rows),
        encoding="utf-8",
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = f"cuda:{args.cuda}"
    tok = TRIE_TOKENIZER(str(VOCAB_PATH))
    encoded = [encode_example(tok, row) for row in train_rows]

    base_name, pth_path = state_utils.normalize_model_arg(args.model)
    train_model, _ = load_train_model_bf16_cpu_first(pth_path, ctx_len=args.ctx_len, grad_cp=1)
    trainable = state_utils.freeze_except_time_state(train_model)
    train_model = train_model.to(device)
    named = [(n, p) for n, p in train_model.named_parameters() if p.requires_grad]
    opt = torch.optim.Adam([p for _, p in named], lr=args.lr, betas=(0.9, 0.95), eps=1e-18)
    forward_dtype = next(train_model.parameters()).dtype

    log_path = out_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"trainable={trainable} examples={len(encoded)}\n")

    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch = random.sample(encoded, k=min(args.batch_size, len(encoded)))
        x, y, m, attention_mask = collate(batch, device=device, dtype=forward_dtype)
        denom = torch.clamp(m.sum(), min=1.0)
        opt.zero_grad(set_to_none=True)
        logits = train_model(x, attention_mask=attention_mask)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        loss_tok = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape_as(m)
        loss = (loss_tok * m).sum() / denom
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0).item())
        opt.step()
        dt = time.time() - t0
        if step == 1 or step % 5 == 0 or step == args.steps:
            line = f"step={step} loss={float(loss.item()):.6f} grad={grad_norm:.6f} dt={dt:.2f}s"
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(line + "\n")
        if step % args.save_interval == 0 or step == args.steps:
            state_utils.save_time_state_only(train_model, str(out_dir / f"time_state_step{step}.pth"))
            state_utils.save_time_state_only(train_model, str(out_dir / "latest_time_state.pth"))

    print(f"saved {out_dir / 'latest_time_state.pth'}", flush=True)


if __name__ == "__main__":
    main()
