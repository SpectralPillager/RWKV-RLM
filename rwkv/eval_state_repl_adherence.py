from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
STATE_TUNING_DIR = ROOT / "third_party" / "state_tuning_min"
REFERENCE_DIR = ROOT / "third_party" / "reference"
MODEL_STEM = Path("/data_temp/mnt/raid5/zjx/rwkv/full-training/rwkv7-g1e-7.2b-20260301-ctx8192")
VOCAB_PATH = REFERENCE_DIR / "rwkv_vocab_v20230424.txt"
RAPID_SAMPLING_DIR = ROOT / "third_party" / "Rapid-Sampling-main"


def setup(ctx_len: int) -> None:
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


def load_helper():
    from local_rlm.state_tune_repl_adherence import make_problem

    return make_problem


def workspace_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p


def safe_decode(tok, ids: list[int]) -> str:
    try:
        return tok.decode(ids, utf8_errors="replace")
    except TypeError:
        return tok.decode(ids)


def score_adherence(text: str) -> dict[str, Any]:
    blocks = re.findall(r"```(?:repl|python)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    bad_markers = ["Input:", "User:", "Instruction:", "Repl Output", "REPL context", "Iteration"]
    has_ready = 'answer["ready"] = True' in text or "answer['ready'] = True" in text
    has_content = 'answer["content"]' in text or "answer['content']" in text
    has_import = "import re" in text
    has_doc = "doc_text" in text
    has_bad_marker = any(x in text for x in bad_markers)
    ok = len(blocks) == 1 and has_ready and has_content and has_import and has_doc and not has_bad_marker
    return {
        "ok": ok,
        "code_blocks": len(blocks),
        "has_ready": has_ready,
        "has_content": has_content,
        "has_import": has_import,
        "has_doc_text": has_doc,
        "has_bad_marker": has_bad_marker,
    }


@torch.no_grad()
def generate_one(
    infer_model,
    train_model,
    tok,
    sample_kernel,
    prompt: str,
    *,
    state_mode: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, int, int]:
    ids = tok.encode(prompt)
    B = 1
    if state_mode == "tuned":
        from local_rlm.state_utils import init_runtime_state_with_time_state

        state = init_runtime_state_with_time_state(infer_model, train_model, B, device)
    else:
        state = infer_model.generate_zero_state(B)

    logits = infer_model.forward_batch([ids], state)
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    generated: list[int] = []
    rand_states = sample_kernel.setup_rand(int(time.time_ns() % (2**31 - 1)), 1)
    vocab_padded = ((infer_model.args.vocab_size + 3) // 4) * 4
    penalties = torch.zeros((1, vocab_padded), dtype=torch.float32, device=device)
    for _ in range(max_new_tokens):
        x = logits.float()
        if x.size(-1) % 4 != 0:
            x = F.pad(x, (0, 4 - (x.size(-1) % 4)), value=-1e30)
        next_id = int(sample_kernel.batch_sampling_repetition_temperature_topk_topp(
            x,
            penalties,
            rand_states,
            1.0,
            0.1,
            0.996,
            float(temperature),
            int(top_k),
            float(top_p),
        )[0].item())
        generated.append(next_id)
        if next_id == 0:
            break
        window = safe_decode(tok, generated[-64:])
        if "\nUser:" in window or "\nInstruction:" in window:
            break
        logits = infer_model.forward_batch([[next_id]], state)
        if logits.dim() == 3:
            logits = logits[:, -1, :]
    return safe_decode(tok, generated), len(ids), len(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_STEM))
    parser.add_argument("--state", default="")
    parser.add_argument("--out", default="bench_outputs/state_tune_repl_adherence_eval.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    setup(args.ctx_len)
    os.chdir(STATE_TUNING_DIR)

    from local_rlm import state_utils
    from torch.utils.cpp_extension import load
    from utils import TRIE_TOKENIZER

    make_problem = load_helper()
    device = f"cuda:{args.cuda}"
    tok = TRIE_TOKENIZER(str(VOCAB_PATH))
    base_name, pth_path = state_utils.normalize_model_arg(args.model)
    from local_rlm.state_tune_repl_adherence import load_train_model_bf16_cpu_first

    train_model, _ = load_train_model_bf16_cpu_first(pth_path, ctx_len=args.ctx_len, grad_cp=1)
    state_utils.freeze_except_time_state(train_model)
    state_path = workspace_path(args.state) if args.state else None
    if state_path:
        loaded = state_utils.load_time_state_only(train_model, str(state_path))
        print(f"loaded_state={loaded} {state_path}", flush=True)
        if not loaded:
            raise FileNotFoundError(f"failed to load time_state checkpoint: {state_path}")
    from rwkv7_fp16 import RWKV_x070
    import types

    infer_args = types.SimpleNamespace(vocab_size=65536, MODEL_NAME=base_name)
    infer_model = RWKV_x070(infer_args)
    sample_kernel = load(
        name="rapid_sampling_rwkv",
        sources=[str(RAPID_SAMPLING_DIR / "sampling.cpp"), str(RAPID_SAMPLING_DIR / "sampling.cu")],
        extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"],
        verbose=False,
    )

    rows = []
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            query = str(20000 + i * 137)
            prompt = make_problem(query, 100_000 + i * 37)
            for state_mode in ["zero", "tuned"] if args.state else ["zero"]:
                t0 = time.time()
                text, in_tok, out_tok = generate_one(
                    infer_model,
                    train_model,
                    tok,
                    sample_kernel,
                    prompt,
                    state_mode=state_mode,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                row = {
                    "id": f"eval_{i}",
                    "state_mode": state_mode,
                    "elapsed_sec": time.time() - t0,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "response": text,
                    **score_adherence(text),
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps({k: row[k] for k in ["id", "state_mode", "ok", "code_blocks", "has_bad_marker", "output_tokens"]}, ensure_ascii=False), flush=True)

    for mode in sorted({r["state_mode"] for r in rows}):
        xs = [r for r in rows if r["state_mode"] == mode]
        print(f"{mode}: ok={sum(r['ok'] for r in xs)}/{len(xs)}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
