## Representative result summary

The final benchmark uses synthetic VT tasks where the long text is available as the REPL variable `doc_text`. The root model must write Python code to compute the answer.

```text
128K, 20 samples:
zero-state    0/20
state-tuned  20/20
```

## Example commands

Train a 1.5B VT-closure state:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python local_rlm/state_tune_repl_adherence.py \
  --model rwkv7-g1g-1.5b-20260526-ctx8192 \
  --steps 80 --batch-size 2 --train-size 128 --lr 1e-4 \
  --save-interval 20 \
  --out-dir bench_outputs/state_tune_repl_vt_closure_2p9_80step \
  --cuda 0
```

Run zero-state RLM on 64K VT:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python local_rlm/run_ruler_synthetic_bench.py \
  --lengths 65536 --tasks vt --samples-per-task 20 --sample-start 0 \
  --modes rlm --max-tokens 1024 --parallelism 10 --worker-devices '' \
  --model-name rwkv7-g1g-1.5b-20260526-ctx8192 \
  --model-path rwkv7-g1g-1.5b-20260526-ctx8192 \
  --engine-module rwkv7_fp16 \
  --output bench_outputs/ruler_vt_64k_2p9_vtclosure_zero20.json \
  --raw-output bench_outputs/ruler_vt_64k_2p9_vtclosure_zero20_generations.jsonl
```

Run tuned-state RLM on 64K VT:

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python local_rlm/run_ruler_synthetic_bench.py \
  --lengths 65536 --tasks vt --samples-per-task 20 --sample-start 0 \
  --modes rlm_tuned --max-tokens 1024 --parallelism 10 --worker-devices '' \
  --model-name rwkv7-g1g-1.5b-20260526-ctx8192 \
  --model-path rwkv7-g1g-1.5b-20260526-ctx8192 \
  --engine-module rwkv7_fp16 \
  --state-path states/vt_closure_2p9_latest_time_state.pth \
  --output bench_outputs/ruler_vt_64k_2p9_vtclosure_tuned20.json \
  --raw-output bench_outputs/ruler_vt_64k_2p9_vtclosure_tuned20_generations.jsonl
```


