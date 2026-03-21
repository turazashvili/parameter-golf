# Stacked V1: Int6 + Muon WD + SWA + Overtone Init + MLP 3x + FA3

**Pending 8xH100 evaluation.**

Stacks multiple zero-overhead improvements into a single configuration, prioritizing
step throughput to maximize training steps within the 10-minute budget.

## Techniques

1. **Int6 post-training quantization**: Per-row int6 (64 levels) for weight matrices, ~25% smaller than int8, freeing artifact space for wider MLP.

2. **Decoupled Muon weight decay** (0.02): Manual `p.mul_(1 - wd * lr)` after Muon step. Improves generalization + compression.

3. **Stochastic weight averaging (SWA)**: Collect model snapshots every 200 steps during warmdown and average them. Better generalization than final point estimate.

4. **Overtone spectral embedding init**: SVD power-law spectrum shaping (`S_k ~ k^{-0.5}`).

5. **Phase-transition residual mixing**: Sigmoid-scheduled `resid_mix` initialization — early layers trust `x0`, late layers trust residual.

6. **FlashAttention 3**: Faster step times = more training steps in 10-minute budget.

7. **zstd-22 compression**: Better compression ratio than zlib-9, freeing artifact space.

8. **AdamW** for token/scalar optimizers (weight_decay=0.01) instead of plain Adam.

## Configuration

All defaults are baked into the script. No environment variable overrides needed.

```bash
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 MLP_HIDDEN=1536
NUM_LAYERS=9 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WEIGHT_DECAY=0.02
SWA_ENABLED=1 SWA_EVERY=200
EVAL_SEQ_LEN=2048 EVAL_STRIDE=256
```

## Running

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_StackedV1/train_gpt.py
```

The script is self-contained. Override defaults via environment variables if needed.
