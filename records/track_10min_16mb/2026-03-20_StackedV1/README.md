# Stacked V1: Int6 QAT + NorMuon WD + SWA + Overtone Init + MLP 3x + 10L + FA3

**Pending 8xH100 evaluation.**

Stacks 10 orthogonal improvements from top-scoring submissions into a single configuration.

## Techniques

1. **Int6 post-training quantization** (from PR #114 by saml212): Per-row int6 (64 levels) for weight matrices, ~25% smaller than int8, freeing artifact space for wider MLP.

2. **STE quantization-aware training**: Fake-quantize weights to int6 during forward pass with straight-through gradient estimator. Teaches the model to be robust to post-training quantization. (Proven by PR #128 and #122.)

3. **NorMuon optimizer**: Normalized Muon — orthogonalize gradients BEFORE momentum, not after. Better optimization dynamics. (From NorMuon paper, used in PR #122.)

4. **Decoupled Muon weight decay** (0.02): Manual `p.mul_(1 - wd * lr)` after Muon step. Improves generalization + compression. (From PR #60 by notapplica.)

5. **Stochastic weight averaging (SWA)**: Collect model snapshots every 200 steps during warmdown and average them. Better generalization than final point estimate. (From PR #122 and #89.)

6. **Overtone spectral embedding init**: SVD power-law spectrum shaping (`S_k ~ k^{-0.5}`). (From PR #60.)

7. **Phase-transition residual mixing**: Sigmoid-scheduled `resid_mix` initialization — early layers trust `x0`, late layers trust residual. (From PR #60.)

8. **FlashAttention 3**: ~10ms/step speedup = ~175 more training steps in 10-minute budget. (From PR #122.)

9. **zstd-22 compression**: Better compression ratio than zlib-9, freeing artifact space.

10. **AdamW** for token/scalar optimizers (weight_decay=0.01) instead of plain Adam.

## Configuration

```bash
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 MLP_HIDDEN=1536
NUM_LAYERS=9 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_WEIGHT_DECAY=0.02
QAT_ENABLED=1 QAT_BITS=6
SWA_ENABLED=1 SWA_EVERY=200
EVAL_SEQ_LEN=2048 EVAL_STRIDE=256
```

## Running

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_StackedV1/train_gpt.py
```

Override defaults via environment variables. The script is self-contained.

## Attribution

Built on techniques from: saml212 (PR #114, #61, #96), notapplica (PR #60),
mattqlf (PR #50), mtybadger (PR #122), vmfunc (PR #89), rsavitt (PR #128).
