# 11L XSA4 + EMA + Partial RoPE + LN Scale + Late QAT

**Pending 8xH100 evaluation.**

## Techniques

Built on the PR #180 stack (SmearGate, BigramHash(10240), int5 MLP/int6 attention,
OrthoInit, Muon WD=0.04, SWA/50) with 5 additional improvements:

1. **11 layers** (up from 10) — extra depth funded by int5 MLP compression savings.
2. **XSA (Exclusive Self-Attention) on last 4 layers** — removes self-value projection
   from attention output, forcing cross-token information flow. Zero new parameters.
3. **EMA (decay=0.997)** replacing SWA — exponential moving average every step gives
   smoother weights and better quantization than periodic checkpoint averaging.
4. **Partial RoPE (16 of 64 dims)** — apply rotary position embeddings to only 25% of
   head dimensions. Remaining dims use position-free attention. Zero new parameters.
5. **LN Scale (1/sqrt(layer+1))** — damp RMSNorm outputs in deeper layers, stabilizing
   training at 11 layers. Zero new parameters.
6. **Late QAT** — STE int6 fake-quantization enabled only in final ~4% of training
   (lr_scale < 0.1). Reduces int6 degradation with minimal throughput cost.

## Configuration

All defaults baked into train_gpt.py:

- 11 layers, 512 dim, 8 heads / 4 KV heads, MLP 3x (1536)
- SmearGate + BigramHash(10240, dim=128)
- seq_len=2048, batch=786K, warmdown=3000
- matrix_lr=0.025, scalar_lr=0.025, tied_embed_lr=0.035
- Muon momentum=0.99, WD=0.04
- EMA decay=0.997, XSA on last 4 layers
- Partial RoPE (16 dims), LN Scale enabled
- Late QAT threshold=0.1
- Int5 MLP / int6 attention + zstd-22
- Sliding eval stride=64

## Running

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_PartialRoPE_LateQAT/train_gpt.py
```

## Attribution

Built on techniques from: thwu1 (PR #180), raahilshah (PR #162), saml212 (PR #114/#61),
notapplica (PR #60). XSA, EMA, Partial RoPE, LN Scale, and Late QAT implementations
adapted from saml212 (PR #315).
