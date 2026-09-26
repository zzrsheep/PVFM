# PVFM architecture and checkpoints

`pvfm/model.py` implements **one graph**, shared by three capacity settings.
There are no ablation, alternative-fusion, RevIN or head-selection switches.

```text
Historical PV + six weather variables       Six future weather variables
            │                                          │
       Patch tokenizer                            Patch tokenizer
            │                                          │
   L factorized encoder blocks                         │
            │                                          │
       PV channel only                                 │
            │                                          │
   History-to-future projection                         │
            └───────────── L decoder blocks ────────────┘
                                  │
                       Future PV refinement block
                                  │
                     LayerNorm + overlapping Q11 head
                                  │
                  Ordered quantiles (q50 point forecast)
```

## Tokens and conditioning

Each tokenizer uses a linear value-patch projection, a variable embedding,
a **learned absolute patch-position embedding**, and a solar projection.
The sum is masked, normalized with LayerNorm, and passed through dropout.
Patch length is 12 and stride is 6 at both resolutions. Historical and
future solar projections have independent parameters. RoPE in temporal
attention is additional to the absolute patch-position embedding.

For each local timestamp, the four solar features are
`[a, 1-a, sin(hour_angle), cos(hour_angle)]`. Minute information is included.
Solar hour is approximated by `local_hour - UTC_offset + longitude/15`;
declination uses the Cooper approximation, and elevation is clipped at zero.
This is a solar-geometry proxy, not a precise astronomical ephemeris.
Calendar channels decode the timestamp used in these calculations; **there
is no separate learned calendar projection** in the published graph.

The site/window adapter receives **eight** features:

- Four window descriptors: context steps, forecast steps, label length,
  and forecast duration in hours (the temporal loader sets label length = H).
- Four geographical descriptors: latitude/90, sin(longitude in radians),
  cos(longitude in radians), and UTC offset/14. Normalized latitude and
  UTC offset are clipped to [-1, 1].

These are concatenated and mapped through Linear–GELU–Linear, then added
to both historical and future tokens **after tokenizer normalization and
dropout**. Calling this an MLP of geographical features alone is incomplete.
Capacity is not part of this adapter; targets already use capacity factors.

## Encoder and decoder

Each encoder block applies temporal attention independently per variable,
then cross-variable attention independently per patch, then an FFN residual.
After L blocks, the PV channel is retained. Although the other channels are
not passed separately to the projection, they have already interacted with
PV through variable attention.

The historical PV patch axis is interpolated to the configured projection
width, linearly mapped to the future reference patch axis, and interpolated
to the actual horizon where necessary.

Each decoder block applies separate temporal self-attention to PV and NWP,
then lets each PV patch attend to the weather variables at that future patch.
**Both PV and NWP receive FFN residual updates**, using the block's shared
FFN and normalization weights. NWP tokens are carried to the next layer.

After the L decoder blocks, **one additional PV-only refinement block**
applies temporal self-attention and FFN, each with pre-normalization and a
residual connection. It introduces no additional data input. It is not the
historical PV-channel extraction operation.

The final head predicts patch-wise Q11 values, averages overlaps, and applies
a noncrossing parameterization: a free median plus positive softplus
increments with scale 0.1. Levels are 0.05, 0.1, ..., 0.9, 0.95. There is no
additional output clipping or ReLU. Training uses Q11 pinball loss; evaluation
selects common-Q9, as specified in [METRICS.md](METRICS.md).

## Capacity settings

| Name | Active parameters | Width | Heads | Encoder / decoder | FFN |
|---|---:|---:|---:|---:|---:|
| PVFM-3.9M | 3,879,252 | 128 | 8 | 6 / 6 | 512 |
| PVFM-15.3M | 15,324,756 | 256 | 4 | 6 / 6 | 1,024 |
| PVFM-45.0M | 44,991,828 | 384 | 6 | 8 / 8 | 1,536 |

All include the additional final refinement block. Only PVFM-3.9M trained
weights are bundled. The other JSON configurations specify capacity settings,
not trained checkpoints.

The research exports counted inactive calendar projections and an unused
scalar point head: 3,882,336 / 15,330,912 / 45,001,056 parameters. Those unused
tensors are not registered in the clean model. **All active tensor names and
values are unchanged**. This removes no operation from the original Q11
prediction path. `verification/cleanup_parity.json` records exact CPU prediction
and training-loss/gradient comparisons against the pre-cleanup exports.

## Checkpoint scope

- `checkpoints/pvfm_3_9m_80k_q11.pt`: all-region pretraining, with the target
  Australian stations held out, not the entire region.
- `checkpoints/pvfm_3_9m_australia_holdout_80k_q11.pt`: excludes all of Australia
  from pretraining. Select `--model-profile australia_holdout` for the regional
  generalization evaluation.

Both are inference exports at 80k steps, without optimizer or RNG state.
Checkpoint loading is strict. No research-checkpoint migration is performed
silently. The clean training entry point uses the same objective, temporal
sampling policy and LR schedule but is not an exact replay of the original
distributed run or its from-scratch initialization sequence.
