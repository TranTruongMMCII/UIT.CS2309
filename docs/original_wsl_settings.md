# Original WSL-VIReID Settings Audit

This file records settings from the original paper and official code. Verify these against the current repository before running final experiments.

## Sources to audit

1. ICCV 2025 paper: Section 4.2 Implementation Details.
2. Official repository README and run scripts.
3. Code defaults in `WSL_ReID/main.py`.
4. Optimizer and learning-rate groups in `WSL_ReID/models/__init__.py` and `WSL_ReID/models/optim.py`.
5. Dataset protocol in `WSL_ReID/datasets/*.py`.

## Paper-level settings

- Framework: PyTorch.
- Hardware reported by the paper: single RTX 4090 GPU.
- Encoder: ResNet-50.
- Input size: 288 x 144.
- Phase 2 collaborative consistency learning: 120 epochs.
- Encoder learning rate: 3e-4.
- Experts and shared classifier learning rate: 6e-4.
- Warmup: first 10 epochs.
- LR milestones: 30 and 70 for SYSU/LLCM setting.
- Prototype momentum: 0.8.
- Loss weights lambda1/lambda2: 0.25 / 0.25.
- Metrics: CMC, mAP, mINP.

## Code/script settings to verify

| Dataset | Stage 1 | Stage 2 | LR | Milestones | Notes |
|---|---:|---:|---:|---|---|
| SYSU-MM01 | 20 | 120 | 3e-4 | 30, 70 | Main paper dataset |
| LLCM | 80 | 120 | 3e-4 | 30, 70 | Main paper dataset |
| RegDB | 50 | 120 | 4.5e-4 | 50, 70 | Development/debug dataset in our project |

## Main defaults to verify in `main.py`

- `img_h = 288`
- `img_w = 144`
- `batch_pidnum = 8`, with RegDB commonly using 5
- `pid_numsample = 4`
- `test_batch = 128`
- `sigma = 0.8`
- `temperature = 3`
- `weak_weight = 0.25`
- `tri_weight = 0.25`
- `relabel = 1`
- `seed = 1`

## Fair comparison policy

For development, compare our method against a local WSL baseline under the same dataset, seed, schedule, and hardware. Paper-reported SYSU/LLCM results can be used in literature tables, but not as the only evidence for a local improvement claim.
