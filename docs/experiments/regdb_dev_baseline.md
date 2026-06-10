# RegDB Development Baseline

This is a development baseline for fast iteration on Kaggle T4. It is not a full reproduction of the ICCV paper.

## Local dev baseline from initial Kaggle run

| Field | Value |
|---|---|
| Dataset | RegDB |
| Trial | 1 |
| Seed | 1 |
| Stage 1 epochs | 5 |
| Stage 2 epochs | 15 |
| Milestones | 8, 12 |
| LR | 0.00045 |
| Batch pid num | 5 |
| PID num sample | 4 |
| Test batch | 64 |
| Best Rank-1 | 62.62% |
| Best mAP | 58.43% |
| Best mINP | 45.97% |
| Best phase2 epoch | 10 |

## Usage

Use this only as a development reference. UPR-CRE experiments on RegDB should use the same schedule when checking whether a change is promising.

## Later final comparison

For a paper claim, run a fair local WSL baseline and UPR-CRE under the same full or agreed schedule. If time allows, validate on SYSU-MM01 or LLCM.
