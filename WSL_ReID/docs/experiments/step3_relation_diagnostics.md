# Step 3: CRE Relation Diagnostics

## Goal

Measure the quality of the original CRE pseudo cross-modal relations before modifying the algorithm.

## Scope

This step is diagnostic-only.

It does not change:

- backbone,
- CRE pair selection,
- loss functions,
- training objectives,
- optimizer or scheduler.

## Files

New files:

- `WSL_ReID/relation_metrics.py`
- `WSL_ReID/scripts/run_regdb_relation_diagnostics.sh`
- `WSL_ReID/scripts/collect_relation_stats.py`
- `docs/experiments/step3_relation_diagnostics.md`

Modified files:

- `WSL_ReID/main.py`
- `WSL_ReID/wsl.py`
- `WSL_ReID/task/train.py`

## Metrics

Per epoch, diagnostics should save:

- number of `r2i` and `i2r` pairs,
- number of `common`, `specific`, and `remain` relations,
- mutual match ratio,
- entropy and top-1/top-2 margin of expert score matrices,
- offline pseudo-relation accuracy using hidden ground-truth correspondence,
- prototype similarity for relation groups.

Ground-truth correspondence is used only for offline diagnostics. It must not be used by training.

## Command

```bash
cd WSL_ReID
bash scripts/run_regdb_relation_diagnostics.sh
```

## Expected output

```text
../saved_regdb_resnet/relation_diag_regdb_smoke_1/relation_stats/epoch_000.json
../saved_regdb_resnet/relation_diag_regdb_smoke_1/relation_stats/epoch_001.json
../saved_regdb_resnet/relation_diag_regdb_smoke_1/relation_stats/relation_stats_summary.csv
```

## Next step

Use the diagnostic metrics to decide the first UPR-CRE variant.
