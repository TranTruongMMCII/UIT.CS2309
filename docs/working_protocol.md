# Working Protocol

We proceed one step at a time.

Each step must include:

1. Goal.
2. Files changed.
3. Implementation details.
4. Expected output.
5. Confirmation before moving to the next step.

## Current sequence

1. Step 0/1: fork, branch, documentation, settings audit.
2. Step 2: reproducible Kaggle/dev environment.
3. Step 3: relation diagnostics without method changes.
4. Step 4: development baseline report.
5. Step 5: UPR-CRE v0.1 hard-score refinement.
6. Step 6: UPR-CRE v0.2 soft relation matrix.
7. Step 7: confidence curriculum.
8. Step 8: final ablation and paper tables.

## Constraints

- Do not claim the original paper lacks prototype or uncertainty.
- Do not modify backbone or add CLIP before the relation refinement module is validated.
- Do not use ground-truth cross-modal correspondence during training; it is allowed only for offline diagnostics.
- Do not compare a short RegDB run directly against the original paper's SYSU/LLCM tables.
