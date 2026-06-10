# Research Positioning: UPR-CRE

## Base paper

Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning, ICCV 2025.

## What the original paper already has

We must not claim that the original method ignores prototype or uncertainty. The original method already includes:

1. Heterogeneous Expert Learning (HEL).
2. Cross-modal Relationship Establishment (CRE).
3. Collaborative Consistency Learning (CCL).
4. Prototype usage in collaborative learning / CLAE.
5. Entropy-based weighting for expert homogeneity / consistency.
6. Relaxed or weak constraints for contradictory matches.

## Gap we target

The original CRE mainly establishes cross-modal identity correspondences from expert predictions and represents them through hard/binary relation matrices such as consistent, specific/unique, and contradictory/remain relations.

The proposed gap is therefore not "adding prototype" or "adding uncertainty" in a generic sense. The gap is:

> Move prototype-level similarity and uncertainty-aware confidence directly into the cross-modal relation establishment/refinement step, before the relation is consumed by collaborative consistency learning.

## Proposed direction

UPR-CRE: Uncertainty-aware Prototype-guided Relation Refinement for weakly supervised visible-infrared Re-ID.

The module should refine relation construction by combining:

- expert prediction score;
- prototype-level visible-infrared similarity;
- uncertainty/confidence score from entropy, top-1/top-2 margin, expert disagreement, or relation stability;
- optional confidence curriculum from high-confidence relations to uncertain soft relations.

## Safe novelty claim

We do not claim to introduce prototypes or uncertainty to VI-ReID for the first time. We claim to use them directly for CRE/relation refinement in weakly supervised VI-ReID.

## Working hypotheses

1. A relation matrix refined by prototype similarity is more reliable than expert-only hard matching.
2. Uncertainty-aware weighting reduces the effect of noisy pseudo correspondences.
3. A confidence curriculum can improve stability by learning first from reliable relations and gradually using uncertain relations.

## Development dataset policy

RegDB is used for fast development and debugging. It should not be presented as the main benchmark of the original ICCV paper. Final claims should use a fair local comparison on the same dataset/schedule, and preferably SYSU-MM01 or LLCM if resources allow.

## Stop rule for method development

Before implementing the method, we first add relation diagnostics. We only move to UPR-CRE after we can log relation quality metrics such as common/specific/remain counts, mutual match ratio, entropy, margin, and offline pseudo-relation accuracy.
