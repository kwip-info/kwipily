# Missing values, categorical matrices and leakage

For feature j, let O_j be the training rows where that feature is observed. Mean imputation uses μ_j = sum(X_ij for i in O_j) / |O_j|. Missing entries become μ_j; observed entries are unchanged. The denominator is the observed training count, not the whole dataset size. Median uses the midpoint of sorted observed values (the average of the two central values for even counts). Most-frequent chooses the modal observed value under the explicit tie rule. If O_j is empty, no statistic exists: the configured error/constant decision applies.

In matrix form, with missingness mask M (1 for missing), mean filling is X_filled = (1 − M) ⊙ X_observed + M ⊙ (1 μᵀ). This notation treats the missing positions in X_observed as zero for the arithmetic; the implementation preserves actual missing values until Preprocess applies its configured rule.

For learned vocabulary c₁…cₖ, one-hot H_ij = 1[x_i = c_j]. A known category contributes exactly one 1 in its block. An ignored unknown contributes only zeros, which is why the count must remain visible. No reference category is dropped in this implementation.

Assessment challenges:

1. Compute training-only means/medians by hand from retained IDs and compare fitted state within numerical tolerance. Move a held-out value to 10⁹; fitted statistics must remain unchanged.
2. Place a new category only in held-out data. It must never appear in learned vocabulary. Predict both configured outcomes: explicit error or all-zero indicators with a positive ignored-category diagnostic.
3. Create a modal tie and an entirely missing training column. Explain and test each explicit policy; do not infer a statistic from held-out observations to resolve either case.
4. Swap two learned preprocessing states while keeping the same column names. The fit-chain check must reject the mismatch; explain why names alone do not establish identical preparation.
5. Separate the effect of a fixed partition seed from the validity of the split. Record disjoint IDs, then discuss subject/time/duplicate-content leakage that row IDs alone cannot exclude.

These are practice prompts for your own investigation. This reading does not award completion or assess your explanation.

In Preparation examples, choose Missing values & categorical features, then Prepare data. Inspect the Preprocess imputation statistics, vocabulary and validation output. The bundled current example uses Columns → Partition → Preprocess, with 24 training, eight validation and eight test rows. Connect all three Partition outputs directly to their matching Preprocess inputs, then inspect its three prepared outputs. Its training vocabulary is blue, new-heldout and red; the historical category name new-heldout does not establish partition membership. To investigate an unknown category, edit a verified validation/test observation after recording its ID and confirm that it never enters the training vocabulary. The prepared table contains several numeric columns; the current classifier uses two features that you select explicitly.

## Reusing verified preparation

Materialization saves completed node outputs with their input/configuration identities. Reuse requires an explicitly pinned prior manifest, still rereads the selected CSV, and validates every requested artifact. Predict which nodes should run again when the source bytes, a transform setting or a fitted state changes. A configuration change invalidates descendants even when the new output happens to be equal. Learned states currently bind the entire graph, so fit/transform invalidation is deliberately conservative.

Choose Recompute & retain, then Prepare data. Change the execution policy to Reuse verified outputs and prepare again. Inspect the computed and reused labels. Next change an imputation statistic and predict the blocks that must recompute before running. A damaged or outdated retained result is rejected; choose Recompute & retain to create a new one. Cache speed does not establish data correctness, and preparation reuse is distinct from training checkpoint resume.
