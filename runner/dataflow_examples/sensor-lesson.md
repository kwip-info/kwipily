# Prepare sensor readings

This synthetic exercise demonstrates selection, renaming and typed row filters. It is not a real sensor study and does not establish model quality. Reading or running it does not award mastery or a completion grade.

## Predict the flow

The file contains 48 rows with sample_id, reading_raw, temperature, quality, label and notes. Columns keeps an ordered subset, drops notes, renames reading_raw to amount, and casts amount/temperature to float64 and label to int64. Quality filter keeps quality equal to the exact string "accepted". Range filter keeps amount greater than or equal to the numeric literal 0. The final Columns node retains amount, temperature and label in that order.

Predict the count at every stage before running. Inspect the actual previews: 48 source rows → 40 accepted-quality rows → 36 nonnegative accepted readings. A seed-42 stratified Partition with 20% validation and 20% test produces 22 training, 8 validation and 6 test rows, with both labels represented. Per-class rounding determines these counts. Connect each Partition output directly to the matching Preprocess input. Preprocess learns standard scaling only from training and applies the same state to validation and test. Model mapping is features [amount, temperature], target label; the existing classifier still supports exactly two numeric features and binary integer labels.

## Columns and values are different choices

Selection order controls output column order. Rename changes a label, never a value or observation identity. Names are exact: whitespace-bearing names remain distinct. Column rules must reference selected columns and cannot create a duplicate final name. Simultaneous swaps are supported. Verify the source row IDs still identify the same observations after selection, filtering and transformation.

All six filter comparisons exist: equal, not equal, less than, less/equal, greater than, greater/equal. A string column requires a string literal; a boolean column requires a boolean; a numeric column accepts an integer or floating-point literal, never a boolean. Strings compare lexicographically: "10" is less than "2". Choose a numeric target type in Columns first when you mean a numeric comparison. Boolean ordering is False before True; use equality when that best states your intent. The current Filter also supports expressions, patterns and multiple conditions. This example uses one simple condition per node; connect the matching output to the next stage so both conditions must be met. Nonmatching outputs retain rejected observations for inspection.

With this example’s explicit error null policy, a missing compared value fails, even for not-equal. Choose an explicit missing-value policy before filtering; this example has no missing values. A filter that removes every row fails rather than publishing an empty training source. Failures produce no completed preparation manifest.

## Learn the right population

These filters express the intended population before splitting. They are predetermined quality/range rules, not thresholds learned from held-out performance. Do not repeatedly adjust them to improve a validation score and still call that score independent. Partition before fitting any learned statistic. For each training feature x, mean = sum(x)/n and population scale = sqrt(sum((x−mean)^2)/n); transformed values are (x−mean)/scale. Only training values enter mean and scale.

Filtering after fitting retains the original fit lineage; it does not refit statistics. Renaming or selecting after fitting also does not rewrite a learned state's expected schema. The engine may reject an incompatible later transform or prepared-training selection. Prefer the demonstrated selection/filter → split → training-fit order.

## Check your reasoning

1. Why does numeric amount >= 0 remove four accepted readings? Identify those original observation IDs.
2. Predict how a string comparison would differ, then test on a copy with an explicit string literal. Do not treat a successful run as proof that the chosen population is appropriate.
3. Swap the selected output order. Did row membership change? Did column order change?
4. Rename sample_id (later dropped by final selection). Final model values can stay equal, but the changed node and descendants must recompute under conservative cache invalidation.
5. Materialize, then change the range threshold. Verify the unchanged prefix reuses and changed descendants recompute. Inspect reported identities, not just elapsed time.
6. Compute the training means and scales independently. Explain why a held-out outlier cannot change them.

Use the standalone export to replay preparation with an explicit local source binding. Source paths and file grants are not embedded in the portable graph. This lesson supports exploration; it does not provide a machine-graded assessment or infer broader competence.
