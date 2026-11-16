# Make repeated spaces consistent

Collapse runs of spaces in notes into one space while keeping the original note for comparison.

**Replacement:** text put in place of a matching part. **First:** change only the first match. **All:** change every match in a cell. **Preserve:** keep a missing cell missing.

1. Open Make repeated spaces consistent. Select Formula, using Replace text and the note column.
2. Select Formula. Choose the Replace text guided editor. Replace runs of spaces in note with one space, replace All matches, and write a new clean_note text column.
3. Keep clean_note with Write set to New column, so note stays visible. Predict how paid followed by three spaces then today will change. Run Flow.
4. Inspect note and clean_note side by side. paid   today becomes paid today; waiting  review becomes waiting review. The order IDs and row count remain unchanged.
5. Notice A08 still has a leading and trailing space. Collapsing repeated spaces is different from trimming edges. Do not assume a tool performed a cleanup you did not configure.

## Inspect

The clean notes are paid today, waiting review, paid, paid yesterday, missing email, refunded, paid and " paid ". The final quoted spaces are intentionally retained.

## Limits

First/all and explicit column replacement are configurable. Replacement group references must exist. Result strings and whole-table work have enforced budgets; oversized or too-slow results are refused rather than partially accepted.

Editable guided practice, not a scored assessment. Opening or running this example does not award completion. Inspect actual output; sample results describe these eight synthetic orders only.
