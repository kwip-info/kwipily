# Shared data guidance

## Check what the source actually read

Confirm a file’s separator, columns and sampled values before building downstream rules.

**Separator:** the character between fields. **Header:** the first row containing column names. **Sample:** a limited set shown for inspection; it does not automatically describe every source row.

1. Select the source. Review the chosen file and permission summary; a displayed filename is not a substitute for write/read access.
2. Choose the CSV separator and header options to match the file. Table Input supports an explicit delimiter and configurable header interpretation; inspect the resulting column names.
3. Refresh the preview after changing the file or interpretation. Compare columns, types and actual cells, and read whether the count is sampled, total, or unknown.
4. Apply deliberate changes and rerun. Previous observed metadata must become stale when source bytes or interpretation change.

A source preview is evidence of the rows actually sampled under the stated budget. A later failed run cannot establish a new successful complete source result.

## Choose, order and rename columns deliberately

Keep column choices and names consistent across the flow.

**Schema:** column names and value types. **Mapping:** matching an input column to the place that uses it. **Draft:** an edit not applied yet.

1. Select Columns. Use the column table to choose which columns remain, their order and any downstream names.
2. Choose a unique nonblank new name. Renaming a column changes its label, not its values or row identity.
3. Apply the whole edit; inspect the downstream schema. If a saved mapping still names the old column, repair it explicitly rather than assuming the app guessed a replacement.
4. Try a draft, collapse the inspector, then reopen it. Draft controls should be preserved until Apply or Revert; Undo should restore an applied edit.

Compare exact ordered names before and after the operation. A numerical cast changes value type; renaming alone does not turn text into numbers.

## Read metadata with its source and freshness

Tell declared column information from values observed during a particular run.

**Metadata:** information about the table, such as names, types and row counts. **Observed:** measured from a stage actually reached. **Stale:** measured before the current graph or source changed.

1. Select an input or output and inspect its column information. Read the originating node and port as well as the names.
2. Distinguish projected schema from observed metadata. A projected schema predicts structure; it does not prove that a run visited the stage.
3. After a later step fails or is cancelled, inspect earlier reached stages. Their bounded observations can remain available with a partial label. Unvisited stages have no newly observed result.
4. Change a source or graph control. Check the freshness label before reusing a column choice; a stale observation must not silently replace the current draft or mark the failed run complete.

A retained partial observation is useful evidence for repair. It is not a completed preparation archive, a successful run, or permission to evaluate a protected partition.
