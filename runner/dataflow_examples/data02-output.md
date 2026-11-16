# Save the complete selected table

Write selected columns from all eight orders into a new semicolon-separated file in a folder you explicitly allow.

**Delimited file:** a text table whose columns are separated by a chosen character. **Destination:** the folder you choose for the saved file. **Overwrite:** replace a file already there. **Preview:** a small on-screen sample, not the whole exported table.

1. Open Save the complete selected table. Inspect the source and Columns output before saving anything.
2. Select Table Output. Choose an allowed destination folder, filename prepared-orders.csv, semicolon separator, and columns order_id, quantity, unit_price in that order.
3. Leave overwrite off. Run Flow. An explicit destination grant is required; opening the example alone does not save a file.
4. Inspect the saved-output result: full row count, column order, filename and status. Open the written file and confirm its header and eight data rows.
5. Run again without changing the filename. The existing file should be preserved and the run should explain the refusal. Choose another filename, or consciously enable overwrite. Cancellation before preparation publication preserves existing files; later training failure may leave a completed preparation write.

## Inspect

The file has three columns and eight complete data rows, plus its header. Its first data row is A01;2;12.5. Semicolons separate fields. The source contains eight rows here; for a larger source, a small preview does not limit the full selected output.

## Limits

Destination grants remain local and private. Complete table writes support CSV/TSV, typed Parquet, JSON/NDJSON and Excel .xlsx. Stop protects existing files; replacement is atomic and Excel replaces the whole workbook. Images require a new copied-assets bundle and verified relative references. Portable replay requires explicit destination bindings. History viewing never writes again; a successful preparation write may remain after later training failure.

Editable guided practice, not a scored assessment. Opening or running this example does not award completion. Inspect actual output; sample results describe these eight synthetic orders only.
