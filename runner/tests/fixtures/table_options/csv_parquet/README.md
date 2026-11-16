# Synthetic CSV and Parquet option fixtures

These tiny files contain invented values only. The owning executable suite is
`runner/tests/test_table_options_csv_parquet.py`; its `OPTION_CASES` maps each
exposed native/compatibility CSV and native/PyArrow Parquet option to concrete tests.

CSV files deliberately cover BOM, invalid UTF-8, Windows-1252, UTF-16, quoting,
quoted/unquoted empty fields, missing/extra fields, explicit escapes and headers.
Additional delimiter/terminator/skip fixtures are written in each test's private
temporary directory with independently stated rows.

Parquet files contain:

- `basic.parquet`: integer `a` values 1,2,3 and text `b` values x,y,z; one row per row group.
- `widths.parquet`: Int8 `a` values 1,2 and Float32 `b` values 1.5,2.5.
- `float64.parquet`: Float64 `b` values 16777217 and 1.5. Float32 narrowing rounds the first to 16777216.
- `scalar_types.parquet`: one column for each supported scalar Polars spelling; text seven, Boolean true, integers 7 and floats 1.5.
- `category.parquet`: categorical red,blue, requiring an explicit categorical-to-string cast for scalar Table Input.
- `int96.parquet`: integer `a` values 1,2 plus two INT96 timestamps. Tests project `a` into Table Input and independently inspect the configured timestamp unit on the real PyArrow reader.
- `checksum.parquet`: plain Int64 values 1,2,3 with page CRC checksums.
- `checksum_corrupt.parquet`: the second encoded Int64 was changed from 2 to 20 without updating its checksum. Verification disabled returns 1,20,3; verification enabled rejects the bad page.

The checksum corruption is intentional and local to this test fixture. No customer
or production artifact was altered.

Resource/strategy tests combine actual parsing with exact argument dispatch.
`DISPATCH_ONLY_OPTIONS` explicitly identifies claims that do not establish faster
execution, lower RSS, specific thread scheduling, cache behavior or pandas-index
reconstruction. Metadata byte limits also receive real decoder-refusal checks;
page checksum and INT96 unit options have direct observable effects.
