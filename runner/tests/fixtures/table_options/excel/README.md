# Workbook option fixtures

`generate.py` creates synthetic workbooks in pytest temporary storage. No customer
data, credentials, macros or fetched resources are included. The hyperlink fixture
contains only a stored `example.invalid` reference. Assertions in
`runner/tests/test_table_options_excel.py` specify expected cells independently.

The test module exports `OPTION_CASES`: 75 qualified active configuration fields
across calamine, openpyxl and xlsx2csv, including common options per engine and
conditional nested settings. Each entry names actual local test functions for the
central inventory guard. The matrix currently has 89 passing cases, including
multiple selections, optional values, active-engine restrictions, explicit
refusals and per-call isolation. This is automated reader evidence; native panel
acceptance remains separate.

`OPTION_CLASSIFICATION` distinguishes 71 effect/refusal fields from four
operational-invariance fields: openpyxl `read_only`, and converted CSV `encoding`,
`low_memory` and converter `quoting`. Those operational cases check stable real
output or explicit inability to represent content; they do not assert a measured
memory improvement or byte-level converter output. Both supported UTF-8 modes
receive well-formed Unicode from the internal converter. Quoting choices are
tested through the converted parser, including unescaped-delimiter refusal.

The sparse calamine default case distinguishes coercion from strict refusal:
without an explicit dtype the mixed column becomes text and preserves every
value. An explicit numeric override instead produces null for incompatible text.
This fixture exposed the Python strict default diverging from the visible Core
coerce default; the default was aligned after recording the failure.

Reader repairs are local to workbook helpers. They preserve named header-only
schemas, scalar visible rich text, invalid-date conversion policy, hidden-sheet
exclusion, and merged cells whose XML omits physical target cells or rows.
Overlapping merge ranges refuse. No global third-party class is patched; the
xlsx2csv facade uses the pinned trusted converter's existing code with a private
per-call Sheet factory. Dependency upgrades must rerun these behavior cases.

All helpers operate on previously admitted owned bytes. Existing archive limits
and DTD/entity refusal run before parsing. openpyxl additionally bounds the
selected worksheet grid to five million cells. Merged ranges require positive
coordinates, at most 128 columns, and an aggregate five-million-cell budget;
merge construction and synthesized rows check cancellation. These bounds are
conservative and do not promise streaming for eager workbook formats.
