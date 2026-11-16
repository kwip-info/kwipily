# File Input document fixtures

`generate.py` produces synthetic PDF, encrypted PDF, DOCX and ODT bytes in memory.
Text and RTF are independently authored byte strings in the dedicated test module.
No customer data, external downloads, macros or credential-bearing files are used.

`runner/tests/test_file_input_documents.py` exercises every exposed document reader
option through real extraction. `OPTION_CASES` maps 29 reader-qualified option IDs
to test functions. PDF layout fixtures independently specify exact expected text
geometry; rotated text and malformed headers distinguish orientation, rotated
layout and strictness policies. The one operational option is PDF `space_width`:
its real parser path and bounded numeric admission are checked, but the fixture's
font metrics do not demonstrate a change from this fallback value. No measured
performance claim follows from that operational case.

Text BOM detection considers UTF-32 before UTF-16 and otherwise UTF-8 with optional
BOM. No statistical encoding guess is made. LF/CRLF normalization is explicit;
line iteration avoids a materialized list. File output for documents joins units
using the explicit separator (default two line feeds). PDF page selection is
1-based, preserves requested order and rejects duplicates/out-of-range pages.
DOCX traverses paragraphs, tables and nested cells in order, deduplicates merged
cell aliases and optionally includes headers/footers. ODT preserves explicit
spaces, tabs and line breaks in paragraphs/headings/list content; repeated table
coordinates refuse until supported with a bounded expansion policy.

The readers accept owned bytes only. Capture is limited to 64 MiB; ZIP admission
limits expanded declared members to 128 MiB, member count to 10,000, each member
to 64 MiB, forbids encrypted ZIP members, duplicate/unsafe member names and XML
DTD/entity declarations. Emitted UTF-8 text is limited to 64 MiB/100,000 units;
PDF additionally limits page count to 10,000 and observed expanded page content
to 128 MiB. Cancellation is checked throughout traversal and unit production.

These limits do not turn third-party eager parsers into streaming readers.
PDF stream decompression and DOCX/RTF parsing can allocate before the application
observes decoded sizes; the isolated worker's hard memory/time budget remains
necessary. PDF performs text extraction only, with no OCR, rasterization,
password handling or visual-reading-order guarantee. No live external
relationship target or macro is executed. Native acceptance is separate.

Approved development dependency additions: Pillow 12.3.0 (MIT-CMU plus bundled
codec notices), pypdf 6.20.1 (BSD-3-Clause), python-docx 1.2.0 (MIT; existing lxml
and typing-extensions closure), striprtf 0.0.33 (BSD-3-Clause). ODT reuses
existing defusedxml; no odfpy is installed. Exact pins reside in
requirements-runner.txt. The Pillow wheel has native extensions and codec dylibs
requiring exact signed-payload/linkage/notice verification before a future
release; this fixture work does not constitute redistribution clearance.
