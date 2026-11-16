# Synthetic Table Input option fixtures

These minimal fixtures were authored for Kwip ML's JSON, JSONL, Markdown, XML and HTML reader tests. They contain no customer data. Links use the reserved `.invalid` domain and are never fetched. Tests encode the text into UTF-8/BOM, UTF-16 and Windows-1252 variants in temporary directories, retaining one readable source of truth here.

Expected values are asserted independently in `test_table_options_markup_json.py`; fixtures do not compute expectations from a reader or native option descriptor. The suite covers public format options and meaningful negative controls. Resource-strategy options such as JSONL batch size and low memory preserve data rather than changing its meaning; those tests compare independently specified records, identities and schema across settings.
