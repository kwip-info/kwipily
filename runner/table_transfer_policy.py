"""Bounded prepared-table transfer and display, independent of Working data sampling.

Swift TableTransferLimits mirrors these constants and is checked for drift. Limits
are admission boundaries, not permission to truncate a complete selected table.
"""
MAX_ARTIFACT_BYTES = 50_000_000
MAX_PAGE_BYTES = 1_000_000
MAX_PAGE_ROWS = 100
MAX_COLUMNS = 128
MAX_ROWS = 5_000_000
MAX_CELLS = 5_000_000
