from pathlib import Path
import re
from runner import table_transfer_policy as policy
from runner import table_view, dataflow_table_reader


def test_native_worker_and_view_share_complete_table_and_page_admission():
    root=Path(__file__).resolve().parents[2]
    # Test lives under runner/tests after implementation.
    text=(root/'Sources/KwipMLCore/TableTransferLimits.swift').read_text()
    for native,python in [('artifactBytes','MAX_ARTIFACT_BYTES'),('pageBytes','MAX_PAGE_BYTES'),('pageRows','MAX_PAGE_ROWS'),('columns','MAX_COLUMNS'),('rows','MAX_ROWS'),('cells','MAX_CELLS')]:
        assert int(re.search(r'let '+native+r' = ([\d_]+)',text).group(1).replace('_',''))==getattr(policy,python)
    from runner import dataflow
    assert dataflow.MAX_TABLE_CELLS==policy.MAX_CELLS and dataflow.MAX_COLUMNS==policy.MAX_COLUMNS
    assert table_view.MAX_INDEX_BYTES==dataflow_table_reader.MAX_FILE_BYTES==policy.MAX_ARTIFACT_BYTES
