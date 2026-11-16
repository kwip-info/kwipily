"""Independent expectations for the packaged, deliberately unscored File Input practice."""
from pathlib import Path

from runner.file_input_options import defaults
from runner.file_input_source import preview_files, source_files
from runner.image_artifacts import artifact_scope, decode_pixels

COLLECTION = Path(__file__).resolve().parents[2] / 'Sources/KwipMLApp/Resources/FileInputPractice'


def read(options):
    published = {}
    with artifact_scope(lambda content, reference: published.setdefault(reference['sha256'], content)):
        table = source_files(COLLECTION, options)
    try:
        names = [column['name'] for column in table.columns]
        return [dict(zip(names, row)) for row in table.rows], list(table.row_ids), published
    finally:
        table.rows.store.close()


def options():
    value = defaults()
    value['sourceMode'] = 'directory'
    return value


def test_packaged_practice_has_twelve_distinct_files_and_exact_captured_pixels():
    expected_names = {f'file-practice-{index:02d}-notes.txt' for index in range(1, 11)}
    expected_names |= {'file-practice-11-checklist.docx', 'file-practice-12-colors.png'}
    assert {path.name for path in COLLECTION.iterdir()} == expected_names
    rows, row_ids, published = read(options())
    assert len(rows) == len(set(row_ids)) == len({row['file_id'] for row in rows}) == 12
    assert all(row['reader_status'] == 'content' for row in rows)
    image = next(row for row in rows if row['name'].endswith('.png'))
    reference = image['image_content']
    header, pixels = decode_pixels(published[reference['sha256']], reference['sha256'])
    assert (header['width'], header['height'], header['channels'], header['dtype'], header['layout']) == (3, 2, 3, 'uint8', 'HWC')
    assert pixels == bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 128, 128, 128, 0, 0, 0])
    assert image['text_content'] is None
    documents = [row for row in rows if row is not image]
    assert all(row['image_content'] is None and row['text_content'] for row in documents)
    assert next(row for row in rows if row['name'].endswith('.docx'))['text_content'] == (
        'A synthetic document for File Input practice.\n\nText extraction reads stored text; it does not run OCR.')


def test_line_units_change_rows_but_retain_identity_and_bounded_preview():
    whole, _, _ = read(options())
    value = options()
    value['readers']['text']['units'] = 'line'
    lines, row_ids, _ = read(value)
    assert len(lines) == len(set(row_ids)) == 22
    original_ids = {row['name']: row['file_id'] for row in whole}
    for row in lines:
        assert row['file_id'] == original_ids[row['name']]
    for index in range(1, 11):
        name = f'file-practice-{index:02d}-notes.txt'
        units = [row for row in lines if row['name'] == name]
        assert [row['unit_index'] for row in units] == [0, 1]
        assert [row['text_content'] for row in units] == [
            f'Synthetic observation {index:02d}.', 'Check its file identity before choosing features.']
    preview = preview_files(COLLECTION, value)
    assert preview['collection']['selectedFiles'] == 8
    assert preview['collection']['complete'] is False
    assert preview['collection']['imagePixelsIncluded'] is False
    assert 'image_content' not in preview['columns']
