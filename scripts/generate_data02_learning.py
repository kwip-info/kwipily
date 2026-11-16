#!/usr/bin/env python3
"""Copy the owner-approved authored DATA-02 teaching/examples into the native bundle.

No teaching text or graph configuration is maintained in Swift. --check refuses drift.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runner' / 'dataflow_examples'
DESTINATION = ROOT / 'Sources' / 'KwipMLCore' / 'Resources'

def resources():
    manifest = SOURCE / 'data02-lessons.json'
    catalog = json.loads(manifest.read_text())
    if catalog['schemaVersion'] != 1 or catalog['family'] != 'data-engineering' or len(catalog['lessons']) != 6:
        raise ValueError('DATA-02 must retain the six reviewed guided-practice records.')
    names = {'data02-lessons.json', 'data02-shared-guidance.md'}
    for lesson in catalog['lessons']:
        if lesson['isAssessed'] is not False or lesson['mode'] != 'guidedPractice':
            raise ValueError('Editable practice cannot acquire scored-assessment claims.')
        for field in ('graph', 'source', 'reading'):
            name = lesson[field]
            if Path(name).name != name or not name.startswith('data02-'):
                raise ValueError('Teaching resources must be local reviewed DATA-02 basenames.')
            names.add(name)
    return sorted(names)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name in resources():
        data = (SOURCE / name).read_bytes()
        destination = DESTINATION / name
        if args.check:
            if not destination.is_file() or destination.read_bytes() != data:
                raise SystemExit(f'Generated DATA-02 resource is stale: {name}; run scripts/generate_data02_learning.py')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    print('DATA-02 native teaching resources verified.' if args.check else 'DATA-02 native teaching resources generated.')

if __name__ == '__main__':
    main()
