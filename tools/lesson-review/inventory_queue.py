#!/usr/bin/env python3
"""Render a local discovery inventory for reconciliation with the owner review ledger.
No status inference, approvals, source writes, project reads or application execution.
"""
import argparse
import json
from pathlib import Path


def render(packet):
    if packet.get('schema') != 1 or not isinstance(packet.get('entries'),list):
        raise ValueError('Unsupported discovery snapshot')
    rows=packet['entries']
    if len({row['entryID'] for row in rows})!=len(rows):
        raise ValueError('Duplicate discovery IDs')
    def safe(value): return str(value).replace('|',' / ').replace('\n',' ')
    lines=['| Review key | Topic | Type | Title |','| --- | --- | --- | --- |']
    for row in rows:
        lines.append('| '+ ' | '.join(safe(row[key]) for key in ['entryID','topic','type','title'])+' |')
        for child in row.get('assessments',[]):
            lines.append('| '+ ' | '.join(map(safe,[row['entryID']+'/'+child['operation'],row['topic'],'Assessment',child['title']]))+' |')
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory',type=Path)
    args=parser.parse_args()
    print(render(json.loads(args.inventory.read_text())),end='')
