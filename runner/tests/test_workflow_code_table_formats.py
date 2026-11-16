"""All local format nodes fit the existing bounded source-display protocol."""
import json
from pathlib import Path
from runner.tests.test_workflow_code import snapshot
from runner.workflow_code import generate_package


def test_all_table_formats_with_full_registries_fit_display_event():
    value=snapshot()
    value['modelRegistry']=json.loads((Path(__file__).parents[2]/'Sources/KwipMLCore/Resources/components.json').read_text())
    value['preparation']['graph']['nodes']=[]
    value['preparation']['nodeLabels']={}
    value['document']['nodes']=[]
    formats=['csv','parquet','json','jsonl','excel','markdown','xml','html','database']
    for fmt in formats:
        config={'format':fmt,'options':{}}
        if fmt in ('json','jsonl','excel','markdown','html'):
            config['options']['schema_overrides']={f'column_{index:03d}':'String' for index in range(128)}
        node={'id':fmt,'operator':'kwip.data.sourceCSV','config':config}
        value['preparation']['graph']['nodes'].append(node)
        value['preparation']['nodeLabels'][fmt]=fmt
        value['document']['nodes'].append({'address':{'domain':'data','localID':fmt},'name':fmt,'x':0,'y':0,
            'payload':{'kind':'dataOperator','operatorID':node['operator'],'config':config}})
    package=generate_package(value)
    event=dict(package,event='codeExport')
    assert len(json.dumps(event,allow_nan=False).encode()) <= 1_000_000
    captured=json.loads(package['files']['workflow-plan.json'])
    assert {node['config']['format'] for node in captured['preparation']['graph']['nodes']}==set(formats)
