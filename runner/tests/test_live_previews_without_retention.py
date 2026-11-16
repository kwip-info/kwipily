"""Live samples and durable intermediate-table retention are separate choices."""
import json
from runner.dataflow_cli import run_dataflow
from runner.workload_policy import WorkloadPolicy


def test_live_preview_does_not_require_archiving_intermediate_cells(tmp_path):
    from runner.tests.test_dataflow_shared import config
    source=tmp_path/'input.csv';source.write_text('text,id\nPRIVATE_INTERMEDIATE_SENTINEL,1\nother,2\n')
    selected=config();selected['rows']=[{'source':'id','outputName':'id','included':True,'targetType':'keep'}]
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}},{'id':'selected','operator':'kwip.data.columns','config':selected}], 'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'selected','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));events=[]
    manifest=run_dataflow(path,{'source':str(source)},tmp_path/'output',events.append,retain_outputs=False,live_previews=True,workload_policy=WorkloadPolicy('balanced').to_document())
    source_event=next(e for e in events if e.get('event')=='dataflowPreview' and e['nodeID']=='source')
    assert source_event['outputs']['table']['preview']['rows'][0][0]=='PRIVATE_INTERMEDIATE_SENTINEL'
    assert manifest['observationRetention']=='requiredOnly'
    assert not any(e.get('nodeID')=='source' and e['kind']=='table' for e in manifest['artifacts'])
    assert not list((tmp_path/'output').glob('previews.json'))
    assert all(b'PRIVATE_INTERMEDIATE_SENTINEL' not in p.read_bytes() for p in (tmp_path/'output').iterdir() if p.is_file())
