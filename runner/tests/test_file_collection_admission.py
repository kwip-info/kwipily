"""Closed collection inventory admits no original directory or free member authority."""
import copy
import json
from pathlib import Path
import pytest
from runner.file_collection_contract import parse_manifest,validate_manifest,owned_member_paths
from runner.file_input_options import defaults
from runner.tests.test_operation_authorization import operation_authority,claim,publish
from runner.tests.test_preparation_operation_authorization import preparation_operation
from runner.tests.test_preview_operation_authorization import inventory
from runner.tests.test_evaluation_authorization import encoded,digest


def document(status='content'):
    member=dict(relativePath='folder/readme.txt',capturePath='members/000000.txt' if status=='content' else None,bytes=5,sha256=digest(b'hello') if status=='content' else None,extension='txt',status=status)
    if status=='error':member['error']='Capture refused.'
    selection={k:v for k,v in defaults().items() if k not in ('metadataFields','readers')}
    selection.update(sourceMode='directory',includeSubfolders=True)
    return dict(schemaVersion=1,kind='kwipFileCollection',sourceMode='directory',readMode='content',selection=selection,complete=True,members=[member])


def stage(f,doc=None):
    root=f[0];folder=root/'input';doc=doc or document();raw=encoded(doc)
    collection=folder/'file-collections'/digest(raw);collection.mkdir(parents=True)
    manifest=collection/'manifest.json';manifest.write_bytes(raw)
    for member in doc['members']:
        if member['status']=='content':
            path=collection/member['capturePath'];path.parent.mkdir(exist_ok=True);path.write_bytes(b'hello')
    configuration=defaults();configuration.update(doc['selection'])
    graph={'schemaVersion':1,'nodes':[{'id':'files','operator':'kwip.data.sourceFiles','config':configuration}],'edges':[]}
    (folder/'graph.json').write_bytes(encoded(graph));(folder/'bindings.json').write_bytes(encoded({'files':str(manifest)}));(folder/'node-labels.json').write_bytes(encoded({'files':'Selected files'}))
    import shutil
    shutil.rmtree(folder/'sources')
    inventory(f);return manifest


def test_preparation_admits_exact_collection_and_rechecks_member(preparation_operation):
    f=preparation_operation;manifest=stage(f)
    with claim(f) as authority:
        authority.check()
        (manifest.parent/'members/000000.txt').write_bytes(b'other')
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('change',['orphan','missing','wrong-pin','escape','absolute','duplicate','wrong-mode','table-binding','selection-change'])
def test_preparation_collection_negative_authority_controls(preparation_operation,change):
    f=preparation_operation;doc=document()
    if change=='escape':doc['members'][0]['capturePath']='../escape.txt'
    if change=='absolute':doc['members'][0]['capturePath']='/tmp/escape.txt'
    if change=='duplicate':doc['members'].append(copy.deepcopy(doc['members'][0]))
    if change=='wrong-mode':doc['readMode']='metadata'
    # Stage malformed path documents without ever following their authored path.
    if change in ('escape','absolute'):
        bad=doc;manifest=stage(f,document());raw=encoded(bad);manifest.write_bytes(raw)
        new=manifest.parent.parent/digest(raw);manifest.parent.rename(new);manifest=new/'manifest.json'
        (f[0]/'input/bindings.json').write_bytes(encoded({'files':str(manifest)}))
    else:manifest=stage(f,doc)
    if change=='orphan':(manifest.parent/'members/free.txt').write_bytes(b'free')
    if change=='missing':(manifest.parent/'members/000000.txt').unlink()
    if change=='wrong-pin':(manifest.parent/'members/000000.txt').write_bytes(b'wrong')
    if change=='selection-change':
        graph=json.loads((f[0]/'input/graph.json').read_bytes());graph['nodes'][0]['config']['includeHidden']=True;(f[0]/'input/graph.json').write_bytes(encoded(graph))
    if change=='table-binding':
        graph=json.loads((f[0]/'input/graph.json').read_bytes());graph['nodes'][0]['operator']='kwip.data.sourceCSV';graph['nodes'][0]['config']={};(f[0]/'input/graph.json').write_bytes(encoded(graph))
    inventory(f)
    with pytest.raises((ValueError,OSError)):claim(f)


def test_metadata_members_can_report_large_sizes_without_body_authority(preparation_operation):
    f=preparation_operation;doc=document('metadata');doc['readMode']='metadata';doc['selection']['readMode']='metadata';doc['members'][0]['bytes']=2**40
    stage(f,doc)
    with claim(f) as authority:authority.check()


@pytest.mark.parametrize('status',['metadata','error'])
def test_uncaptured_member_cannot_claim_a_body(status):
    doc=document(status);doc['members'][0]['capturePath']='members/000000.txt'
    with pytest.raises(ValueError):validate_manifest(doc)


def test_content_limits_and_duplicate_json_fields_fail_closed():
    doc=document();doc['members'][0]['bytes']=67_108_865
    with pytest.raises(ValueError):validate_manifest(doc)
    with pytest.raises(ValueError):parse_manifest(b'{"schemaVersion":1,"schemaVersion":1}')


def inspection(f):
    manifest=stage(f);config=json.loads((f[0]/'input/graph.json').read_bytes())['nodes'][0]['config']
    for name in ('graph.json','bindings.json','node-labels.json'):(f[0]/'input'/name).unlink()
    f[1].update(toolMode='sourceInspection',arguments=['--inspect-data',str(manifest),'--file-options',json.dumps(config)])
    inventory(f);return manifest


def test_file_inspection_admits_pinned_closed_collection(preparation_operation):
    f=preparation_operation;manifest=inspection(f)
    with claim(f) as authority:
        authority.check();(manifest.parent/'members/000000.txt').write_bytes(b'other')
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('change',['orphan','selection-change','original-directory-binding'])
def test_file_inspection_refuses_unbound_or_changed_selection(preparation_operation,change):
    f=preparation_operation;manifest=inspection(f)
    if change=='orphan':(manifest.parent/'members/free.txt').write_bytes(b'free')
    if change=='selection-change':
        config=json.loads(f[1]['arguments'][3]);config['includeHidden']=True;f[1]['arguments'][3]=json.dumps(config)
    if change=='original-directory-binding':f[1]['arguments'][1]=str(manifest.parent)
    inventory(f)
    with pytest.raises((ValueError,OSError)):claim(f)
