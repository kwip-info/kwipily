"""Independent text/classifier lifecycle and refusal checks, not UI snapshots."""
import copy
import json
import subprocess
import sys
import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from runner import document_classification as dc

TEXTS=['invoice payment overdue', 'payment invoice paid', 'invoice total payment',
       'contract signed agreement', 'agreement contract terms', 'signed legal contract',
       'support login problem', 'login support help', 'problem help support']
LABELS=['Invoice']*3+['Contract']*3+['Support']*3


def test_features_reference_and_training_only_vocabulary():
    c=dc.feature_defaults();state=dc.fit_features(TEXTS,c)
    x=dc.transform(TEXTS,state)
    ref=TfidfVectorizer(**dc.vectorizer_options(c));expected=ref.fit_transform(TEXTS)
    assert sparse.isspmatrix_csr(x)
    assert state['vocabulary']==ref.get_feature_names_out().tolist()
    np.testing.assert_allclose(x.toarray(),expected.toarray(),atol=1e-12)
    before=copy.deepcopy(state)
    new=dc.transform(['unseenvalidationword invoice'],state)
    assert state==before and 'unseenvalidationword' not in state['vocabulary']
    assert new.shape==(1,len(state['vocabulary']))


def test_multiclass_binary_save_reload_and_independent_predictions(tmp_path):
    for texts,labels in [(TEXTS,LABELS),(TEXTS[:6],LABELS[:6])]:
        model=dc.fit(texts,labels,dc.feature_defaults(),dc.estimator_defaults())
        unseen=['payment total invoice','signed contract agreement','login help']
        prediction=dc.predict(model,unseen)
        state=model['processor'];x=dc.transform(texts,state)
        ref=LogisticRegression(**dc.estimator_options(dc.estimator_defaults())).fit(x,labels)
        np.testing.assert_allclose(prediction['probabilities'],ref.predict_proba(dc.transform(unseen,state)),atol=1e-10)
        assert prediction['classes']==ref.predict(dc.transform(unseen,state)).tolist()
        package=tmp_path/('binary' if len(texts)==6 else 'multi')
        dc.export(model,package)
        # Portable program is tested as a separate interpreter, no training sources.
        inputfile=tmp_path/'input.json';inputfile.write_text(json.dumps(unseen))
        actual=json.loads(subprocess.check_output([sys.executable,str(package/'inference.py'),str(inputfile)]))
        assert actual==prediction
        loaded=dc.load((package/'state.json').read_bytes())
        assert dc.predict(loaded,unseen)==prediction


def test_evaluation_is_full_population_and_independent_reference():
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
    m=dc.fit(TEXTS,LABELS,dc.feature_defaults(),dc.estimator_defaults())
    labels=['Invoice','Support','Contract'];texts=['invoice payment','contract signed','support problem']
    p=dc.predict(m,texts);e=dc.evaluate(m,texts,labels)
    assert e['count']==3
    assert e['accuracy']==accuracy_score(labels,p['classes'])
    assert e['balancedAccuracy']==balanced_accuracy_score(labels,p['classes'])
    assert e['macroF1']==f1_score(labels,p['classes'],labels=m['classes'],average='macro',zero_division=0)
    assert e['confusionMatrix']==confusion_matrix(labels,p['classes'],labels=m['classes']).tolist()
    with pytest.raises(ValueError,match='unseen'):dc.evaluate(m,texts,['Unknown']*3)


def test_invalid_labels_empty_text_state_tamper_and_resources():
    for labels in [['one']*9,LABELS[:-1],LABELS[:-1]+['']]:
        with pytest.raises(ValueError):dc.fit(TEXTS,labels,dc.feature_defaults(),dc.estimator_defaults())
    with pytest.raises(ValueError):dc.fit_features(['   '],dc.feature_defaults())
    m=dc.fit(TEXTS,LABELS,dc.feature_defaults(),dc.estimator_defaults())
    broken=copy.deepcopy(m);broken['processor']['idf'][0]=float('nan')
    with pytest.raises(ValueError):dc.load(json.dumps(broken).encode())
    broken=copy.deepcopy(m);broken['classes'].reverse()
    with pytest.raises(ValueError):dc.load(json.dumps(broken).encode())
    with pytest.raises(ValueError):dc.predict(m,[''])
    with pytest.raises(ValueError):dc.transform(['x'*1_048_577],m['processor'])
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):dc.fit_features(TEXTS,dc.feature_defaults(),check=cancel)


def test_nondefault_character_features_and_unicode():
    c={**dc.feature_defaults(),'analyzer':'char','ngramMin':2,'ngramMax':3,'lowercase':False,'stripAccents':'unicode','sublinearTF':True,'normalization':'l1','minimumDF':1,'maximumDF':1.0}
    texts=['Café AB', 'cafe XY', 'école AB']
    state=dc.fit_features(texts,c)
    ref=TfidfVectorizer(**dc.vectorizer_options(c)).fit(texts)
    np.testing.assert_allclose(dc.transform(texts,state).toarray(),ref.transform(texts).toarray(),atol=1e-12)


def test_dataflow_fitted_feature_mode_preserves_rows_and_refuses_leakage():
    from runner import dataflow as df
    from runner.text_encoding import apply,load,DEFAULTS
    cols=({'name':'text','dtype':'string','nullable':False},{'name':'category','dtype':'string','nullable':False})
    def table(role,texts,labels,ids):
        return df.derive(df.Table(cols,tuple(zip(texts,labels)),tuple(ids),'a'*64,role,partition_digest='b'*64))
    c={**DEFAULTS,'mode':'documentFeatures','featureSettings':dc.feature_defaults(),'outputColumn':'features'}
    train=table('training',TEXTS,LABELS,[f't{i}' for i in range(9)])
    val=table('validation',['invoice unseen','contract new'],['Invoice','Contract'],['v0','v1'])
    out=apply({'training':train,'validation':val},c,'c'*64,lambda:None)
    state=load(json.dumps(out['state']))
    assert state['processor']['vocabulary']==dc.fit_features(TEXTS,dc.feature_defaults())['vocabulary']
    assert list(out['training'].row_ids)==list(train.row_ids)
    assert out['training'].columns[-1]['name']=='features_nonzero'
    assert 'unseen' not in state['processor']['vocabulary']
    duplicate=table('validation',[TEXTS[0]],['Invoice'],['v3'])
    with pytest.raises(ValueError,match='duplicate'):apply({'training':train,'validation':duplicate},c,'c'*64,lambda:None)


def test_real_preparation_worker_fit_export_and_capture_policy(tmp_path):
    import hashlib
    from runner import dataflow as df
    from runner.dataflow_cli import run_dataflow
    from runner.sklearn_workflow import run_prepared
    from runner.text_encoding import DEFAULTS
    def node(id,op,config):return dict(id=id,operator='kwip.data.'+op,config=config)
    def edge(a,ap,b,bp):return {'from':{'node':a,'port':ap},'to':{'node':b,'port':bp}}
    source=tmp_path/'documents.csv'
    source.write_text('text,category\n'+''.join(f'"{text} reference{i}",{label}\n' for i in range(8) for text,label in zip(TEXTS,LABELS)))
    def defaults(name):return copy.deepcopy(df.DEFINITIONS[name]['defaultConfig'])
    part=defaults('partition');part.update(strategy='random',stratifyColumn='category')
    g=dict(schemaVersion=1,nodes=[node('source','sourceCSV',defaults('sourceCSV')),node('split','partition',part),node('features','textEncode',{**DEFAULTS,'mode':'documentFeatures','featureSettings':dc.feature_defaults(),'outputColumn':'features'})],edges=[edge('source','table','split','table')]+[edge('split',r,'features',r) for r in ['training','validation','test']])
    graph=tmp_path/'graph.json';graph.write_text(json.dumps(g));out=tmp_path/'prepared'
    run_dataflow(graph,{'source':str(source)},out,lambda _:None)
    raw=(out/'manifest.json').read_bytes()
    request=dict(schemaVersion=1,backendID='sklearn',configuration={'task':'documentClassification','settings':dc.estimator_defaults()},capturePredictions=True,
       prepared=dict(manifest=str(out/'manifest.json'),manifestSHA256=hashlib.sha256(raw).hexdigest(),graphDigest=json.loads(raw)['graphDigest'],trainingNode='features',trainingPort='training',validationNode='features',validationPort='validation',featureColumns=['text'],targetColumn='category'))
    result=run_prepared(request,tmp_path/'fit',lambda _:None)
    assert result['task']=='documentClassification' and result['evaluation']['accuracy']>=0.9
    assert result['provenance']['trainingSamples']+result['evaluation']['count']<72
    assert result['predictions']['classOrder']==['Contract','Invoice','Support']
    working=json.loads((tmp_path/'fit/predictions.json').read_text())
    assert len(working['rows'])==result['evaluation']['count']>8
    assert working['lineage']['observationIDs']==result['provenance']['validationRowIDs']
    assert working['lineage']['role']=='validation'
    assert result['predictionColumns']==working['columns']
    assert [c['name'] for c in working['columns'][:4]]==['category','predicted_category','top_probability','classification_status']
    model=dc.load((tmp_path/'fit/portable/state.json').read_bytes())
    assert dc.predict(model,['invoice payment total'])['classes']==['Invoice']
    assert not any('preparation' in p.name for p in (tmp_path/'fit/portable').iterdir())
    request['capturePredictions']=False
    other=run_prepared(request,tmp_path/'uncaptured',lambda _:None)
    assert 'predictions' not in other and (tmp_path/'uncaptured/model-state.json').exists()
    assert len(json.loads((tmp_path/'uncaptured/predictions.json').read_text())['rows'])==other['evaluation']['count']


def test_unseen_document_prediction_worker_emits_complete_typed_output(tmp_path):
    import hashlib
    from runner import dataflow as df
    from runner.dataflow_cli import run_dataflow
    from runner.sklearn_workflow import run_prepared
    from runner.dataflow_table_reader import read_table_artifact
    m=dc.fit(TEXTS,LABELS,dc.feature_defaults(),dc.estimator_defaults())
    state=tmp_path/'model-state.json';state.write_bytes(dc.canonical(m))
    source=tmp_path/'unseen.csv';source.write_text('document,text\nnew-1,"invoice payment total"\nnew-2,"contract signed agreement"\nscan,""\n')
    graph=tmp_path/'graph.json';graph.write_text(json.dumps(dict(schemaVersion=1,nodes=[dict(id='source',operator='kwip.data.sourceCSV',config=df.DEFINITIONS['sourceCSV']['defaultConfig'])],edges=[])))
    prepared=tmp_path/'prepared';run_dataflow(graph,{'source':str(source)},prepared,lambda _:None)
    raw=(prepared/'manifest.json').read_bytes()
    request=dict(schemaVersion=1,backendID='sklearn',configuration={'task':'documentClassification','operation':'predict','settings':dc.estimator_defaults()},capturePredictions=False,
        model={'path':str(state),'sha256':hashlib.sha256(state.read_bytes()).hexdigest()},prepared=dict(manifest=str(prepared/'manifest.json'),manifestSHA256=hashlib.sha256(raw).hexdigest(),graphDigest=json.loads(raw)['graphDigest'],trainingNode='source',trainingPort='table',validationNode='source',validationPort='table',featureColumns=['text'],targetColumn='category'))
    result=run_prepared(request,tmp_path/'result',lambda _:None)
    assert result['evaluation']=={'count':3,'classified':2,'failed':1,'operation':'predict'}
    assert 'accuracy' not in result['evaluation'] and 'predictions' not in result
    e=next(x for x in result['artifacts'] if x['path']=='predictions.json')
    table=read_table_artifact(tmp_path/'result/predictions.json',e['sha256'])
    try:
        assert len(table.rows)==3
        rows=list(table.rows);index=table.index('predicted_category')
        assert rows[0][index]=='Invoice' and rows[1][index]=='Contract' and rows[2][index] is None
        assert rows[2][table.index('classification_status')]=='No extractable text'
    finally:table.rows.store.close();table.row_ids.store.close()
    bad=copy.deepcopy(request);bad['model']['sha256']='0'*64
    with pytest.raises(ValueError,match='pin'):run_prepared(bad,tmp_path/'bad',lambda _:None)


def test_frozen_evaluation_and_complete_table_output_delivery(tmp_path):
    import csv
    import hashlib
    from runner.dataflow_cli import run_dataflow
    from runner.sklearn_workflow import run_prepared
    model=dc.fit(TEXTS,LABELS,dc.feature_defaults(),dc.estimator_defaults())
    state=tmp_path/'model.json';state.write_bytes(dc.canonical(model))
    source=tmp_path/'labeled.csv'
    # More than the optional 8-row preview, with intentionally wrong labels.
    texts=['invoice payment','contract agreement','support help']*5
    labels=['Invoice','Support','Contract']*5
    with source.open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['text','category']);w.writerows(zip(texts,labels))
    graph=tmp_path/'source-graph.json';graph.write_text(json.dumps({'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}],'edges':[]}))
    prepared=tmp_path/'prepared';run_dataflow(graph,{'source':str(source)},prepared,lambda _:None)
    raw=(prepared/'manifest.json').read_bytes();pin=hashlib.sha256(state.read_bytes()).hexdigest()
    request=dict(schemaVersion=1,backendID='sklearn',configuration={'task':'documentClassification','operation':'evaluate','settings':dc.estimator_defaults(),'reviewThreshold':0.99},capturePredictions=False,
        model={'path':str(state),'sha256':pin},prepared=dict(manifest=str(prepared/'manifest.json'),manifestSHA256=hashlib.sha256(raw).hexdigest(),graphDigest=json.loads(raw)['graphDigest'],trainingNode='source',trainingPort='table',validationNode='source',validationPort='table',featureColumns=['text'],targetColumn='category'))
    output=tmp_path/'evaluated';result=run_prepared(request,output,lambda _:None)
    expected=dc.evaluate(model,texts,labels)
    assert result['evaluation']==expected and result['evaluation']['count']==15
    assert result['evaluation']['accuracy']==pytest.approx(1/3)
    assert 'predictions' not in result and hashlib.sha256(state.read_bytes()).hexdigest()==pin
    artifact=next(x for x in result['artifacts'] if x['path']=='predictions.json')
    captured=output/'predictions.json'
    delivery={'schemaVersion':1,'nodes':[{'id':'predictions','operator':'kwip.data.capturedTable','config':{'sha256':artifact['sha256']}},{'id':'save','operator':'kwip.data.outputDelimited','config':dict(filename='predicted.csv',separator=',',columns=[],overwrite=False,format='csv',options={})}], 'edges':[{'from':{'node':'predictions','port':'table'},'to':{'node':'save','port':'table'}}]}
    route=tmp_path/'delivery.json';route.write_text(json.dumps(delivery));destination=tmp_path/'destination';destination.mkdir();events=[]
    run_dataflow(route,{'predictions':str(captured)},tmp_path/'delivery-run',events.append,destination_bindings={'save':str(destination)})
    with (destination/'predicted.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==15 and [r['predicted_category'] for r in rows]==['Invoice','Contract','Support']*5
    assert all(r['classification_status']=='Below review threshold' for r in rows)
    assert set(k for k in rows[0] if k.startswith('probability_'))=={'probability_'+str(i)+'_'+c for i,c in enumerate(model['classes'])}
    receipt=next(e for e in events if e.get('event')=='dataflowOutputSaved');assert receipt['rowCount']==15 and receipt['completeSelectedTable']
    # Normal destination handling must refuse replay into an existing filename.
    with pytest.raises(Exception):run_dataflow(route,{'predictions':str(captured)},tmp_path/'repeat-delivery',lambda _:None,destination_bindings={'save':str(destination)})


def test_high_dimensional_sparse_features_stay_sparse_and_budget_refuses(monkeypatch):
    texts=['category'+str(i%3)+' '+ ' '.join('term'+str(i*20+j) for j in range(20)) for i in range(1000)]
    processor=dc.fit_features(texts,{**dc.feature_defaults(),'vocabularyLimit':30000,'ngramMax':1})
    matrix=dc.transform(texts,processor)
    assert sparse.isspmatrix_csr(matrix) and matrix.shape[0]==1000 and matrix.shape[1]>20000
    assert matrix.nnz<=21000
    # Sparse storage is a tiny fraction of the corresponding dense float64 payload.
    assert matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes < matrix.shape[0]*matrix.shape[1]*8/100
    monkeypatch.setattr(dc,'MAX_NNZ',100)
    with pytest.raises(ValueError,match='Sparse'):
        dc.transform(texts,processor)
