import copy
import pytest
from runner import dataflow as df
from runner.text_encoding import DEFAULTS, apply, load, decode
from runner.polars_preparation import preparation_engine


def table(role,rows,ids):
    return df.derive(df.Table(({'name':'text','dtype':'string','nullable':True},),tuple((x,) for x in rows),tuple(ids),'a'*64,role,partition_digest='b'*64))


def test_training_vocabulary_and_heldout_unknown_without_leakage():
    c={**DEFAULTS,'textColumn':'text','vocabularyLimit':8}
    output=apply({'training':table('training',['aba','a'],['a','b']),'validation':table('validation',['ac'],['c'])},c,'c'*64,lambda:None)
    state=load(df.json.dumps(output['state']))
    assert state['vocabulary']==['<unk>','<pad>','a','b']
    assert output['training'].rows[0][-2:]==([2,3,2],3)
    assert output['validation'].rows[0][-2:]==([2,0],2)
    assert decode([2,3,2],state)=='aba'
    assert output['validation'].fitted_states[-1]==state['contentDigest']
    assert state['fitEvidence']['observationIDs']==['a','b']
    assert output['vocabulary'].rows[2][:3]==(2,'a',3)
    with pytest.raises(ValueError):apply({'training':table('training',['x'],['a']),'validation':table('validation',['y'],['a'])},c,'c'*64,lambda:None)


@pytest.mark.parametrize('engine',['disk','polars'])
def test_supplied_word_vocabulary_normalization_special_tokens_and_storage(engine):
    c={**DEFAULTS,'vocabularySource':'supplied','vocabulary':['café','world'],'tokenizer':'word','unicodeNormalization':'NFC','letterCase':'lower','whitespace':'collapse','bos':True,'eos':True,'keepSourceText':False}
    with preparation_engine(engine):
        output=apply({'table':table('unassigned',['CAFE\u0301  WORLD'],['a'])},c,'c'*64,lambda:None)
        assert output['table'].columns[0]['dtype']=='int64Sequence'
        assert output['table'].rows[0]==([2,4,0,5,3],5)
        assert load(df.json.dumps(output['state']))['configuration']==c
        t=output['table']
        artifact={'schemaVersion':4,'kind':'table','columns':list(t.columns),'rows':[list(r) for r in t.rows],'lineage':{**t.evidence(),'observationIDs':list(t.row_ids),'dataVersion':t.data_version,'partitionDigest':t.partition_digest,'fittedStateDigests':list(t.fitted_states),'diagnostics':list(t.diagnostics)}}
        from runner.dataflow_cache import table_from_artifact
        assert artifact['schemaVersion']==4
        assert table_from_artifact(artifact).rows==tuple(output['table'].rows)


def test_options_missing_empty_unknown_and_frequency_limit():
    c={**DEFAULTS,'minimumFrequency':2,'vocabularyLimit':3,'idOrder':'lexicographic'}
    o=apply({'training':table('training',['bb aa c'],['a'])},c,'c'*64,lambda:None)
    assert o['state']['vocabulary']==['<unk>','<pad>',' ']
    with pytest.raises(ValueError):apply({'training':table('training',[None],['a'])},c,'c'*64,lambda:None)
    c={**DEFAULTS,'vocabularySource':'supplied','vocabulary':['a'],'missingText':'empty','unknownPolicy':'error'}
    assert apply({'table':table('unassigned',[None],['a'])},c,'c'*64,lambda:None)['table'].rows[0][-2:]==([],0)
    with pytest.raises(ValueError):apply({'table':table('unassigned',['b'],['a'])},c,'c'*64,lambda:None)


def test_tampered_state_and_cancellation():
    c={**DEFAULTS}
    state=apply({'training':table('training',['a'],['a'])},c,'c'*64,lambda:None)['state']
    state['vocabulary'].append('stolen')
    with pytest.raises(ValueError):load(df.json.dumps(state))
    def cancelled():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):apply({'training':table('training',['a'],['a'])},c,'c'*64,cancelled)


@pytest.mark.parametrize('options,text,expected',[
    ({'unicodeNormalization':'none'},'e\u0301',['e','\u0301']),
    ({'unicodeNormalization':'NFC'},'e\u0301',['é']),
    ({'unicodeNormalization':'NFKC'},'Ａ',['A']),
    ({'letterCase':'lower'},'A',['a']),
    ({'letterCase':'casefold'},'ß',['s','s']),
    ({'whitespace':'preserve'},' a\n',[' ','a','\n']),
    ({'whitespace':'trim'},' a\n',['a']),
    ({'whitespace':'collapse'},' a\n b ',['a',' ','b']),
    ({'tokenizer':'word'},'a  b',['a','  ','b']),
])
def test_each_text_policy_with_independent_expected_tokens(options,text,expected):
    vocabulary=list(dict.fromkeys(expected));c={**DEFAULTS,'vocabularySource':'supplied','vocabulary':vocabulary,**options}
    out=apply({'table':table('unassigned',[text],['row'])},c,'c'*64,lambda:None)
    state=out['state'];ids=out['table'].rows[0][-2]
    assert [state['vocabulary'][i] for i in ids]==expected


def test_vocabulary_limit_frequency_order_special_roles_and_compact_identity():
    t=table('training',['bba','a','c'],['document:0001','document:0002','document:0003'])
    o=apply({'training':t},{**DEFAULTS,'minimumFrequency':2,'vocabularyLimit':3,'idOrder':'frequency','pad':False},'c'*64,lambda:None)
    assert o['state']['vocabulary']==['<unk>','a','b']
    assert o['state']['frequencies']==[0,2,2]
    state=load(df.json.dumps(o['state']));assert list(df.fit_observation_ids(state))==list(t.row_ids)
    c={**DEFAULTS,'vocabularySource':'supplied','vocabulary':['a'],'unknownPolicy':'error','pad':False,'bos':True,'eos':True,'vocabularyLimit':3,'outputColumn':'ids','keepSourceText':False}
    out=apply({'table':table('unassigned',['a'],['a'])},c,'c'*64,lambda:None)
    assert out['state']['vocabulary']==['<bos>','<eos>','a']
    assert out['table'].rows[0]==([0,2,1],3)
    assert [col['name'] for col in out['table'].columns]==['ids','ids_length']
    with pytest.raises(ValueError):apply({'table':table('unassigned',['a'],['a'])},{**c,'outputColumn':'text','keepSourceText':True},'c'*64,lambda:None)
