import copy
import time
import unittest
import unicodedata
import torch
from runner import text_generation as g
from runner.portable_text_encoding import encode


def metadata(eos=False,bos=False):
    c=dict(unknownPolicy='unk',pad=True,bos=bos,eos=eos,missingText='error',unicodeNormalization='none',letterCase='none',whitespace='preserve',tokenizer='character')
    v=['<unk>','<pad>']+(['<bos>'] if bos else [])+(['<eos>'] if eos else [])+['a','b',' ']
    return dict(schemaVersion=1,configuration=c,vocabulary=v,unicodeVersion=unicodedata.unidata_version,sourceStateSHA256='a'*64)


class Predictor:
    def __init__(self,meta=None):
        self.tokenizers={'tokens':meta or metadata()}
        v=self.tokenizers['tokens']['vocabulary'];self.v=v
        self.plan={'schemaVersion':2,'inputs':[dict(name='tokens',dtype='int64',representation='tokens',sampleShape=[3],vocabularySize=len(v),paddingID=1)],'sequenceObjective':{'inputName':'tokens'}}
        self.names={'tokens':self.plan['inputs'][0]};self.output_shape=(3,len(v));self.encode_tokenizer=encode;self.named_values=False
        class Bigram(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('calls',torch.tensor(0));self.inputs=[]
            def forward(self,x):
                self.calls.add_(1);self.inputs.append(x.cpu().clone())
                out=torch.full((*x.shape,len(v)),-100.,device=x.device)
                out.scatter_(-1,torch.where(x==v.index('a'),v.index('b'),v.index('a')).unsqueeze(-1),100.)
                return out
        self.model=Bigram()


class GenerationTests(unittest.TestCase):
    def test_feedback_last_valid_sliding_and_state(self):
        p=Predictor();before=copy.deepcopy(p.model.state_dict())
        result=g.generate(p,['a','a'],dict(maxNewTokens=5))
        self.assertEqual([r['generated_text'] for r in result['rows']],['babab','babab'])
        self.assertEqual(result['rows'][0]['generated_ids'],[3,2,3,2,3])
        self.assertTrue(torch.equal(p.model.calls,before['calls']))
        self.assertEqual(p.model.inputs[0].tolist(),[[2,1,1]])
        self.assertEqual(p.model.inputs[1].tolist(),[[2,3,1]])
        self.assertEqual(p.model.inputs[3].tolist(),[[3,2,3]])
        self.assertFalse(p.model.training)
    def test_overflow_and_empty(self):
        p=Predictor()
        for prompt in ['', 'ababa']:
            with self.assertRaises(ValueError):g.generate(p,[prompt],{})
        r=g.generate(p,['ababa'],dict(maxNewTokens=1,longPrompt='latest'))
        self.assertEqual(r['evidence'][0]['droppedPromptTokens'],2)
    def test_sampler_ties_filtering_and_rng_isolation(self):
        scores=torch.tensor([9.,9.,1.,0.])
        rng=torch.Generator().manual_seed(3)
        self.assertEqual(g.choose(scores,{0},g.settings({}),rng),1)
        global_state=torch.get_rng_state().clone()
        a=g.generate(Predictor(),['a'],dict(method='sample',temperature=2.,topK=1,topP=.5,seed=123,maxNewTokens=8))
        b=g.generate(Predictor(),['a'],dict(method='sample',temperature=2.,topK=1,topP=.5,seed=123,maxNewTokens=8))
        self.assertEqual(a,b);self.assertTrue(torch.equal(global_state,torch.get_rng_state()))
        self.assertEqual(g.filtered_probabilities(torch.tensor([0.,0.,0.]),set(),1.,0,.4).tolist(),[.5,.5,0.])
    def test_eos_not_appended_to_prompt_and_specials(self):
        p=Predictor(metadata(eos=True,bos=True));v=p.v
        class EOS(torch.nn.Module):
            def forward(self,x):
                self.seen=x.clone();z=torch.zeros(1,3,len(v));z[:,:,v.index('<eos>')]=10;return z
        p.model=EOS()
        r=g.generate(p,[''],dict(maxNewTokens=4))
        self.assertEqual(r['rows'][0],dict(generated_text='',generated_ids=[v.index('<eos>')],new_tokens=1,stop_reason='eos'))
        self.assertEqual(p.model.seen.tolist(),[[v.index('<bos>'),1,1]])
    def test_guards_cancel_deadline_and_budgets(self):
        for config in [dict(maxNewTokens=0),dict(temperature=float('nan')),dict(topP=0),dict(seed=True),dict(foo=1)]:
            with self.assertRaises(ValueError):g.generate(Predictor(),['a'],config)
        with self.assertRaises(ValueError):g.generate(Predictor(),['a']*129,{})
        with self.assertRaises(ValueError):g.generate(Predictor(),['a']*128,dict(maxNewTokens=1024))
        with self.assertRaises(InterruptedError):g.generate(Predictor(),['a'],{},cancelled=lambda:True)
        with self.assertRaises(TimeoutError):g.generate(Predictor(),['a'],{},deadline=time.monotonic()-1)
        with self.assertRaises(ValueError):g.choose(torch.tensor([float('nan'),1.]),set(),g.settings({}),torch.Generator())
        with self.assertRaises(ValueError):g.choose(torch.tensor([1.,2.]),{0,1},g.settings({}),torch.Generator())
    def test_contract_and_output_shape_refusals(self):
        p=Predictor();p.output_shape=(2,5)
        with self.assertRaises(ValueError):g.generate(p,['a'],{})
        p=Predictor();p.plan['inputs'].append(p.plan['inputs'][0])
        with self.assertRaises(ValueError):g.generate(p,['a'],{})
        p=Predictor();p.model.forward=lambda x:torch.zeros(1,1,5)
        with self.assertRaises(ValueError):g.generate(p,['a'],{})
    def test_mid_loop_cancellation_restores_state_and_never_returns_partial_rows(self):
        p=Predictor();calls=0
        def cancelled():
            nonlocal calls
            calls+=1
            return calls>10
        with self.assertRaises(InterruptedError):g.generate(p,['a','b'],dict(maxNewTokens=8),cancelled=cancelled)
        self.assertEqual(p.model.calls.item(),0)
    def test_large_word_output_refuses_before_unbounded_join(self):
        p=Predictor();p.tokenizers['tokens']['configuration']['tokenizer']='word';p.v[4]='z'*100000
        class LongWord(torch.nn.Module):
            def forward(self,x):
                out=torch.zeros(1,3,5);out[:,:,4]=10;return out
        p.model=LongWord()
        with self.assertRaisesRegex(ValueError,'byte allowance'):g.generate(p,['a'],dict(maxNewTokens=64))
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_real_forward(self):
        p=Predictor();r=g.generate(p,['a'],dict(maxNewTokens=3,device='mps'))
        self.assertEqual(r['rows'][0]['generated_text'],'bab')



class ManagedGenerationTests(unittest.TestCase):
    def fixture(self,root,mode='one',table=None,**overrides):
        import hashlib,json,uuid
        from pathlib import Path
        root=Path(root).resolve()/'workflow-operations'/str(uuid.uuid4()).upper()/str(uuid.uuid4()).upper();root.mkdir(parents=True)
        (root/'source').mkdir();(root/'input').mkdir()
        source={'inference.py':b'from runner.tests.test_text_generation import Predictor\nload_model=Predictor\n','model.py':b'# synthetic trusted fixture\n','checkpoint.pt':b'synthetic checkpoint authority','input-contract.json':json.dumps(dict(schemaVersion=4,objective=dict(recipe=dict(componentID='loss_cross_entropy',schemaVersion=1,configurationVersion=1,config=dict(class_axis=-1,class_weights=[],ignore_index=-100,label_smoothing=0,reduction='mean',target_mode='class_indices')),scalarization='mean'))).encode(),'named_inputs.py':b'# fixture\n','tokenizers.json':json.dumps(metadata()).encode(),'portable_text_encoding.py':b'# fixture\n'}
        entries=[]
        for name,raw in source.items():
            (root/'source'/name).write_bytes(raw);entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        m=json.dumps(dict(schemaVersion=1,kind='kwipmlTrainedModel',files=entries)).encode();(root/'source/export-manifest.json').write_bytes(m)
        c={**g.DEFAULTS,'maxNewTokens':3,'promptMode':mode,'prompt':'a','promptColumn':'text','textColumn':'generated_text','idsColumn':'generated_ids','countColumn':'new_tokens','stopColumn':'stop_reason',**overrides}
        inner=dict(schemaVersion=1,projectID=root.parent.name,operationID=root.name,requestIdentity='b'*64,sourceManifestSHA256=hashlib.sha256(m).hexdigest(),configuration=c)
        (root/'input/generation.json').write_text(json.dumps(inner));(root/'input/generation-kernel.py').write_bytes(b'# captured runtime kernel\n')
        if table is not None:
            from runner import dataflow as df
            from runner.image_artifacts import table_schema_version
            table=copy.deepcopy(table);ids=table['lineage']['observationIDs'];cols=table['columns'];rows=table['rows']
            table.update(schemaVersion=table_schema_version(cols),kind='table')
            table['lineage'].update(role='unassigned',sourceDigest='a'*64,membershipDigest=df.membership_digest(ids),dataVersion=df.digest(dict(columns=cols,rows=rows,ids=ids)),partitionDigest='',fittedStateDigests=[],diagnostics=[])
            (root/'input/prompts.table.json').write_text(json.dumps(table))
        inventory=[]
        for path in root.glob('*/*'):
            b=path.read_bytes();inventory.append(dict(path=str(path.relative_to(root)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        class Authority:pass
        a=Authority();a._path=root;a.request=dict(operationID=root.name,projectID=root.parent.name,requestIdentity='b'*64,toolMode='textGeneration',arguments=['--generate-text',str(root/'input/generation.json')],inputFiles=inventory)
        return a
    def test_full_table_and_ids_roundtrip(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from runner.text_generation_worker import execute
        from runner.dataflow_table_reader import read_table_artifact
        table=dict(columns=[dict(name='text',dtype='string',nullable=False),dict(name='order',dtype='int64',nullable=False)],rows=[['a',10],['b',20]],lineage=dict(observationIDs=['original-a','original-b']))
        with tempfile.TemporaryDirectory() as root:
            a=self.fixture(root,'table',table);events=[];execute(a,events.append,lambda:None)
            p=a._path/'output/table.json';t=read_table_artifact(p,hashlib.sha256(p.read_bytes()).hexdigest(),lambda:None)
            self.assertEqual(list(t.row_ids),['original-a','original-b'])
            self.assertEqual([list(r)[:3] for r in t.rows],[['a',10,'bab'],['b',20,'aba']])
            self.assertEqual(events[-1]['event'],'generationCompleted')
    def test_refuses_collision_excess_rows_and_changed_authority_without_completion(self):
        import tempfile
        from runner.text_generation_worker import execute
        table=dict(columns=[dict(name='text',dtype='string',nullable=False)],rows=[['a']],lineage=dict(observationIDs=['id']))
        for changed in ('collision','size','tamper','cancel'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as root:
                a=self.fixture(root,'table',table,**({'textColumn':'text'} if changed=='collision' else {'maxTotalTokens':1} if changed=='size' else {}))
                if changed=='tamper':(a._path/'source/checkpoint.pt').write_bytes(b'changed')
                events=[]
                def check():
                    if changed=='cancel':raise InterruptedError('cancelled')
                with self.assertRaises((ValueError,InterruptedError)):execute(a,events.append,check)
                self.assertFalse(any(e['event']=='generationCompleted' for e in events))

if __name__=='__main__':unittest.main()
