"""Reviewed single-label document classifier. Portable inference never imports sklearn.

No executable state, arbitrary tokenizers, downloads or implicit fitting at prediction.
Sparse matrices exist only inside the isolated operation; required fitted state is portable.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
import warnings

import numpy as np
from scipy import sparse

FORMAT='kwipml.document-classifier.v1'
MAX_STATE=32_000_000
MAX_TEXT=1_048_576
MAX_CORPUS=64_000_000
MAX_NNZ=3_000_000
MAX_FEATURES=65536
MAX_CLASSES=128
MAX_COEFFICIENTS=1_000_000
MAX_TERMS=262144


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sealed(value):return {**value,'contentDigest':digest(value)}


def feature_defaults():
    return dict(analyzer='word',ngramMin=1,ngramMax=2,lowercase=True,stripAccents='none',
                minimumDF=1,maximumDF=1.0,vocabularyLimit=20000,binary=False,sublinearTF=False,
                useIDF=True,smoothIDF=True,normalization='l2')


def validate_features(c):
    require(type(c) is dict and set(c)==set(feature_defaults()),'Document features require exact versioned settings.')
    require(c['analyzer'] in ('word','char') and c['stripAccents'] in ('none','ascii','unicode') and c['normalization'] in ('none','l1','l2'),'Unsupported text feature policy.')
    require(type(c['ngramMin']) is int and type(c['ngramMax']) is int and 1<=c['ngramMin']<=c['ngramMax']<=5,'N-grams must be integers from 1 to 5.')
    require(type(c['minimumDF']) is int and 1<=c['minimumDF']<=1_000_000,'Minimum document frequency must be a positive count.')
    require(type(c['maximumDF']) in (int,float) and 0<c['maximumDF']<=1,'Maximum document frequency must be a proportion in (0,1].')
    require(type(c['vocabularyLimit']) is int and 1<=c['vocabularyLimit']<=MAX_FEATURES,'Vocabulary cap exceeds its supported range.')
    require(all(type(c[k]) is bool for k in ('lowercase','binary','sublinearTF','useIDF','smoothIDF')),'Feature checkboxes require Boolean values.')
    return c


def vectorizer_options(c):
    validate_features(c)
    return dict(analyzer=c['analyzer'],ngram_range=(c['ngramMin'],c['ngramMax']),lowercase=c['lowercase'],
                strip_accents=None if c['stripAccents']=='none' else c['stripAccents'],
                min_df=c['minimumDF'],max_df=float(c['maximumDF']),max_features=c['vocabularyLimit'],
                binary=c['binary'],sublinear_tf=c['sublinearTF'],use_idf=c['useIDF'],smooth_idf=c['smoothIDF'],
                norm=None if c['normalization']=='none' else c['normalization'],dtype=np.float64)


def documents(texts,check=lambda:None):
    require(type(texts) in (list,tuple) and 1<=len(texts)<=1_000_000,'Supply a nonempty document population.')
    total=0
    for i,text in enumerate(texts):
        if i%32==0:check()
        require(type(text) is str and bool(text.strip()),'Document has no extractable text; remove it or use OCR before classification.')
        size=len(text.encode());total+=size
        require(size<=MAX_TEXT and total<=MAX_CORPUS,'Text exceeds the per-document or corpus byte budget; split the workload explicitly.')
    return texts


def terms(text,c):
    if c['lowercase']:text=text.lower()
    if c['stripAccents']!='none':
        text=unicodedata.normalize('NFKD',text)
        text=text.encode('ascii','ignore').decode() if c['stripAccents']=='ascii' else ''.join(x for x in text if not unicodedata.combining(x))
    if c['analyzer']=='word':
        pieces=re.findall(r'(?u)\b\w\w+\b',text)
        for size in range(c['ngramMin'],c['ngramMax']+1):
            for i in range(len(pieces)-size+1):yield ' '.join(pieces[i:i+size])
    else:
        text=re.sub(r'\s\s+',' ',text)
        for size in range(c['ngramMin'],min(c['ngramMax'],len(text))+1):
            for i in range(len(text)-size+1):yield text[i:i+size]


def fit_features(texts,c,check=lambda:None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    validate_features(c);documents(texts,check)
    # Admission before sklearn builds its unbounded temporary vocabulary.
    vocabulary=set();nnz=0;vocabulary_bytes=0
    for text in texts:
        check();row=set()
        for i,term in enumerate(terms(text,c)):
            if i%1024==0:check()
            require(i<MAX_TERMS,'Document token expansion exceeds the feature budget; revise analyzer/ngrams.')
            row.add(term)
            if term not in vocabulary:vocabulary.add(term);vocabulary_bytes+=len(term.encode())
            require(len(vocabulary)<=MAX_FEATURES*4 and vocabulary_bytes<=8_000_000,'Candidate vocabulary exceeds the fitting budget; reduce corpus or n-grams.')
        nnz+=len(row);require(nnz<=MAX_NNZ,'Sparse feature fitting exceeds the nonzero budget; reduce vocabulary/ngrams or workload.')
    check();vectorizer=TfidfVectorizer(**vectorizer_options(c));matrix=vectorizer.fit_transform(texts);check()
    require(matrix.nnz<=MAX_NNZ,'Sparse features exceed the nonzero budget.')
    state=dict(schemaVersion=1,kind='documentTFIDF',configuration=dict(c),vocabulary=vectorizer.get_feature_names_out().tolist(),
               idf=vectorizer.idf_.tolist() if c['useIDF'] else [1.0]*matrix.shape[1],unicodeVersion=unicodedata.unidata_version)
    return sealed(state)


def validate_processor(s):
    require(type(s) is dict and set(s)=={'schemaVersion','kind','configuration','vocabulary','idf','unicodeVersion','contentDigest'} and s['schemaVersion']==1 and s['kind']=='documentTFIDF','Unsupported fitted document processor.')
    require(s['contentDigest']==digest({k:v for k,v in s.items() if k!='contentDigest'}),'Processor content digest differs.')
    c=validate_features(s['configuration']);v=s['vocabulary'];idf=s['idf']
    require(type(v) is list and 1<=len(v)<=c['vocabularyLimit'] and all(type(t) is str and t and len(t.encode())<=MAX_TEXT for t in v) and len(set(v))==len(v),'Invalid fitted vocabulary.')
    require(v==sorted(v) and type(idf) is list and len(idf)==len(v) and all(type(x) in (int,float) and math.isfinite(x) and 0<x<=100 for x in idf),'Invalid fitted IDF weights/order.')
    require(s['unicodeVersion']==unicodedata.unidata_version,'Processor Unicode version differs; use its matching runtime.')
    return s


def transform(texts,state,check=lambda:None,*,_validated=False,_lookup=None):
    if not _validated:validate_processor(state)
    documents(texts,check);c=state['configuration'];v=_lookup if _lookup is not None else {t:i for i,t in enumerate(state['vocabulary'])}
    indices=[];values=[];offsets=[0]
    for text in texts:
        check();counts=Counter()
        for i,term in enumerate(terms(text,c)):
            if i%1024==0:check()
            require(i<MAX_TERMS,'Document token expansion exceeds the feature budget.')
            if term in v:counts[v[term]]+=1
        pairs=[]
        for index,count in sorted(counts.items()):
            value=1.0 if c['binary'] else float(count)
            if c['sublinearTF']:value=1+math.log(value)
            if c['useIDF']:value*=state['idf'][index]
            pairs.append((index,value))
        norm=(sum(abs(x) for _,x in pairs) if c['normalization']=='l1' else math.sqrt(sum(x*x for _,x in pairs))) if c['normalization']!='none' else 1
        require(len(values)+len(pairs)<=MAX_NNZ,'Sparse transformation exceeds its nonzero budget; reduce workload.')
        for index,value in pairs:indices.append(index);values.append(value/(norm or 1))
        offsets.append(len(indices))
    return sparse.csr_matrix((np.asarray(values,dtype=np.float64),np.asarray(indices,dtype=np.int32),np.asarray(offsets,dtype=np.int64)),shape=(len(texts),len(v)))


def estimator_defaults():return dict(C=1.0,maxIterations=1000,tolerance=0.0001,fitIntercept=True,classWeight='none',solver='lbfgs')


def estimator_options(c):
    require(type(c) is dict and set(c)==set(estimator_defaults()),'Classifier requires exact reviewed solver settings.')
    require(type(c['C']) in (int,float) and math.isfinite(c['C']) and 1e-8<=c['C']<=1e8,'C is outside its supported range.')
    require(type(c['tolerance']) in (int,float) and math.isfinite(c['tolerance']) and 1e-12<=c['tolerance']<=1,'Tolerance is outside its supported range.')
    require(type(c['maxIterations']) is int and 1<=c['maxIterations']<=10000 and type(c['fitIntercept']) is bool,'Invalid iterations/intercept.')
    require(c['classWeight'] in ('none','balanced') and c['solver']=='lbfgs','This adapter qualifies L2/lbfgs with none/balanced class weights.')
    return dict(C=c['C'],max_iter=c['maxIterations'],tol=c['tolerance'],fit_intercept=c['fitIntercept'],class_weight=None if c['classWeight']=='none' else 'balanced',solver='lbfgs')


def labels_valid(labels,count):
    require(type(labels) is list and len(labels)==count and all(type(x) is str and x.strip() and len(x.encode())<=1024 for x in labels),'Supply one nonblank category per document.')


def fit(texts,labels,feature_settings,configuration,*,processor=None,check=lambda:None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import sklearn
    require(sklearn.__version__=='1.7.2','Document classifier requires reviewed scikit-learn1.7.2.')
    documents(texts,check);labels_valid(labels,len(texts));classes=sorted(set(labels))
    require(2<=len(classes)<=MAX_CLASSES,'Fitting needs at least two categories within the class budget.')
    p=fit_features(texts,feature_settings,check) if processor is None else validate_processor(processor)
    require(p['configuration']==feature_settings,'Feature settings differ from the selected processor.')
    require(len(classes)*len(p['vocabulary'])<=MAX_COEFFICIENTS,'Class count × feature width exceeds the coefficient budget.')
    x=transform(texts,p,check)
    require(all(np.diff(x.indptr)>0),'Training document has no features after filtering; revise text/feature settings.')
    check()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always',ConvergenceWarning)
        model=LogisticRegression(**estimator_options(configuration)).fit(x,labels)
    check()
    return sealed(dict(schemaVersion=1,stateFormat=FORMAT,runtimeVersion='1.7.2',task='singleLabelDocumentClassification',
        processor=p,classes=model.classes_.tolist(),coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),
        configuration=dict(configuration),iterations=model.n_iter_.tolist(),converged=not any(issubclass(w.category,ConvergenceWarning) for w in recorded),
        trainingCounts=[labels.count(x) for x in model.classes_]))


def validate_model(m):
    require(type(m) is dict and set(m)=={'schemaVersion','stateFormat','runtimeVersion','task','processor','classes','coefficients','intercept','configuration','iterations','converged','trainingCounts','contentDigest'},'Invalid classifier state fields.')
    require(m['schemaVersion']==1 and m['stateFormat']==FORMAT and m['runtimeVersion']=='1.7.2' and m['task']=='singleLabelDocumentClassification','Unsupported classifier package.')
    require(m['contentDigest']==digest({k:v for k,v in m.items() if k!='contentDigest'}),'Classifier content digest differs.')
    p=validate_processor(m['processor']);estimator_options(m['configuration']);classes=m['classes']
    require(type(classes) is list and 2<=len(classes)<=MAX_CLASSES and len(set(classes))==len(classes) and classes==sorted(classes),'Invalid category order.')
    labels_valid(classes,len(classes));n=len(classes) if len(classes)>2 else 1;width=len(p['vocabulary'])
    require(n*width<=MAX_COEFFICIENTS and type(m['coefficients']) is list and len(m['coefficients'])==n and all(type(r) is list and len(r)==width and all(type(x) in (int,float) and math.isfinite(x) for x in r) for r in m['coefficients']),'Invalid classifier coefficients.')
    require(type(m['intercept']) is list and len(m['intercept'])==n and all(type(x) in (int,float) and math.isfinite(x) for x in m['intercept']),'Invalid classifier intercept.')
    require(type(m['converged']) is bool and type(m['iterations']) is list and m['iterations'] and all(type(x) is int and 0<=x<=10000 for x in m['iterations']),'Invalid solver receipt.')
    require(type(m['trainingCounts']) is list and len(m['trainingCounts'])==len(classes) and all(type(x) is int and 0<x<=1_000_000 for x in m['trainingCounts']),'Invalid training class counts.')
    return m


def predict(m,texts,check=lambda:None):
    validate_model(m);x=transform(texts,m['processor'],check)
    require(len(texts)*len(m['classes'])<=MAX_COEFFICIENTS,'Prediction probabilities exceed the result budget; reduce workload.')
    scores=np.asarray(x @ np.asarray(m['coefficients']).T)+np.asarray(m['intercept'])
    require(np.isfinite(scores).all(),'Classifier scores are non-finite.')
    if len(m['classes'])==2:
        z=scores[:,0];prob=np.empty(len(z));positive=z>=0
        prob[positive]=1/(1+np.exp(-z[positive]));exp=np.exp(z[~positive]);prob[~positive]=exp/(1+exp)
        probabilities=np.column_stack((1-prob,prob));chosen=(z>0).astype(int)
    else:
        p=np.exp(scores-scores.max(axis=1,keepdims=True));probabilities=p/p.sum(axis=1,keepdims=True);chosen=scores.argmax(axis=1)
    check()
    return dict(classes=[m['classes'][i] for i in chosen],classOrder=list(m['classes']),probabilities=probabilities.tolist(),
                noFeatures=(np.diff(x.indptr)==0).tolist())


def evaluate(m,texts,labels,check=lambda:None):
    labels_valid(labels,len(texts));p=predict(m,texts,check);classes=m['classes'];require(set(labels)<=set(classes),'Evaluation contains unseen categories; do not refit on held-out data.')
    index={v:i for i,v in enumerate(classes)};matrix=[[0]*len(classes) for _ in classes]
    for actual,predicted in zip(labels,p['classes']):matrix[index[actual]][index[predicted]]+=1
    rows=[]
    for i,label in enumerate(classes):
        tp=matrix[i][i];support=sum(matrix[i]);predicted=sum(r[i] for r in matrix);precision=tp/predicted if predicted else 0;recall=tp/support if support else 0
        rows.append(dict(category=label,precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0,support=support))
    n=len(labels);present=[r for r in rows if r['support']];majority=classes[int(np.argmax(m['trainingCounts']))]
    return dict(count=n,accuracy=sum(matrix[i][i] for i in range(len(classes)))/n,balancedAccuracy=sum(r['recall'] for r in present)/len(present),
                macroF1=sum(r['f1'] for r in rows)/len(rows),weightedF1=sum(r['f1']*r['support'] for r in rows)/n,
                logLoss=-sum(math.log(max(1e-15,p['probabilities'][i][index[v]])) for i,v in enumerate(labels))/n,
                classOrder=classes,confusionMatrix=matrix,perClass=rows,majorityBaseline=sum(v==majority for v in labels)/n,
                missingCategories=[r['category'] for r in rows if not r['support']],zeroDivision=0)


def load(raw):
    require(type(raw) is bytes and len(raw)<=MAX_STATE,'Classifier package state exceeds its byte allowance.')
    def unique(pairs):
        out={}
        for k,v in pairs:require(k not in out,'Duplicate classifier field.');out[k]=v
        return out
    try:m=json.loads(raw,object_pairs_hook=unique)
    except (json.JSONDecodeError,UnicodeDecodeError,RecursionError) as e:raise ValueError('Malformed classifier state.') from e
    return validate_model(m)


def export(m,destination):
    validate_model(m);raw=canonical(m);require(len(raw)<=MAX_STATE,'Classifier state exceeds the package byte allowance.')
    dest=Path(destination);require(not dest.exists() and not dest.is_symlink(),'Choose an unused classifier package destination.')
    stage=Path(tempfile.mkdtemp(prefix='.document-classifier-',dir=dest.parent))
    try:
        (stage/'state.json').write_bytes(raw)
        shutil.copyfile(Path(__file__),stage/'document_classification.py')
        (stage/'inference.py').write_text("import json,sys\nfrom pathlib import Path\nfrom document_classification import load,predict\ndef bounded(path,limit):\n    with path.open('rb') as stream: raw=stream.read(limit+1)\n    if len(raw)>limit: raise ValueError('Input exceeds its byte budget')\n    return raw\nbase=Path(__file__).resolve().parent\nprint(json.dumps(predict(load(bounded(base/'state.json',32000000)),json.loads(bounded(Path(sys.argv[1]),64000000))),allow_nan=False))\n")
        (stage/'README.txt').write_text('Local single-label document inference. Requires NumPy and SciPy; does not import scikit-learn for prediction. Run python inference.py documents.json, a JSON array of extracted text. No optimizer continuation, OCR or calibration is implied. Keep the complete package together. Training source documents are not included.\n')
        inventory=[dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(stage.iterdir())]
        (stage/'manifest.json').write_bytes(canonical(dict(schemaVersion=1,stateFormat=FORMAT,artifacts=inventory)))
        # Exclusive directory reservation; never replace an existing user's package.
        dest.mkdir(mode=0o700);identity=(dest.stat().st_dev,dest.stat().st_ino)
        descriptor=os.open(dest,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            require((os.fstat(descriptor).st_dev,os.fstat(descriptor).st_ino)==identity,'Package destination identity changed.')
            for child in sorted(stage.iterdir(),key=lambda p:p.name=='manifest.json'):
                os.rename(child,child.name,dst_dir_fd=descriptor)
        except BaseException:
            try:
                actual=dest.lstat()
                if (actual.st_dev,actual.st_ino)==identity:shutil.rmtree(dest)
            except FileNotFoundError:pass
            raise
        finally:os.close(descriptor)
    finally:shutil.rmtree(stage)
