"""Bounded CSV source adapter; schema mapping stays separate from the model graph."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import torch

MAX_BYTES=10_000_000
MAX_ROWS=100_000

def _preview_cell(value):
    if value is None:return ''
    if isinstance(value,str):return value[:200]
    if type(value) in (bool,int,float):return json.dumps(value,allow_nan=False,separators=(',',':'))[:200]
    raise ValueError('Table Input inspection supports scalar cells only')

def read_csv(path,separator=",",*,options=None,check_cancelled=lambda:None):
    if __package__:from .csv_parser import parse_csv
    else:from csv_parser import parse_csv
    # Legacy direct training kept blank-skipping and empty strings; preparation uses
    # the sourceCSV defaults. Explicit options unify new source inspection/execution.
    selected={'skipBlankRecords':True,'emptyField':'text'} if options is None else options
    header,rows,source,_=parse_csv(path,selected,separator,check_cancelled=check_cancelled,max_bytes=MAX_BYTES,max_rows=MAX_ROWS)
    return header,[list(row) for row in rows],source

def inspect_data(path,separator=",",*,options=None,table_options=None,check_cancelled=lambda:None):
    if table_options is not None:
        if options is not None:raise ValueError('CSV and Table Input inspection settings cannot be combined')
        if __package__:
            from .source_inspection_operation_contract import normalize_table_options
            from .table_sources import preview_file
        else:
            from source_inspection_operation_contract import normalize_table_options
            from table_sources import preview_file
        check_cancelled()
        selected=normalize_table_options(table_options)
        if selected.get('parserProfile')=='polars':
            return inspect_data(path,options=selected,check_cancelled=check_cancelled)
        if selected['format']=='api':
            if __package__:from .api_table_snapshot import preview_evidence
            else:from api_table_snapshot import preview_evidence
            sample=preview_evidence(path,selected,check_cancelled=check_cancelled)
        elif selected['format']=='database':
            if __package__:from .database_source import preview_evidence
            else:from database_source import preview_evidence
            sample=preview_evidence(path,selected,check_cancelled=check_cancelled)
        else:sample=preview_file(path,selected,check_cancelled=check_cancelled)
        header=[column['name'] for column in sample['columns']]
        rows=sample['rows'];source_hash=sample['sourceSHA256']
        check_cancelled()
        if len(rows)>8:raise ValueError('Table Input inspection exceeds its eight-row sample budget')
        return {**({'apiFetch':sample['apiFetch']} if sample.get('apiFetch') is not None else {}),'event':'dataPreview','columns':header,'rows':[[ _preview_cell(value) for value in row] for row in rows],
            'columnSchema':[{'name':column['name'],'dtype':column['dtype'],'nullable':None} for column in sample['columns']],
            'rowCount':len(rows),'sourceSHA256':source_hash,
            'parsing':{'options':table_options,'validatedRows':len(rows),'previewRows':len(rows),'sampleOnly':True},
            'nullCells':[[i,j] for i,row in enumerate(rows) for j,value in enumerate(row) if value is None]}
    native=isinstance(options,dict) and options.get('parserProfile')=='polars'
    column_schema=None
    if native:
        if __package__:from .polars_csv import preview_csv
        else:from polars_csv import preview_csv
        header,rows,source_hash,column_schema=preview_csv(path,options,check_cancelled=check_cancelled,include_source_hash=True,include_column_schema=True)
    elif options is not None:
        if __package__:from .csv_parser import parse_csv
        else:from csv_parser import parse_csv
        header,rows,source_hash,_=parse_csv(path,options,separator,check_cancelled=check_cancelled,preview=True)
    else:
        header,rows,source_hash=read_csv(path,separator,options=options,check_cancelled=check_cancelled)
    result={'event':'dataPreview','columns':header,'rows':[[_preview_cell(value) for value in row] for row in rows[:8]],'rowCount':len(rows),'sourceSHA256':source_hash}
    if column_schema is not None:result['columnSchema']=column_schema
    if options is not None:
        if __package__:from .csv_parser import normalize_options
        else:from csv_parser import normalize_options
        if native:
            if __package__:from .polars_csv_configuration import normalize_options as native_options
            else:from polars_csv_configuration import normalize_options as native_options
        result['parsing']={'options':native_options(options) if native else normalize_options(options,separator),'validatedRows':len(rows),'previewRows':min(8,len(rows)),'sampleOnly':True}
        result['nullCells']=[[i,j] for i,row in enumerate(rows[:8]) for j,v in enumerate(row) if v is None]
    return result

def csv_dataset(path,feature_columns,target_column,seed=42,sample_limit=0,validation_percent=20):
    header,rows,digest=read_csv(path)
    if not feature_columns or len(feature_columns)!=2 or len(set(feature_columns))!=2:
        raise ValueError('Select exactly two different numeric feature columns for this POC.')
    if target_column in feature_columns: raise ValueError('Target cannot also be a feature; that would leak the answer.')
    if any(name not in header for name in [*feature_columns,target_column]): raise ValueError('Selected feature or target column is absent from the CSV header.')
    fi=[header.index(c) for c in feature_columns]; ti=header.index(target_column); x=[]; y=[]
    seen={}
    for number,row in enumerate(rows,2):
        try: features=[float(row[i]) for i in fi]; label=float(row[ti])
        except ValueError as error: raise ValueError(f'CSV row {number}: features and target must be numeric.') from error
        if not all(math.isfinite(v) and abs(v)<=1e10 for v in features): raise ValueError(f'CSV row {number}: features must be finite and within ±1e10.')
        if not math.isfinite(label) or label not in (0,1): raise ValueError(f'CSV row {number}: target must be integer 0 or 1.')
        # Duplicate observations cannot straddle partitions. Reject ambiguous duplicates
        # rather than pretending row-level splitting establishes an independent holdout.
        key=tuple(features)
        if key in seen: raise ValueError(f'CSV row {number}: duplicate selected features; deduplicate observations before splitting to avoid holdout leakage.')
        seen[key]=number; x.append(features); y.append(int(label))
    x=torch.tensor(x,dtype=torch.float32); y=torch.tensor(y,dtype=torch.long)
    if not torch.isfinite(x).all(): raise ValueError('Features exceed float32 limits.')
    if len(torch.unique(x,dim=0))!=len(x): raise ValueError('Selected features become duplicates at float32 precision; deduplicate before splitting.')
    data,selection=select_split(x,y,seed,sample_limit,validation_percent,row_offset=0)
    selection['splitHash']=hashlib.sha256(json.dumps({'sourceHash':digest,'featureColumns':feature_columns,'targetColumn':target_column,'partitionHash':selection['splitHash']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    provenance={'dataset':'local-csv-v1','sourceName':Path(path).name,'sourceHash':digest,'featureColumns':feature_columns,'targetColumn':target_column,'preprocessing':'none',**selection}
    return data,provenance


def select_split(x,y,split_seed=42,sample_limit=0,validation_percent=20,row_offset=0):
    if type(split_seed) is not int or not 0<=split_seed<=2**32-1: raise ValueError('Split seed must be an integer from 0 to 4294967295.')
    if type(sample_limit) is not int or (sample_limit!=0 and not 20<=sample_limit<=100000): raise ValueError('Sample limit must be 0 (all), or an integer from 20 to 100000.')
    if type(validation_percent) is not int or not 5<=validation_percent<=50: raise ValueError('Validation percent must be an integer from 5 to 50.')
    if sample_limit>len(x): raise ValueError(f'Sample limit {sample_limit} exceeds the {len(x)} available observations.')
    generator=torch.Generator().manual_seed(split_seed)
    groups=[(y==label).nonzero().flatten() for label in (0,1)]
    if any(len(indices)<10 for indices in groups): raise ValueError('Training requires at least 10 distinct observations in each target class.')
    count=sample_limit or len(x)
    first=max(10,min(len(groups[0]),count-10,round(count*len(groups[0])/len(x))))
    first=max(first,count-len(groups[1]))
    counts=[first,count-first]
    selected=[]; train=[]; validation=[]
    for indices,take in zip(groups,counts):
        indices=indices[torch.randperm(len(indices),generator=generator)][:take]
        selected.append(indices)
        # Random subsample first, then independently shuffle that subset for the split.
        indices=indices[torch.randperm(len(indices),generator=generator)]
        holdout=max(1,min(len(indices)-1,round(len(indices)*validation_percent/100)))
        validation.append(indices[:holdout]); train.append(indices[holdout:])
    train=torch.cat(train); validation=torch.cat(validation); selected=torch.cat(selected)
    train=train[torch.randperm(len(train),generator=generator)]; validation=validation[torch.randperm(len(validation),generator=generator)]
    selected_ids=sorted((selected+row_offset).tolist()); train_ids=(train+row_offset).tolist(); validation_ids=(validation+row_offset).tolist()
    split_hash=hashlib.sha256(json.dumps({'train':train_ids,'validation':validation_ids},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    provenance={'sourceSamples':len(x),'samples':len(selected),'trainingSamples':len(train),'validationSamples':len(validation),'splitSeed':split_seed,'sampleLimit':sample_limit,'validationPercent':validation_percent,'rowIDDefinition':'zero-based parsed observation index; header and empty records excluded','selectedRowIDs':selected_ids,'trainingRowIDs':train_ids,'validationRowIDs':validation_ids,'splitHash':split_hash,'split':'seeded stratified selection then stratified holdout; disjoint observation IDs'}
    return (x[train],y[train],x[validation],y[validation]),provenance
