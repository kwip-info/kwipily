import hashlib
import pytest
from runner.polars_csv import source_csv,preview_csv
from runner.polars_csv_configuration import normalize_options


def source(tmp_path,text):
    path=tmp_path/'input.csv';path.write_text(text);return path


def test_inference_projection_rename_index_and_full_source_pin(tmp_path):
    path=source(tmp_path,'a,b,c\n1,true,1.5\n2,false,2.5\n3,true,3.5\n')
    config={'inferSchema':True,'newColumns':['number','flag','amount'],'rowIndexName':'record','rowIndexOffset':12,'columns':['record','amount','flag'],'nRows':2}
    table=source_csv(path,config,sampling={'mode':'rowCount','rowCount':1})
    assert [c['dtype'] for c in table.columns]==['int64','float64','bool']
    assert list(table.rows)==[(12,1.5,True)]
    assert table.source_digest==hashlib.sha256(path.read_bytes()).hexdigest()
    assert list(table.row_ids)==[table.source_digest+':0']


def test_explicit_schema_maps_by_names_with_sorted_transport(tmp_path):
    path=source(tmp_path,'z,a\n4,true\n5,false\n')
    table=source_csv(path,{'schema':{'a':'Boolean','z':'Int64'}})
    assert [c['name'] for c in table.columns]==['a','z']
    assert list(table.rows)==[(True,4),(False,5)]


def test_decimal_nulls_errors_and_preview_schema(tmp_path):
    path=source(tmp_path,'id;amount;flag\n1;1,25;true\n2;N/A;false\n3;broken;true\n')
    config={'separator':';','decimalComma':True,'schemaOverrides':{'id':'Int64','amount':'Float64','flag':'Boolean'},'columnNullValues':{'amount':'N/A'},'ignoreErrors':True}
    header,rows,pin,columns=preview_csv(path,config,include_source_hash=True,include_column_schema=True)
    assert header==['id','amount','flag'] and rows==[[1,1.25,True],[2,None,False],[3,None,True]]
    assert pin==hashlib.sha256(path.read_bytes()).hexdigest()
    assert [c['dtype'] for c in columns]==['int64','float64','bool']
    assert all(c['nullable'] is None for c in columns)
    assert list(source_csv(path,config).rows)==[tuple(row) for row in rows]
    with pytest.raises(ValueError):source_csv(path,{**config,'ignoreErrors':False})


def test_custom_record_ending_projection_and_row_limit(tmp_path):
    path=source(tmp_path,'a;b|1;2|3;4|5;6|')
    config={'separator':';','eolChar':'|','columns':[1],'nRows':2}
    assert list(source_csv(path,config).rows)==[('2',),('4',)]
    assert preview_csv(path,config)==(['b'],[['2'],['4']])


def test_empty_header_table_allowed_explicitly_and_missing_schema_insert(tmp_path):
    path=source(tmp_path,'a,b\n')
    assert list(source_csv(path,{'raiseIfEmpty':False}).rows)==[]
    assert preview_csv(path,{'raiseIfEmpty':False})==(['a','b'],[])
    with pytest.raises(ValueError):source_csv(path)
    path.write_text('a\n1\n')
    table=source_csv(path,{'schema':{'a':'Int64','missing':'String'},'missingColumns':'insert','emptyField':'null'})
    assert list(table.rows)==[(1,None)]


@pytest.mark.parametrize('options',[
 {'columns':['a',1]}, {'columns':[True]}, {'columns':['a','a']}, {'schema':{'a':'Date'}},
 {'schema':{'a':'Int64'},'schemaOverrides':{'a':'Int64'}}, {'columnNullValues':{'a':'NA'},'nullMarkers':['NA']},
 {'nRows':True}, {'inferSchemaLength':5_000_001}, {'decimalComma':True}, {'rowIndexOffset':1},
 {'eolChar':','}, {'usePyarrow':True}, {'newColumns':['a','a']}, {'extraColumns':'truncate'}])
def test_invalid_or_incompatible_options_refused(options):
    with pytest.raises(ValueError):normalize_options(options)


def test_date_inference_refuses_unsupported_temporal_values(tmp_path):
    path=source(tmp_path,'day\n2026-10-13\n')
    with pytest.raises(ValueError):source_csv(path,{'inferSchema':True,'tryParseDates':True})
    with pytest.raises(ValueError):preview_csv(path,{'inferSchema':True,'tryParseDates':True})


def test_ragged_extra_fields_are_only_ignored_when_selected(tmp_path):
    path=source(tmp_path,'a,b\n1,2,extra\n3,4,other\n')
    with pytest.raises(ValueError):source_csv(path)
    assert list(source_csv(path,{'extraColumns':'ignore'}).rows)==[('1','2'),('3','4')]
    assert preview_csv(path,{'extraColumns':'ignore'})==(['a','b'],[['1','2'],['3','4']])


def test_numeric_width_normalization_and_unsigned_overflow_refusal(tmp_path):
    path=source(tmp_path,'small,big,amount\n2,4,1.5\n')
    table=source_csv(path,{'schemaOverrides':{'small':'UInt8','big':'Int32','amount':'Float32'}})
    assert [c['dtype'] for c in table.columns]==['int64','int64','float64']
    assert list(table.rows)==[(2,4,1.5)]
    path.write_text('big\n18446744073709551615\n')
    with pytest.raises((ValueError,OverflowError)):source_csv(path,{'schemaOverrides':{'big':'UInt64'}})
    with pytest.raises(ValueError):preview_csv(path,{'schemaOverrides':{'big':'UInt64'}})


def test_schema_map_order_has_no_execution_semantics(tmp_path):
    path=source(tmp_path,'z,a\n4,true\n')
    left={'schema':{'z':'Int64','a':'Boolean'}};right={'schema':{'a':'Boolean','z':'Int64'}}
    assert normalize_options(left)==normalize_options(right)
    assert source_csv(path,left).columns==source_csv(path,right).columns
    assert list(source_csv(path,left).rows)==list(source_csv(path,right).rows)
