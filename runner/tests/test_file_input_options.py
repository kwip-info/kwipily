import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]

def api():
    from runner import file_input_options
    return file_input_options


def test_defaults_are_complete_isolated_and_source_metadata_is_mirrored():
    module=api();first=module.defaults();second=module.defaults()
    assert module.validate_configuration({})==first==second
    assert first['maxFiles']==1000 and first['maxDepth']==32
    assert first['maxFileBytes']==67108864 and first['maxTotalBytes']==268435456
    assert first['sourceMode']=='file' and first['metadataFields'][0]=='file_id'
    first['readers']['image']['frames']='all'
    assert second['readers']['image']['frames']=='first'
    assert (ROOT/'runner/file-input-options.json').read_bytes()==(ROOT/'Sources/KwipMLCore/Resources/file-input-options.json').read_bytes()


@pytest.mark.parametrize('config',[
 {'path':'/private/file'}, {'sourceMode':'url'}, {'includeSubfolders':1}, {'maxFiles':True},
 {'maxFiles':0}, {'maxFiles':1001}, {'maxDepth':33}, {'maxFileBytes':67108865},
 {'maxTotalBytes':268435457}, {'maxFileBytes':20,'maxTotalBytes':10},
 {'extensions':['bad.ext']}, {'extensions':['txt','txt']}, {'extensions':['TXT']},
 {'metadataFields':['path']}, {'metadataFields':['file_id','file_id']},
 {'includePatterns':['/absolute/*']}, {'includePatterns':['../*']}, {'includePatterns':['a/../b']},
 {'includePatterns':['https://example.com/*']}, {'excludePatterns':['a\x00b']},
 {'readers':{'unknown':{}}}, {'readers':{'image':{'path':'secret'}}},
 {'readers':{'image':{'frames':'range','frameStart':2,'frameEnd':1}}},
 {'readers':{'pdf':{'orientations':[45]}}}, {'readers':{'pdf':{'orientations':[]}}},
 {'readers':{'pdf':{'pages':[0]}}}, {'readers':{'pdf':{'pages':[1,1]}}},
 {'readers':{'pdf':{'space_width':float('nan')}}}, {'readers':{'pdf':{'space_width':10**1000}}}, {'readers':{'image':{'frameStart':True}}},
])
def test_invalid_or_private_configuration_is_refused(config):
    with pytest.raises(ValueError):api().validate_configuration(config)


@pytest.mark.parametrize('key,choices',[
 ('sourceMode',['file','directory']),('symlinks',['skip','error']),('readMode',['content','metadata']),
 ('errorPolicy',['fail','record','skip']),('unknownFiles',['metadata','error','skip'])])
def test_top_choices_are_typed_and_round_trip(key,choices):
    for choice in choices: assert api().validate_configuration({key:choice})[key]==choice


def test_every_declared_reader_option_default_and_choice_validates():
    spec=json.loads((ROOT/'runner/file-input-options.json').read_text())
    for reader,definition in spec['readers'].items():
        assert api().get_reader(reader)==definition
        for field in definition['fields']:
            values=[field['defaultValue']]+[choice['value'] for choice in field.get('choices',[])]
            for value in values:
                result=api().validate_configuration({'readers':{reader:{field['key']:value}}})
                assert result['readers'][reader][field['key']]==value


def test_valid_ranges_pages_orientations_and_metadata_projection():
    result=api().validate_configuration({'extensions':['png','pdf'],'includePatterns':['*.pdf','**/*.png'],
      'metadataFields':['file_id','content_sha256'], 'readers':{'image':{'frames':'range','frameStart':1,'frameEnd':3},
      'pdf':{'pages':[2,1],'orientations':[90,0],'extraction_mode':'layout'}}})
    assert result['readers']['image']['frameEnd']==3
    assert result['readers']['pdf']['pages']==[2,1]
    assert result['readers']['pdf']['orientations']==[90,0]


def test_extensions_allow_explicit_all_and_custom_unknown_formats():
    assert api().validate_configuration({'extensions':[]})['extensions']==[]
    assert api().validate_configuration({'extensions':['custom','jpg']})['extensions']==['custom','jpg']


def test_options_admission_imports_only_the_standard_library():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c',"import sys; from runner.file_input_options import validate_configuration; validate_configuration({}); assert not any(x in sys.modules for x in ['PIL','pypdf','docx','polars','numpy'])"],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('mode',['content','metadata'])
def test_file_identity_is_required_even_when_other_metadata_is_disabled(mode):
    for fields in ([],['name']):
        with pytest.raises(ValueError):api().validate_configuration({'readMode':mode,'metadataFields':fields})
    assert api().validate_configuration({'readMode':mode,'metadataFields':['file_id']})['metadataFields']==['file_id']
    assert api().validate_configuration({'metadataFields':['name','file_id']})['metadataFields']==['name','file_id']
