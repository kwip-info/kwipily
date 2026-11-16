"""Discovery must expose the literal producer, not inferred component metadata."""
from runner.program_export_configuration import capability,DEFAULTS
from runner.engine import capabilities


def test_exact_program_capability_producer_and_discovery():
    expected={'schemaVersion':1,'version':'1.0.0','adapterID':'kwip.runner.program_export','recipeSchemaVersion':1,'configurationVersion':1,
              'modes':['program-export','shape-constraints','program-reload-execution'],'devices':['cpu','mps'],
              'execution':{'evaluationOnly':True,'singleTensorInputOutput':True,'leadingBatch':['static','bounded'],'fixedSampleAxes':True,'devicePolicy':'explicit-no-fallback','bagPacket':'refused'},
              'trust':'trusted-app-generated-only'}
    produced=capability()
    assert {k:produced[k] for k in expected}==expected
    assert set(produced)==set(expected)|{'controls','budgets','nativeAPIs'}
    assert set(produced['controls'])==set(DEFAULTS)
    assert {k:v['default'] for k,v in produced['controls'].items()}==DEFAULTS
    assert produced['budgets']=={'parameterElements':2000000,'bufferElements':2000000,'constantElements':2000000,'inputElements':2000000,'outputElements':2000000,'tensorRank':8,'sampleAxis':4096,'graphNodes':4096,'programBytes':100000000,'manifestBytes':2000000,'archiveEntries':512,'archiveUncompressedBytes':100000000}
    assert set(produced['nativeAPIs'])=={'export','save','load'}
    assert capabilities().get('programExportCapabilities')=={'program_export':produced}
