"""Artifact-action authoring cannot masquerade as an ordinary forward operator."""
import json
from pathlib import Path
import pytest
from scripts import component_bundles as bundles
from runner.program_export_configuration import capability
ROOT=Path(__file__).resolve().parents[2]
def authored():return json.loads((ROOT/'components/program-export/bundle.json').read_text())

def test_actual_program_action_contract_seven_fields_and_two_assessed_lessons():
    value=authored();bundles.validate(value,ROOT)
    row=value['capabilities'][0]
    assert row['runtimeContract']==capability()
    assert len(row['configurationFields'])==7 and len(row['modes'])==3
    assert len(row['learning']['lessonIDs'])==2
    assert 'definition' not in row and row['bindings'][0]['surface']=='history'

@pytest.mark.parametrize('mutation',[
    lambda v:v['capabilities'][0].update(definition={'id':'program_export'}),
    lambda v:v['capabilities'][0]['configurationFields'][2].update(max=4096),
    lambda v:v['capabilities'][0]['runtimeContract']['execution'].update(evaluationOnly=1),
    lambda v:v['capabilities'][0]['runtimeContract']['execution'].update(bagPacket='supported'),
    lambda v:v['capabilities'][0]['bindings'][0].update(surface='canvas'),
    lambda v:v['capabilities'][0]['learning']['lessonIDs'].pop(),
    lambda v:v['capabilities'][0]['sources'].update(native='../outside'),
])
def test_program_action_unreviewed_forward_controls_limits_privacy_binding_refused(mutation):
    value=authored();mutation(value)
    with pytest.raises(ValueError):bundles.validate(value,ROOT)

def test_program_generation_reuses_artifact_metadata_without_registry_forward_entry():
    first=bundles.render(ROOT);assert first==bundles.render(ROOT)
    swift=first['Sources/KwipMLCore/BundledProgramExportCapabilities.generated.swift']
    assert 'public enum BundledProgramExportCapabilities' in swift
    assert 'public static let entry' in swift and 'configurationFields: [ComponentField]' in swift
    assert 'case exportProgramPackage' in swift
    registry=json.loads(first['Sources/KwipMLCore/Resources/components.json'])
    assert 'program_export' not in {r['id'] for r in registry['components']}
    assert 'RESOLVERS' not in first['runner/program_export_capabilities_generated.py']

def test_published_program_action_requires_registered_generated_artwork(monkeypatch):
    value=authored();value['capabilities'][0]['published']=True
    real=Path.is_file
    monkeypatch.setattr(Path,'is_file',lambda p:False if p.name=='program-export-v1.png' and 'ComponentIcons' in p.parts else real(p))
    with pytest.raises(ValueError,match='artwork|registered'):bundles.validate(value,ROOT)
