from scripts.tests.registry_history import historical_output_presentation
"""Candidate publication must expose real reviewed deliverables without forward fiction."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_pair_published_with_actual_assessed_sources_and_original_assets():
    rows=json.loads((ROOT/'components/gradient_controls/bundle.json').read_text())['components']
    expected={'gradient_policy':['gradient-selected-influence','gradient-combine-small-groups'],'trainability':['trainability-keep-settings','trainability-next-run']}
    for row in rows:
        ident=row['definition']['id'];assert row['published'] is True
        learning=row['learning'];assert learning['lessonIDs']==['assessment.'+x for x in expected[ident]]
        assert learning['exampleIDs']==['prepared-'+x for x in expected[ident]]
        for path in learning['sources']:assert (ROOT/path).is_file(),path
        assert row['definition']['inputs']==[] and row['definition']['outputs']==['out']
        assert learning['paragraphs'] and all(isinstance(x,str) and x.strip() for x in learning['paragraphs'])

def test_candidate_registry_appends_only_pair_preserving_prior84_definition_bytes():
    import subprocess
    baseline=json.loads((ROOT / 'Tests/historical/d13a57e/Sources/KwipMLCore/Resources/components.json').read_bytes())['components']
    current=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())['components']
    assert len(baseline)==84
    current=[r for r in current if r['id'] not in ('bag_input','embedding_bag','dataset_source','data_loader','rms_norm','rotary_position','mixture_experts')]
    assert historical_output_presentation(current[:84])==baseline
    assert [r['id'] for r in current[84:]]==['gradient_policy','trainability']
