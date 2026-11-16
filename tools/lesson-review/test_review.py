import http.client
import importlib.util
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('lesson_review', Path(__file__).with_name('server.py'))
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'source'
        self.state = Path(self.temp.name) / 'state'
        self.source = self.root / review.CATALOG.relative_to(review.ROOT)
        self.source.parent.mkdir(parents=True)
        self.source.write_text(review.CATALOG.read_text())
        for source in review.SOURCE_INPUTS:
            target=self.root/source.relative_to(review.ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(source.read_text())
        self.packet = review.packet(self.root)
        self.request = dict(action='approve', note='Reviewed in local viewer.', revision=self.packet['revision'], sourceHash=self.packet['current']['sourceHash'])

    def test_exact_canonical_steps_and_reference(self):
        self.assertEqual([s['id'] for s in self.packet['steps']], [s['id'] for s in self.packet['current']['steps']])
        self.assertIn('12 rows', review.reference())
        self.assertEqual(len(self.packet['steps']),10)
        self.assertEqual(self.packet['assessmentID'],'assessment.activation-relu')
        self.assertEqual([s['id'] for s in self.packet['current']['steps'][:3]],['goal','try.0','inspect'])
        self.assertEqual(self.packet['publicExamples'],self.packet['current']['publicExamples'])
        self.assertEqual(self.packet['publicExamples'],[dict(input=[-2,0],output=[0,0]),dict(input=[1,2],output=[1,2])])
        self.assertIn('adapter also requires a Linear',self.packet['current']['steps'][0]['text'])
        self.assertIn('no settings or trainable',self.packet['steps'][1]['text'])
        self.assertNotIn('taskSeed',self.packet)
        self.assertNotIn('recoveredWeights',self.packet)

    def test_decision_captures_exact_proposal_without_source_mutation(self):
        before = self.source.read_bytes()
        result = review.decision(self.request, self.root, self.state)
        self.assertEqual(result['proposal']['steps'], self.packet['steps'])
        self.assertEqual(self.source.read_bytes(), before)
        saved = json.loads((self.state / 'affine-relu-decision.json').read_text())
        self.assertEqual(saved['action'], 'approve')
        self.assertIn('native chapter checks pending', saved['scope'])

    def test_repeated_approval_is_idempotent_and_changed_decision_archives(self):
        first = review.decision(self.request, self.root, self.state)
        second = review.decision(self.request, self.root, self.state)
        self.assertEqual(first, second)
        review.decision({**self.request, 'action':'request_changes', 'note':'Explain batch first.'}, self.root, self.state)
        self.assertEqual(len(list(self.state.glob('affine-relu-decision-*.json'))), 1)

    def test_stale_proposal_and_changed_source_refused(self):
        for key in ('revision', 'sourceHash'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                review.decision({**self.request, key:'old'}, self.root, self.state)
        reference=self.root/review.RELU_REFERENCE.relative_to(review.ROOT)
        reference.write_text(reference.read_text().replace('Build a flow that','Build another flow that'))
        with self.assertRaisesRegex(ValueError, 'Current lesson changed'):
            review.packet(self.root)
        self.assertFalse(self.state.exists())

    def test_notes_required_for_changes_and_bounded(self):
        for request in ({**self.request,'action':'request_changes','note':' '}, {**self.request,'note':'x'*8001}, {**self.request,'action':'run'}):
            with self.subTest(action=request['action']), self.assertRaises(ValueError):
                review.decision(request, self.root, self.state)
        self.assertFalse(self.state.exists())

    def test_approved_input_copy_is_shared_and_matches_signed_off_packet(self):
        approved=json.loads((review.HERE/'proposals/model-input.json').read_text())
        applied=review.read_current(self.root,'model-input')
        self.assertEqual(applied['steps'],approved['steps'])
        catalog=review.CATALOG.read_text()
        self.assertIn('deck(ModelInputLessonContent.steps.map',catalog)
        view=(review.ROOT/'Sources/KwipMLApp/ModelInputLearningView.swift').read_text()
        self.assertIn('ModelInputLessonContent.steps.prefix(2)',view)
        self.assertIn('ModelInputLessonContent.steps.suffix(2)',view)

    def test_approved_guided_copy_matches_and_preserves_legacy_source(self):
        approved=json.loads((review.HERE/'proposals/guided.json').read_text())
        applied=review.read_current(self.root,'guided')
        self.assertEqual(applied['steps'],approved['steps'])
        guide=review.GUIDE.read_text()
        self.assertIn('if version>=5,let current=GuidedLessonCopy.stages[nodeID]',guide)
        self.assertIn('if version<5 && nodeID=="mapping"',guide)
        self.assertIn('Start with a table.',guide)
        self.assertIn('ClassifierLessonGuide.title(for:',review.RAIL.read_text())

    def test_public_map_and_contract_changes_invalidate_review(self):
        for source in (review.RELU_REFERENCE,review.AFFINE_VIEW,review.AFFINE_TASKS):
            target=self.root/source.relative_to(review.ROOT)
            original=target.read_text()
            target.write_text(original+'\n// changed public contract source\n')
            with self.assertRaisesRegex(ValueError,'Current lesson changed'):
                review.packet(self.root)
            target.write_text(original)

    def test_approved_composition_copy_matches_and_keeps_native_public_maps(self):
        approved=json.loads((review.HERE/'proposals/affine-composition.json').read_text())
        applied=review.read_current(self.root,'affine-composition')
        self.assertEqual(applied['steps'],approved['steps'])
        self.assertEqual(applied['publicMaps'],approved['publicMaps'])
        view=review.AFFINE_VIEW.read_text()
        self.assertIn('if isComposition',view)
        self.assertIn('isRecovery ? AffineRecoveryLessonCopy.text(id):AffineCompositionLessonCopy.text(id)',view)
        self.assertIn('reviewedText("reference.criteria")',view)
        self.assertIn('steps:AffineCompositionLessonCopy.steps',review.RAIL.read_text())
        tables=review.AFFINE_TABLES.read_text()
        self.assertIn('var showsCombinedEquation=true',tables)
        self.assertIn('referenceMap(reference.first',tables)
        self.assertIn('referenceMap(reference.second',tables)

    def test_approved_recovery_copy_matches_without_exposing_project_values(self):
        approved=json.loads((review.HERE/'proposals/affine-recovery.json').read_text())
        applied=review.read_current(self.root,'affine-recovery')
        self.assertEqual(applied['steps'],approved['steps'])
        self.assertEqual(applied['publicObservations'],approved['publicObservations'])
        self.assertTrue(all(set(p)=={'id','title','input'} for p in applied['publicObservations']))
        self.assertEqual([p['input'] for p in applied['publicObservations']],[[0,0],[1,0],[0,1]])
        self.assertIn('steps:AffineRecoveryLessonCopy.steps',review.RAIL.read_text())
        view=review.AFFINE_VIEW.read_text()
        self.assertIn('if hasReviewedCopy {return reviewedText("reference.goal")}',view)
        reference=review.RECOVERY_REFERENCE.read_text()
        self.assertIn('if lesson.mode == .assessment',reference)
        self.assertIn('AffineRecoveryLessonCopy.text("reference.probes")',reference)
        self.assertIn('vector(probe.output',reference)
        self.assertIn('lesson.mode != .guided',reference)
        self.assertIn('lesson.revealHint',reference)

    def test_approved_backward_copy_matches_and_keeps_original_forward_tables(self):
        approved=json.loads((review.HERE/'proposals/affine-backward.json').read_text())
        applied=review.read_current(self.root,'affine-backward')
        self.assertEqual(applied['steps'],approved['steps'])
        self.assertEqual(applied['publicMaps'],approved['publicMaps'])
        self.assertIn('steps:AffineBackwardLessonCopy.steps',review.RAIL.read_text())
        view=review.AFFINE_VIEW.read_text()
        self.assertIn('isBackward ? AffineBackwardLessonCopy.text(id)',view)
        self.assertIn('AffineBackwardLessonCopy.mapIntroduction',view)
        self.assertIn('AffineBackwardLessonCopy.mapConclusion,showsCombinedEquation:false',view)
        self.assertIn('AffineBackwardLessonCopy.text("reference.meaning")',view)
        self.assertNotIn('L is a result whose sensitivity',view)

    def test_approved_frozen_copy_matches_exact_packet_and_reference_scope(self):
        approved=json.loads((review.HERE/'proposals/affine-frozen.json').read_text())
        applied=review.read_current(self.root,'affine-frozen')
        self.assertEqual(applied['steps'],approved['steps'])
        self.assertEqual(applied['publicStatistics'],approved['publicStatistics'])
        self.assertIn('steps:AffineFrozenLessonCopy.steps',review.RAIL.read_text())
        view=review.AFFINE_VIEW.read_text()
        for key in ['reference.statistics','reference.scope']:
            self.assertIn('AffineFrozenLessonCopy.text("'+key+'")',view)
        self.assertIn('if isFrozen {return AffineFrozenLessonCopy.text(id)}',view)
        self.assertNotIn('The saved training average is',view)

    def test_applied_relu_matches_original_owner_approved_copy(self):
        approved=json.loads((review.HERE/'proposals/archive/affine-relu-approved.json').read_text())
        applied=review.read_current(self.root,'affine-relu')
        self.assertEqual(applied['steps'],approved['steps'])
        self.assertEqual(applied['publicExamples'],approved['publicExamples'])
        self.assertIn('steps:AffineReLULessonCopy.steps',review.RAIL.read_text())
        self.assertIn('hasReviewedActivation',review.AFFINE_VIEW.read_text())

    def test_changed_guided_version_refused(self):
        plan=self.root/review.PLAN.relative_to(review.ROOT)
        plan.write_text(plan.read_text().replace('version:Int=5','version:Int=6'))
        with self.assertRaisesRegex(ValueError,'Guided course version changed'):
            review.read_current(self.root,'guided')

    def test_http_refuses_cross_origin_and_unknown_routes(self):
        server = review.ThreadingHTTPServer(('127.0.0.1', 0), review.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_port
        def send(method, path, headers, body=None):
            conn = http.client.HTTPConnection('127.0.0.1', port)
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse(); status = response.status; payload = response.read();conn.close()
            return status, payload
        self.assertEqual(send('GET','/api/review',{'Host':'attacker.invalid'})[0],403)
        self.assertEqual(send('POST','/api/decision',{'Origin':'https://attacker.invalid','X-Review-Token':review.TOKEN},'{}')[0],403)
        self.assertEqual(send('GET','/../../Sources/KwipMLApp/LessonContentCatalog.swift',{})[0],404)
        status, data = send('GET','/api/review',{})
        self.assertEqual(status,200)
        self.assertEqual(json.loads(data)['entryID'],'affine-relu')
        headers={'Origin':f'http://127.0.0.1:{port}','Content-Type':'application/json','X-Review-Token':review.TOKEN}
        before=self.source.read_bytes()
        original=review.decision
        with patch.object(review,'decision',side_effect=lambda data: original(data,self.root,self.state)):
            status, result=send('POST','/api/decision',headers,json.dumps(self.request))
        self.assertEqual(status,200)
        self.assertEqual(json.loads(result)['action'],'approve')
        self.assertEqual(self.source.read_bytes(),before)
        self.assertTrue((self.state/'affine-relu-decision.json').exists())
        self.assertEqual(send('POST','/api/decision',{**headers,'X-Review-Token':'invalid'},'{}')[0],403)
        self.assertEqual(send('POST','/api/decision',headers,'x'*80001)[0],413)

if __name__ == '__main__':
    unittest.main()
