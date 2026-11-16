# Bag archive/request transport — 2026-10-09

Status: scoped native archive/request and launch verification passes. This is a bounded transport slice, not bundle admission or native acceptance.

## Contract

Workflow12/compiler11 uses capture7. Workflow11/compiler10/capture6 retains its existing exact qualified composition descriptor and legacy behavior. Graph2, Composite source3, native expansion2 and CompositionPlan1 remain unchanged. The same runtime-bound composition discovery evidence qualifies source3 on workflow12; the descriptor itself is not globally rewritten to claim a different transport contract.

WorkflowArchivePlan stores optional typed bagInput, omitted for historical and dense projections. Source controls are derived from the actual captured graph; there is no second editable source configuration. Envelope validation binds mapping presence, exact source domain, role controls, ordered indices and binary/objective targets. Full snapshot validation independently expands captured source with captured registry, verifies role/consumer admission through the Core bag requirement (minimum vocabulary / maximum output width across compatible consumers), and checks semantic identity. Unknown mapping fields and explicit optional nulls remain rejected by strict decoding/normalization.

Request capture requires runtime-qualified composition discovery for workflow12 and pins the same document/registry/runtime identities. Immutable writer emits capture7 and pins real source bytes, model/data registries and archived mapping. Root owns phase arguments, supervised authorization, training and result readers outside this lease.

## Actual TDD evidence

Root executed `.local/bag-workflow-archive-red.log`: six methods, six failures (four unexpected), 0.931 seconds. Failures included unsupported compiler11 archive projection, incorrect workflow12→compiler12 fallback and request failing before runtime qualification. No production changes preceded this RED.

New BagWorkflowArchiveTests cover binary/objective fixed/packed weighted/unweighted mappings, source3 re-expansion, missing/null/unknown/role/version/descriptor failures, legacy6 canonical bytes and omitted mapping, missing runtime qualification, and immutable writer source/registry pins. Descriptors/interpreter files are independent synthetic unit fixtures; they do not constitute runtime probing or native evidence.

Root GREEN: `.local/bag-native-transport-green.log`,28 tests with zero failures, including all6 new archive methods and3 launch argument methods. Expanded adjacent replay `.local/bag-native-transport-regression-green.log`:54 tests,zero failures,1 explicit environment-gated schema-five real-training replay skipped. The53 executed methods pass; the actual legacy worker replay remains pending until frozen worker authority changes are ready. Synthetic unit fixtures do not establish packaged runtime or native visual acceptance. No tests/builds or native computer use were run by the implementing subagent.
