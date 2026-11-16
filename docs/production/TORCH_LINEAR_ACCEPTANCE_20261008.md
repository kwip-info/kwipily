# Linear bounded v1 acceptance · 2026-10-08

Accepted reviewed consumer capability pytorch.torch.nn.Linear in all four preserved modes: inference, training, input-gradients, state-roundtrip. This frozen receipt supports one bounded unit, not whole PyTorch coverage, release readiness or expert mastery.

## Contract and authoring

Native1.2.0, schema2 explicit boolean bias; schema1 historical implicit enabled bias retained. Exact in_features default2/out_features default8, both1–512; bias defaulttrue. Dense float32 CPU/MPS rank2–8, shared weights transform the last feature direction, earlier directions and batch order preserved. Independent node parameter ownership, absent bias parameter when disabled; existing allocation/tensor/node/parameter budgets retained. Sparse/ragged/nested/quantized/other graph dtypes and unsupported devices remain refused. No mandatory optional discovery descriptor was added to legacy Linear workers. Assessment runtime1.1.0 remains separately recorded and compatible; native1.2.0 shape contract does not rewrite historical receipts.

One exact definition migrated into components/layers/bundle.json, existing validator/generator produces BundledLayerComponents; no duplicate authority or arbitrary executable metadata. Purpose/title, Learn discovery, prepared source and actual example dispatch consume authored metadata. Existing generated rose hexagon Linear artwork/provenance retained, no placeholder substituted.

## Meaningful RED/GREEN and numerical evidence

Metadata API scaffold RED recorded missingLinearComponent/linearDeclaration. Exact drift refusal and field/default/rank/state/lesson/schema checks pass. Native-discovery integration RED exposed missingLinear Learn entry; shared discovery/launch integrated. Legacy fixture corrected to represent actual schema1 without initializer-addedbias; no migration behavior loosened. Familyauthoring RED1unsupported-family failure/27refusal controls; GREEN110authoringchecks and deterministicdrift check. Registry count remains30.

47Core+29App=76scoped checks pass without skips (.local/linear-scoped-swift-green.log), including actual measured affine/bias/private receipt/persistence flows. Additional typed configuration replay passes; existing legacyworker semantics retained. 167worker checks/no skips (.local/linear-axes-independent-worker.log):68new scalar math.fsum oracles for CPU/MPS rank2–8, biasboth,batch1/3, noncontiguous input, forward/dX/dW/db, statekeys/count, unchanged inference/train-eval,rowisolation; actual generated modules and whole exportedgraph agree. Rank3endpoint1→1/1→512/512→1 complements existing512→512matrix. Existing40rank2matrix/17bias/19boundary/15checkpoint/8learning replay. No fabricated numericalRED or tolerance relaxation; worker semantics unchanged. Supplementaltest SHA256 be2459e53ff59a6ba881403d9887768c4ee38134ec6376e8840d6b9bbb181131.

Further38worker lessonchecks pass/no skips (.local/linear-affine-worker-replay.log): optionalbias, backwardchain, seededprobe recovery and frozenstandardization. Lesson objectives and independent private assessment criteria preserved. CPU exact-resume/cache/RNG/optimizer evidence and MPS weights-only fine-tune with explicit exact-resume refusal come from replayed checkpoint suites; no cross-device bitwise guarantee.

## Native learner and application evidence

Candidate821dbf4fb47a8712385c35419629abcfe672b516ef2c9df2ab8721e76a03041e integrity/deep local signature passes (.local/linear-package.log), build-owned runtime .local/linear-runtime-20261008. Python3.13.6/PyTorch2.14.1 arm64macOS26.6; actualCPU/MPS.

Project6BA36BE1-8166-421B-B140-4E4C8DE281C3 nativepublic affineconstruction: addedLinear, configured2→2, initializedexplicitvalues, enteredweights[3,5;-3.5,5] andoffsets[-2.75,-3], actualportdragsInput→Linear→Output. Guided16measuredvaluespass,max4.88281e−6, formatted input/output/expected/signeddifferences. Biasoff preserves disabled storedoffsets and editableweights; actualcheckfailsmax3 with actionableoffset/order/Parametersguidance. Re-enable restoresexactoffsets; privateAssessment16valuespass,max2.44141e−6, savedattempt5E641727-0516-45B7-83EE-293C3FE4313D, fixture462d3161a7d111367a08cbef2863f9e81841fd483f924f826dc4f28e6adaaa29. Private numericalexamples hidden. Save/quit/relaunch restoresmatchingverifiedprivatecompletion; screenshot .local/linear-affine-private.png.

Linked existing assessed affinecomposition, seededaffinefromprobes, affinebackwardchain and frozenstandardization plus parallelcourse; no newgrader or masteryclaim. NativeLearnsearchweightsfinds authoredLinearentry and opens actualeditableparallelclassifier. Requiredcopy explainsweights,offsets,feature direction andbatch; optional mathematicaldetails retain rigor.

Project404114F0-A388-4650-9441-6DFFDD7166F8 actual prepared14componentflow; nativeCode beforetraining shows two independentLinear2→8 and head16→2,biasTrue plus architecture-only disclosure. Auto resolvesMPS,3epochs/32batch/last16 finishrun89631E4E-7568-4329-BD82-66226FCC223F, archive13AF9A2E-A686-414F-864E-D5371DC0A178, request10f406b85bc4037a1a4e1857038c356eb62bcaf9a3e6f88798a9bb2619172263. History verifies31artifacts/6computed/0reused. Validationloss0.24466007947921753/accuracy100% on16rows is fixtureonly. BottomResults selectedhead out[8,2]float32 table8/16retainedscalars; sampled limitation visible.

NativeSavedmodelsource→Exporttrainedmodel→review input contract→localSavepackage produced .local/KwipML-model-89631E4E. inference.py --example --batch-size1 exits0. Independent scalar weighted-sum→ReLU/tanh→weighted-sum oracle agrees with strictloaded checkpoint on3preparedinputs,max2.2436885016574593e−7; batch1parity. model.py SHA985822001dd06521d61ff9adbe004c1562bf14180eda81bfe9ed5b948a71428e; checkpointSHAf30c8e788e55bc66586a0aebaa52951becf7c10cfc7c04420d8445253377c9a4. Sixexactstatekeys/twoindependentbranches/head verified. Exportrequiresfinitepreparedfloat32[N,2],signal→context; fittedpreprocessing andprivate rows excluded.

NativeLight/Dark150%selectednode/palette, sharp Configuration/Information, generatedrosehexagonconnected-node art andcapturedtable readable. OriginalLightrestored. Screenshots .local/linear-artwork-light.png/.local/linear-artwork-dark.png. Sharedprior6replay4Core+21App/no skips and actualLayerNormpreparedruns documented TORCH_PRIOR_LINEAR_REPLAY_20261008.md; originalreceiptsunchanged.

## Limits and remaining work

Supportedmachine/cleanaccount/distribution/licensing/accessibility/stress gates remain open. MPS checkpointguarantee weightsOnly; exactresume CPU scoped. No standalone rawsource/preprocessingexport claim. Native26/27tab coldstartup slow from repeated fullcatalogdecoding; bounded PERF-RESTORE queuednext without deletingownerstate. This observed applicationperformanceissue does not invalidate verifiedLinear numerical/configuration/state behavior. Other88unit inventory work and family/finaldeferral gates remain separate.
