# MaxPool3d delivery

Status: scoped delivery accepted in candidate68e95c1,2026-10-07. Broader production, owner ART and assessed curriculum acceptance remain open.

## Contract and acceptance

Use [N,C,D,H,W] with explicit [depth,height,width] kernel, nullable stride, padding and dilation triples; ceiling mode and optional indices. Null stride follows each kernel dimension. Preserve channel/batch independence. Bounds follow the existing pooling axis contract: channels1–512, spatial/kernel/stride/dilation1–4096 and padding0–floor(kernel/2). Reuse shared spatial normalization and reviewed1Daxis geometry in native/worker implementations.

Reference: [PyTorch2.14 MaxPool3d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool3d.html), checked2026-10-07. Int64 indices identify flattened input volume positions within a channel; they are not differentiable features. Pooling has no learned parameters but passes gradients upstream.

1. Strict triples, native/worker geometry parity and malformed/empty/oversized-axis refusal, with floor/ceiling, dilation and padding edges.
2. Exact values/indices outputs in graph validation, runtime, generated code, caches, composites, Results and saved History.
3. Independent volume-window/index oracle, ties/overlap gradient accumulation, noncontiguous inputs, actual CPU/MPS support and all-padding finite-output diagnostics.
4. Real training, exact CPU checkpoint resume, cache reuse, standalone export/inference and restart.
5. Public discovery/registry, configurable controls, themed presentation, runnable example and explanation connecting spatial selection to linear-algebra gradients. Native interaction verification; owner ART/scored curriculum separate.

## 2026-10-07 — Worker geometry and device feasibility

Missing-module RED (.local/maxpool3d-worker-red.log). Added shared spatial contract for2D/3Dusing reviewed1Daxis geometry;2Ddescriptor remains unchanged. Worker70tests passed with no skips (.local/maxpool3d-worker-green.log), including128asymmetric volume geometries against actualTorch and actual MPS forward/backward/indices in both return modes. Includes2Dgraph/discovery and1Dgeometry regressions. Native foundation and existing2Dintegration also passed; see evidence below. No graph/runtime admission claimed.


## 2026-10-07 — Shared spatial pooling and MaxPool3d geometry

- Added strict MaxPool3d depth/height/width triples and optional indices geometry. Extracted shared spatial normalization/resolution in Swift/Python;2Dand3Ddelegate every axis to reviewedMaxPool1dgeometry. Preserved2Dpublic config serialization and runtime descriptor.
- Tests-first missing module/API RED (.local/maxpool3d-worker-red.log, .local/maxpool3d-native-red.log). Worker70checks passed without skips, including128asymmetric volume geometries compared toTorch and actual MPS forward/backward/indices in both modes (.local/maxpool3d-worker-green.log).
- Native16checks:15passed/1opt-in2Dintegration skipped due missing environment (.local/maxpool3d-native-green.log). Explicitly reran that real3epoch integration with KWIPML_TEST_DEVELOPMENT_ROOT; passed without skip (.local/maxpool3d-shared-integration.log). Native tests cover triple boundaries, null stride, indices, independent depth/height/width and existing2Dgraph/discovery.
- [MaxPool3d contract](TORCH_MAXPOOL3D.md) records remaining execution/registry/discovery/example/Learn/native acceptance. Next implement graph/runtime/exports using exact output routing. Last accepted candidate17e405c remains quit and unchanged. All verification handles terminal; broad gates unchanged.


## 2026-10-07 — Volume runtime/native graph checkpoint and owner bundle direction

- MaxPool3d worker tests-first8failures (.local/maxpool3d-graph-red.log); added volume resolver/function dispatch and schema2 semantics through shared pooling execution/export. Final62Python checks passed, zero skips (.local/maxpool3d-runtime-green.log): independent volume-window/flattened-index oracle, ties/overlap and noncontiguous gradients, CPU/MPS, exact export, empty-batch refusal, nullable stride, typed-port refusal, checkpoint/cache/saved samples and all-padding nonfinite capture. One pre-existing nested-tensor warning.
- Native missing-component RED (.local/maxpool3d-native-graph-red.log); implemented reviewed triple fields, bridge, graph migration/types/shape and optional port availability using a private test registry. Final10Core checks passed (.local/maxpool3d-native-graph-green.log). Not publicly registered; capture/discovery/example/native acceptance still open.
- Owner highlighted per-component bundles for icon, configuration templates and code. Audit confirmed existing descriptors/shared controls but excessive scattered registration. Recorded [bounded bundle migration](../architecture/ADR-002-COMPONENT-BUNDLES.md): use pooling as the first migrated family, preserve IDs and saved behavior, finish3Dthrough that path before further catalog expansion. This supersedes adding more manual registration chains, not the full MVP objective. All processes terminal; last accepted candidate17e405c unchanged/quit.


## 2026-10-07 — Public bundle delivery and example

- Replaced private test registry with shipped registry; missing public descriptor/example RED (.local/maxpool3d-public-red.log, .local/maxpool3d-example-red.log). Published through manifest, adding3Dexample factory/source and worked math to shared Learn/launcher. Native discovery, counts, reset, eligibility and capture admission required no per-member app state.
- Shared2D/3Dworker capability factory preserves2Dcontract and advertises volume axes. Example projects2→120, reshapes[N,2,3,4,5], poolsK[2,2,3]/S[1,1,2]→[N,2,2,3,2], flattens24→2.410parameters. Independent worked learning volume checks values/indices and Aᵀg; real CPU/MPS graph/export/gradient/cache/resume tests now use public registry.
- Final38Python checks passed, zero skips (.local/maxpool3d-public-green.log).13Core+1actual three-epoch app code/train/capture/History integration passed (.local/maxpool3d-example-green.log); additional1Coreworkflow roundtrip/bothmodes/layout plus7capture-contract checks passed (.local/maxpool3d-example-contract.log). No dedicated3Dapp flag, Learn view or launcher action. Native package/controls/export/restart next.


## 2026-10-07 — Packaged native and standalone acceptance

- Candidate68e95c1 built with refreshed/verified .local/maxpool3d-runtime-20261007 (.local/maxpool3d-package.log). Native1440×900Learn displayed volume math, worked coordinates and backward map; generic example launcher opened the authored flow. All six controls visible: [depth,height,width] triples, nullable stride, ceiling mode and indices. Stride off→Notset/on→[1,1,1], committed[1,1,2]; indices off/on removed/restored connector. Information collapsed below configuration.
- Codebeforetraining rendered syntax-highlighted5Drank/exactshape guard and named max_pool3d values/indices. Actual3epoch run58C0CA38-CAC5-466C-8F41-C0D94F988208, revision5, request0c1598be206ec5991864e80bfa1d673888cb97aa15d4e021f09a11178106a1a8, start2026-10-07T15:14:20.082Z/end15:14:25.613Z.30verified artifacts,6computed preparation stages. BottomResults indices[8,2,2,3,2]int64 [22,22,30,33,16,33,22,48];outfloat32 [1.284,1.284,0.919,1.066,1.045,1.066,1.284,1.557].
- Native reviewed export saved .local/KwipML-model-58C0CA38. Standalone ../../.venv/bin/python inference.py --example --batch-size2 exited0, logits[[-0.3900639415,-0.1373859793],[-1.9401874542,0.7033547759]],classes[1,1]. This uses synthetic prepared inputs; raw preprocessing is excluded and disclosed by export review.
- Saved,quit/processabsence,relaunch/resume; History reverified same run/request and exact typed activation samples. Quitagain/processabsence.
- Acceptance mapping: strict triples/geometry→Core/worker configuration tests; typed optional outputs→public graph/bridge/capture checks plus nativeconnector toggle; numericalsemantics→independent volume oracle,CPU/MPS,gradients and all-padding failure; lifecycle→exactcheckpoint/cache tests plus actualtraining/standaloneexport/restart; product→nativecontrols/code/Results/History and bundledmath/example. Scoped delivery accepted; no full production gate or assessed-course claim.
