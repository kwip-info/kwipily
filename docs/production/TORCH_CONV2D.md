# TORCH-01 · Conv2d delivery contract

Status: scoped Conv2d delivery accepted locally through packaged native run/export/restart; broader convolution and production acceptance remain open. Owner: Codex/root. Conv1d scoped delivery closed; broader convolution coverage remains open.

## Authored settings

Actual nn.Conv2d; independent observations with sample axes [channels,height,width]. Kernel size, stride, dilation and explicit padding are canonical two-element JSON arrays [height,width], edited with existing bounded list controls. Equal pairs express scalar-equivalent settings without ambiguous persisted forms. Remaining exact keys: in_channels,out_channels,groups,bias,padding_policy,padding_mode. No hidden defaults or extra keys.

The operation uses cross-correlation. Groups divide both channel counts. Policies explicit/valid/same and modes zeros/reflect/replicate/circular follow [PyTorch2.14 Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html). Same requires both strides1. Local pinned2.14.1 non-square kernel probes cover all12policy/mode combinations in `.local/conv2d-runtime-probe.log`. Product input remains float32 and batched; raw PyTorch's batchless/complex alternatives are not claimed.

## Bounds and common geometry

Channels/groups1…512; each kernel/stride/dilation value1…4096; explicit padding0…4096; input/output height and width1…4096. Bias is an actual Boolean, policy/mode actual known strings, pair elements integers (integral numeric JSON canonicalizes; Boolean dimensions refused). Nonexplicit padding requires [0,0]. Parameters out×(in/groups)×kernelHeight×kernelWidth + optional out bias ≤2M.

Reuse one axis resolver between Conv1d/Conv2d, preserving Conv1d accepted behavior. Per axis: effectiveKernel=dilation×(kernel−1)+1; same splits effectiveKernel−1 with floor half before and remainder after; explicit pads equally; valid0. Each side≤4096, reflection side<input extent, circular side≤input extent. Reject negative output numerator before Swift division; output=floor((input+before+after−effectiveKernel)/stride)+1. Return resolved top/bottom/left/right explicitly to avoid height/width versus PyTorch pad-order ambiguity.

Conv2d adds a padded-sample-volume preflight: inputChannels×paddedHeight×paddedWidth≤100M elements, including implicit-zero policy as a conservative bound. Large stride must not bypass this check just because output is small. Output retains shared100M element bound. Estimated kernel multiply-adds=outputHeight×outputWidth×outChannels×(inChannels/groups)×kernelHeight×kernelWidth≤128M per observation. These are scoped product bounds, not peak memory estimates; batch, gradients and backend workspaces add costs. Config bounds keep intermediate integer arithmetic within Int64.

## Acceptance sequence

Tests precede paired normalization/shape implementation and shared-axis refactor. Include non-square kernels, independent stride/dilation values, asymmetric same padding, depthwise/grouped counts, strict malformed pairs, each-axis boundary refusals and resource limits. Preserve all Conv1d regression evidence. Do not expose a shared registry entry at this stage.

Then graph/runtime/standalone code and named-state integration with exact authored batch/sample boundary guards; independently calculated outputs/gradients; CPU/MPS policy/mode coverage; checkpoint order, exact CPU continuation, fine-tune, preview/cache and export. Strict capability negotiation and a real prepared example/math explanation precede fresh packaging/native controls/run/History/export/reopen. Image-file sourcing and broad computer-vision training tasks remain separate inventoried capabilities, not implied by a Conv2d node.

## Foundation evidence — 2026-10-07

Strict Swift/Python normalization and shape contracts implemented with reusable per-axis geometry shared by Conv1d. No registry/palette/runtime exposure for Conv2d yet. Python RED missingmodule collectionerror (`.local/conv2d-contract-python-red.log`); GREEN190 tests including Conv1d contracts/graph/all12MPS modes/lifecycle/discovery, no skips, one expected same-padding copy warning (`.local/conv2d-contract-python-green.log`). New tests compare all12Conv2dpolicy/mode output shapes to actual pinned PyTorch and independently test non-square geometry, exact pair types, bias crossing2Mparameters, padded volume for allmodes and output/work limits.

Swift RED missingAPI (`.local/conv2d-contract-swift-red.log`); GREEN13 tests including Conv1d configuration/graph/preparedexample regressions, no failures/skips (`.local/conv2d-contract-swift-green.log`). Review confirmed bounded Int64 arithmetic and top/bottom versus left/right mapping. Next graph/runtime/numerical gradients/export/state integration; this evidence does not imply a usable Conv2d canvas component yet.


## Graph/runtime and lifecycle evidence — 2026-10-07

Added private ten-field component definition, schema-2 canonicalization, float32 channels-first graph inference and trainable parameter accounting. Actual nn.Conv2d shares constructor/execution/export plumbing with Conv1d while preserving paired spatial settings. Runtime and standalone source reject empty, unbatched or mismatched authored sample shapes. Export retains named weights and handles a user label named `tuple` without shadowing a builtin call.

Swift RED missing component API; GREEN10 Core tests, no failures/skips (`.local/conv2d-graph-swift-green.log`). Python graph RED31failed/6passed; first graph GREEN81 including Conv1d (`.local/conv2d-graph-python-green.log`). An independent indexed two-dimensional cross-correlation oracle checks values and all gradients for four boundary modes and both bias states; additional cases cover non-square stride/dilation, groups/depthwise, noncontiguous inputs, state/export gradients, resource limits and actual MPS all12policy/mode pairs.

Lifecycle RED4failed/5passed identified absent Conv2d optimizer registration order (`.local/conv2d-lifecycle-red.log`). Checkpoint schedule now includes Conv2d parameter nodes; old nonconv/Conv1d signatures and lazy-only initialization remain unchanged. Final combined GREEN279 Python tests, zero skips, one expected PyTorch same-padding copy warning (`.local/conv2d-runtime-accepted.log`). Exact CPU continuation compares model/optimizer/RNG/metrics; tests also verify frozen/export parity, parameter gradient inventory, fine-tune reset, initialized/trained preview reuse and stale refusal, and actual MPS training/preview.

This closes graph/runtime lifecycle only. Public shared registry, capability negotiation, prepared example/math material and fresh packaged native acceptance remain pending. Last accepted native candidate remains57dc4c1, quit; no production gate closes from these checks.


## Discovery and example evidence — 2026-10-07

Shared palette entry matches the reviewed ten-field paired configuration. Exact optional worker/native discovery binds sample rank3, spatial pair length2, dtype, padding policies/modes and padded/output limits. Stale support clears on refresh/failure and missing or incompatible descriptors refuse Conv2d execution while preserving legacy graphs. Bridge ports describe channels/height/width plus exactly one batch axis, learned parameter state and the prepared example.

The prepared workflow projects two measurements to40 features, reshapes[2,4,5], applies grouped Conv2d2→4 with kernel[2,3]/same padding, ReLU, flatten80 and classifier2. The rose Layers icon depicts a spatial kernel. Learn explains cross-correlation, shared weights, group indexing, axis output equations and asymmetric same padding; it explicitly distinguishes constructed grids from image sourcing. This is an example/explanation, not a graded course.

Python discovery RED2fail (missing descriptor/API); GREEN51pass0skip including realMPS and graph/lifecycle regressions after shared registry changes (`.local/conv2d-discovery-catalog-green.log`). Swift combined RED missing example APIs and dependent type inference (`.local/conv2d-discovery-example-red.log`); GREEN27tests (21Core/6App),0fail/skip (`.local/conv2d-discovery-example-green.log`). Tests cover exact descriptor refusal, registry/bridge consistency, stale support, live worker probe, bothbias shapes, layout/serialization/private source binding/reopen, and actual code-before-run/preparation/training/verifiedHistory (3.453seconds).

Next fresh canonical worker packaging and native controls/Learn/run/export/restart. Automated integration is not rendered acceptance. Last accepted native candidate remains57dc4c1, quit; production gates remain open.


## Packaged native acceptance — 2026-10-07

Candidate449ac8e built from fresh `.local/conv2d-runtime-449ac8e`; canonical72-file source verification and pre/post-sign integrity passed (`.local/conv2d-payload.log`, `.local/conv2d-package.log`). Local ad-hoc signing only. Prior candidate retained by packaging script.

Observed native Learn entry and expanded equation/explanation, including constructed-grid limitation; initial disclosure animation briefly clipped the last line, subsequent scroll showed the full content. Opened the real example; connector contracts display [N,2,4,5] input and [N,4,4,5] output. Kernel pair2,3→3,3 committed with Return, Undo restored2,3. Biasoff and Reflect dropdown selection persisted; Code before training displayed nn.Conv2d(2,4,(2,3),stride=(1,1),padding='same',dilation=(1,1),groups=2,bias=False,padding_mode='reflect'). No blanket all-fields/accessibility claim.

Actual3epoch native run **4443D13E-3839-49AD-9484-5A0EDBF628EB**, request **4a8ee77fc1efa35666d114926d59b519360a52c48bc9829794fb34a7a78f26c5**, started2026-10-07T10:16:52.281Z, finished10:16:57.708Z. History verified29artifacts,6computed/0reused preparation nodes, validationloss0.010884 and100%smallfixture accuracy (not generalization evidence).

History→Saved model source→Export trained model produced `.local/KwipML-model-4443D13E`. Native review disclosed preparedfloat32[N,2] signal/context input and excluded source rows/fittedpreprocessing. Independent `.venv/bin/python inference.py --example --batch-size2` exited0 (`.local/conv2d-export-inference.log`), logits[[0.1213795245,-0.0051856041],[-2.0191068649,1.9419440031]],classes[0,1].

Quit/relaunch resumed the saved project with kernel2,3,Same,Reflect,biasoff; History reverified the same completed archive and identities. Candidatequit, process absence confirmed. Scoped Conv2d delivery closed; no gradedlesson or fullproductiongate closure implied. Continue remaining inventoried convolution families, next Conv3d.
