# TORCH-01 · Conv1d delivery contract

Status: paired configuration/shape foundation verified; configuration, shared fields, graph/runtime/export and CPU lifecycle verified; discovery, shared palette and prepared example verified in source; scoped delivery accepted, including packaged native authoring/run/export/reopen. Broader convolution families and production gates remain open. Owner: Codex/root. Prior LazyLinear scoped delivery is closed. This begins the convolution family; it does not establish family-wide or production coverage.

## Operation and authored choices

Use actual nn.Conv1d with sample axes [channels,length] and independent batch N. Kernel weights perform cross-correlation, shared across positions. Groups partition channel connections; depthwise is the groups=in_channels case with an integer output-channel multiplier. Support explicit symmetric padding, valid and same policies; same requires stride1. Include zeros, reflect, replicate and circular modes, stride, dilation and optional bias. These semantics follow [PyTorch2.14 Conv1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv1d.html); exact installed2.14.1 signature/mode and boundary probes are recorded in `.local/conv1d-runtime-probe.log`.

Canonical schema2 config has exactly in_channels,out_channels,kernel_size,stride,dilation,groups,bias,padding_policy,padding,padding_mode. Policy is explicit/valid/same; mode is zeros/reflect/replicate/circular. Nonexplicit policies require padding0, avoiding an ignored stale value. This product representation maps to the corresponding PyTorch padding argument in construction/export. Scalars represent one-dimensional tuple equivalents. Integers and actual Boolean/string types are validated without coercing booleans into dimensions. Integral numeric JSON normalizes like existing components.

Expose these categorical choices through a reusable registry choice field, not numeric magic values or a Conv1d-only editor. The shared registry now supports labeled string choices; Conv1d exposure still requires graph/runtime and discovery integration. Preserve existing field decoding and editor context/Undo safeguards. No shared registry exposure until runtime and discovery agree.

## Preallocation contracts

Initial product bounds: channels/groups1…512; kernel/stride/dilation1…4096; explicit padding0…4096; input/output length1…4096. Both channel counts must divide by groups with no remainder. Input sample rank must be exactly2 and its channel count match in_channels; no implicit transpose or batchless interpretation.

Effective kernel=dilation×(kernel_size−1)+1. Resolve left/right padding before allocation: explicit uses padding on both sides; valid uses0; same uses total effectiveKernel−1, floor half on left and remainder on right. Each resolved side is bounded at4096. Reflection requires each side strictly less than input length; circular permits at most input length. Output length=floor((inputLength+left+right−effectiveKernel)/stride)+1. Reject a negative numerator before division; Swift truncation must not accept a zero-position output accidentally.

Parameter count=out_channels×(in_channels/groups)×kernel_size + optional out_channels bias, at most2M. Output retains the shared tensor element bound. Bound estimated kernel multiply-adds per observation at128M, computed as outputLength×out_channels×(in_channels/groups)×kernel_size before execution. This is a product workload bound, not a measured FLOP/memory/timing promise; padding, gradients and batch size add work. Full graph/batch resource policy remains separately governed. Configurable algorithmic modes stay available within these disclosed bounds.

## Integration and acceptance

Paired Swift/Python normalization and shape fixtures precede graph/runtime integration. Validate float32 product inputs before allocation. Test numerical outputs and input/weight/bias gradients against independently expressed convolution, including groups/depthwise, stride/dilation, asymmetric same padding and all boundary modes. Verify CPU/MPS support and disclose observed fallback; do not assume determinism across devices.

Preserve nn.Conv1d constructor modes, named state, checkpoint signature compatibility and standalone export. New convolution-containing graphs must bind parameter registration order without changing old nonconvolution signatures. Verify exact CPU continuation, weights-only fine-tune, initialized/trained preview and cache invalidation. Include a real prepared example and concise mathematical learning material; selected first-release curriculum remains post-ART. Native controls must explain channel versus length axes and derived output shape. Package and observe example/edit/run/history/export before scoped closure. Production gates, broad sources/model tasks and convolution2d/3d/transposed families remain open.

## Configuration foundation evidence — 2026-10-07

Paired strict normalization and allocation-free shape resolution implemented without registry or runtime exposure. Python RED stopped on the missing module (`.local/conv1d-contract-python-red.log`); GREEN with LazyLinear/Bilinear contract regressions: 83 passed, no skips (`.local/conv1d-contract-python-green.log`). Representative shapes compare all 12 policy/mode combinations against actual pinned PyTorch. One expected warning documents that same/zeros with an even kernel can allocate a padded copy.

Swift RED encountered the missing API plus invalid string-literal fixtures; fixtures were corrected to explicit JSONValue.string before GREEN. Four grouped/depthwise, padding, strict-config and resolution-bound tests pass with no skips (`.local/conv1d-contract-swift-green.log`). No claim of a clean assertion-only Swift RED. Review confirmed Python floor division and Swift negative-numerator refusal both reject nonpositive output lengths.

Next: reusable categorical field contract and native shared controls, followed by Conv1d graph/runtime/state/export integration. No Conv1d native or end-to-end acceptance yet.

## Shared categorical fields — 2026-10-07

Added choice fields with distinct machine values and display titles, 1–64 unique nonblank choices, 128-scalar value/title bounds and strict string selection. Legacy fields omit choices on encoding. Choice definitions refuse numeric/list bounds and nullability; reusable Block parameters may narrow but never expand target value sets. Shared native editor uses the existing flat keyboard-accessible selector with disabled/context guards and accepted-change callback. Boundary captions now distinguish choices and real numbers from booleans.

RED missing choice APIs: `.local/choice-field-red.log`. GREEN22 tests (14Core/8App), no failures/skips: `.local/choice-field-green.log`, covering choice contracts/normalization plus numeric/list/optional fields, Block editing/publication/Undo and shared selector navigation regressions. This verifies source and automated behavior; rendered choice controls await the packaged Conv1d journey. Python registry choice validation will be integrated with Conv1d runtime before any built-in choice field is exposed.

## Graph, runtime and lifecycle evidence — 2026-10-07

Swift graph validation now resolves channels-first Conv1d contracts and counts trainable parameters; schema1 and integer tensors are refused. The declarative component has all ten fields, including both categorical settings, but is not yet in the shared palette. RED missing component API, GREEN18 Core tests including configuration/choice and prior graph regressions (`.local/conv1d-graph-swift-red.log`, `.local/conv1d-graph-swift-green.log`).

Python uses actual nn.Conv1d with authored modes and strict categorical registry validation. Independent indexed cross-correlation checks outputs and input/parameter gradients for all four boundary modes, both bias settings, asymmetric same, grouped/depthwise and stride/dilation cases. CPU float64 and actual MPS float32 checks pass. Runtime and generated architecture require a nonempty batch and the exact authored channel/length shape before convolution; standalone source preserves constructor settings and state keys. Aggregate parameter overflow is refused before constructing modules.

Graph RED20 failures/11 passing negative cases (`.local/conv1d-graph-python-red.log`); initial GREEN113 tests. Lifecycle RED6 failures/1 legacy passing case (`.local/conv1d-lifecycle-red.log`); GREEN30 including historical lifecycle regressions. Verified actual four-epoch CPU training versus two plus exact continuation, RNG-consuming callbacks, model/optimizer/RNG and metric equality, weights-only fine-tune, gradient inventory, frozen/export parity, trained preview/cache reuse and configuration invalidation. Conv-containing checkpoint signatures bind all parameter registration order, preserving historical nonconv/nonlazy schedules.

Review caught generated shape-check use of builtin tuple shadowed by a user label. Regression RED1 failure, fixed with direct shape-slice comparison (`.local/conv1d-export-label-red.log`). Final combined acceptance144 Python tests, no skips, one expected same-padding copy warning (`.local/conv1d-runtime-accepted.log`). No MPS exact-resume or native acceptance claim. Next capability discovery/bridge, shared registry and prepared example/math material, then fresh package and native authoring/run/export checks.

## Discovery and prepared example — 2026-10-07

Added exact bounded Conv1d capability emission/decoding and optional runtime gating, including stale-support reset and actual worker discovery. Shared registry matches the reviewed ten-field component; bridge declares channels/length plus one batch axis, learned parameter state and the actual prepared-conv1d-classifier example. Icon follows the rose Layers family with a sliding-kernel visual. No new graded lesson claimed.

Prepared example projects two numeric measurements to fourteen features, reshapes [N,2,7], applies grouped 2→4 convolution with same padding, then ReLU, flatten28 and a two-score head. Both convolution bias modes compile/round-trip; canvas layout keeps cards separated. Learn material derives grouped cross-correlation, output length and116 default parameters, explains shared weights and explicitly distinguishes constructed feature positions from genuine temporal observations.

Discovery Python RED missing capability; GREEN47 tests (`.local/conv1d-discovery-python-red.log`, `.local/conv1d-discovery-python-green.log`). Swift discovery/example RED missing APIs; combined GREEN27 tests (18Core/9App), no failures/skips (`.local/conv1d-discovery-example-green.log`). Actual bundled example test generated code before training and completed/verified History in3.292s. MPS coverage expanded to all12policy/mode combinations, all pass with no skips and the expected same/zeros padded-copy warning (`.local/conv1d-mps-mode-matrix.log`).

Next fresh canonical worker package and observed native Learn/example, categorical controls, shape edits/Undo, training, History and trained export. Automated integration is not native acceptance.

## Native candidate and choice-control finding — 2026-10-07

Fresh canonical payload `.local/conv1d-runtime-82e0273` verified70 worker files; package/source integrity and local signing passed (`.local/conv1d-payload.log`, `.local/conv1d-package.log`). Native Learn explanation rendered with grouped math, full text after scrolling, and example opened on universal canvas with correct [N,2,7]→[N,4,7]→[N,28] labels.

Native RED: Padding mode received focus but did not open after AX click, Space or pointer click. Shared selector delayed opening until a FocusState callback that nested inspector focus did not supply. Also choice controls lacked visible field titles. Repair57dc4c1 opens the popover directly and lets its existing presentation task focus the list; visible registry choice labels added. Five selector-navigation/choice tests pass (`.local/conv1d-choice-repair-green.log`). Candidate quit before repair. Rebuilt candidate requires native replay before acceptance; do not infer rendered fix from automated tests.

## Native scoped acceptance — 2026-10-07

Candidate57dc4c1 rebuilt against source-verified payload (`.local/conv1d-choice-package.log`). Native replay confirmed visible Padding policy/mode labels, pointer-opened dropdown and selection of Reflect. Keyboard Up/Return selected Valid; Undo restored Same length and inferred [N,4,7]/flatten28 contracts. Initial rapid batched Up/Return did not change selection; separately observed highlight and Return committed correctly. No general rapid-key timing guarantee claimed. Toggled biasoff; Code before training showed nn.Conv1d with groups2/same/reflect/biasFalse and the exact sample-shape guard.

Saved/run completed three epochs. Native History verified29artifacts,6computed preparation nodes,0reused for run46D0A590-972D-4499-B54C-F1FA9B2334F3, request4998c0be328a347bc2e42a8954fe9d9755d79ee1531c52ccaa7315a4ae5bc92a; started2026-10-07T09:53:29.091Z, finished09:53:34.477Z. Validationloss0.19433/100%accuracy is this small teachingfixture only.

Native trained export saved `.local/KwipML-model-46D0A590`. Independent development Python ran inference.py --example --batch-size2 from that directory, exit0, finite logits[[0.0955008268,0.1063868701],[-0.6390278339,0.8821907043]], classes[1,1]. Export explicitly discloses prepared inputs and excluded preprocessing/source rows. Restart restored same/reflect/biasoff and reverified the same History archive. Candidatequit.

Scoped Conv1d delivery closed; no new gradedlesson, fullnativeaccessibility matrix, general source/task coverage, convolution2d/3d/transposed or productiongate completion implied. Shared choice opening repair has native pointer/arrow/Return evidence; consolidated UAT remains post-ART.
