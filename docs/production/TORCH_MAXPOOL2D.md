# MaxPool2d delivery

Status: scoped MaxPool2d delivery accepted locally in candidate17e405c on2026-10-07. Automated numerical/lifecycle evidence and native controls/code/run/Results/History/export/restart passed. Owner ART, scored courses and broad production/distribution acceptance remain separate.

## Contract and acceptance

Use batched channels-first [N,C,H,W]. Expose paired [height,width] kernel, nullable stride (null follows the paired kernel), padding and dilation, plus ceiling mode and optional indices. Explicit pairs make rectangular windows and unequal steps editable without conflating axes. App bounds follow MaxPool1d: channels1–512, spatial/kernel/stride/dilation1–4096, each padding0–floor(kernel/2). Indices are int64 flattened positions within the input spatial plane, not channel indices or differentiable outputs. No learned parameters.

Reference: [PyTorch2.14 MaxPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html), checked 2026-10-07. Reuse reviewed MaxPool1d axis normalization/geometry in both native and worker implementations; do not duplicate signed floor and ceiling correction arithmetic.

1. Strict paired settings and native/worker geometry parity for rectangular inputs, nullable stride, dilation, ceiling edges, malformed settings and resource bounds.
2. Exact optional values/indices routing through graph validation, composites, execution, cache, captured outputs, History and generated code. Preserve float/int distinction.
3. Independent 2D window and flattened-index oracle, tie/overlap gradients, noncontiguous input, actual CPU/MPS parity and all-padding finite-output diagnostics.
4. Real training, checkpoint/resume, caching, standalone export/inference and saved History reopening.
5. Public capability discovery, six configurable controls, themed palette/ports, bundled example and explanation connecting spatial selection, shape arithmetic and sparse gradient routing. Native controls/code-before-run/Results/restart verification. Guided/scored curriculum and owner ART remain separately tracked.

## 2026-10-07 — Configuration foundation

Tests first: Python failed missing module (.local/maxpool2d-contract-red.log); Swift failed missing configuration API (.local/maxpool2d-native-red.log). Added paired configuration wrappers around the reviewed MaxPool1d axis contract. Native canonical values retain nullable stride; both implementations annotate errors with the spatial axis. No runtime registry admission yet.

Worker38tests passed, zero skips (.local/maxpool2d-contract-green.log), including existing 1D contract regressions and144asymmetric 2D cases compared to actual Torch output geometry or refusal. Native7Coretests passed, zero failures/skips (.local/maxpool2d-native-green.log), including 1D regressions. Last accepted packaged candidate3d86660 remains unchanged and quit.


## 2026-10-07 — MaxPool2d runtime, gradients and export

- Added shared pooling resolver/function dispatch for MaxPool1d/2d, paired schema2 semantics, exact authored-shape guards, float values/int64 indices layouts and exported named tuple assignment. Registry field validation now honors explicitly nullable integer lists, required for paired stride. Component remains in a private test registry until native admission/discovery/controls are delivered.
- Tests-first4failures (.local/maxpool2d-graph-red.log) before execution support. Final73Python checks passed, zero skips (.local/maxpool2d-runtime-final.log), including both actual MPS modes, CPU/gradient/export parity against an independent rectangular-window/flattened-position oracle, ties/overlap, noncontiguous tensors, nullable stride, disabled/incompatible indices refusal, all-padding infinity rejection, real3epoch versus1+resume exact model/optimizer/RNG, durable activation samples and preview cache reuse. Includes MaxPool1d/product/preview regressions; one pre-existing nested-tensor warning.
- Next: native component/typed graph integration, nullable-pair controls, saved capture verifier admission and runtime discovery, then example/Learn and packaged acceptance. Last accepted candidate3d86660 remains unchanged/quit. Worker changed; future packaged candidate requires refreshed runtime integrity. All test handles terminal; broad production gates unchanged.


## 2026-10-07 — Native MaxPool2d graph, optional pairs and captures

- Added reviewed six-field native component declaration, explicit spatial tensor ports, graph migration/shape/type validation and disabled-indices port availability. Tests use an appended private registry; public runtime admission remains pending.
- Extended optional field contract to integer lists: strict paired bounds, null defaults and valid enable values [1,1]. Existing registry-driven controls already dispatch optional fields to the integer-list editor; compilation and contract tests passed, native interaction remains unverified.
- Missing-component RED .local/maxpool2d-native-graph-red.log; final20Core checks passed, no skips (.local/maxpool2d-native-fields-final.log), including spatial values/int64 indices, nullable stride, bridge declaration, disabled output refusal, optional field serialization and existing choice/derived-binding/1D regressions.
- Named capture RED reproduced the rejection of spatial pooling outputs (.local/maxpool2d-capture-red.log). Extended reviewed pooling admission to MaxPool2d, retaining exact output sets, matching shapes/dtypes and privacy policy. Final11App checks passed, no skips (.local/maxpool2d-capture-green.log), including presentation lifecycle regressions.
- Next public runtime discovery/registry integration, examples/learning and packaged controls/readouts. Candidate3d86660 unchanged/quit; no native MaxPool2d acceptance yet. All test handles terminal; production gates unchanged.


## 2026-10-07 — Public MaxPool2d registry and discovery

- Added the reviewed paired-field component to the shared registry; native/worker graph tests now use that actual registry. Added an exact independent runtime descriptor identifying spatial axes, flattened spatial indices, bounds, null stride and value/index types. Native runtime refresh/probe/reset/count/admission now requires it, preserving unrelated older capabilities when absent or altered.
- Tests-first missing worker descriptor and native API failures (.local/maxpool2d-discovery-red.log, .local/maxpool2d-discovery-native-red.log). Public worker graph/discovery9passed including actual MPS, no skips (.local/maxpool2d-public-worker.log). Native11checks passed (5Core/6App), including actual worker probe and descriptor mutation/reset (.local/maxpool2d-public-native.log).
- Updated the explicit registry inventory expectation for the new component. Broader graph/choice/optional-field regression70Corechecks passed, no skips (.local/maxpool2d-registry-regression.log).
- Next bundled example, explanatory learning and themed component presentation, then refreshed packaged runtime/native acceptance. Last accepted packaged candidate3d86660 unchanged/quit. All handles terminal. No scoped MaxPool2d completion or broad gate acceptance yet.


## 2026-10-07 — MaxPool2d example and learning entry

- Added a prepared universal-canvas example: two measurements→40 learned features→[N,2,4,5]→window[2,3]/stride[1,2]→[N,2,3,2]→ReLU→12features→two class scores,146parameters. Both indices modes compile and survive workflow roundtrip; layout checks keep nodes separated. Connected the capability example reference and Learn launch action.
- Explanatory entry covers paired settings, independent floor/ceiling geometry, negative-infinity padding, flattened spatial indices, a worked3×4grid with winners and accumulated gradients, sparse selection y=Ax/Aᵀg, ties, information loss and capture privacy. Explicitly distinguishes constructed tabular features from actual images. This is an explanation/runnable exercise, not a scored course completion. Reused the established teal pooling glyph family with a distinct2Dlabel/purpose; owner ART remains separate.
- Missing-example API RED .local/maxpool2d-example-red.log. Final3Core checks passed (.local/maxpool2d-example-green.log). Actual example integration passed1test without skips (.local/maxpool2d-example-integration.log): generated code before training, real3epoch prepared training, live/saved int64 samples, verified History, construction/binding staleness and retired-host refusal.
- Next refreshed runtime/package and native paired-controls, connector toggle, Results/History, export/inference/reopen walkthrough. Last accepted packaged candidate3d86660 remains unchanged/quit. All handles terminal; no native MaxPool2d acceptance or production-gate closure yet.


## 2026-10-07 — MaxPool2d packaged scoped acceptance

- Built candidate17e405c with refreshed .local/maxpool2d-runtime-20261007. Initial refresh redundantly called watchdog build after copy_worker_sources (which already builds it); the helper refused existing output and initial package integrity refused the unsealed copy. Refreshed its integrity without rebuilding and verified matches-current-source; final package succeeded (.local/maxpool2d-package-final.log). No app/runtime policy bypass.
- Native1440×900: Learn disclosure fully rendered its spatial math, worked indices/gradients and caveats. Opened bundled example. All six controls visible. Nullable stride toggled off to Not set and on to [1,1]; edited/committed [1,2]. Indices toggle changed canvas labels in/out versus in/out/indices. Code-before-run displayed syntax highlighting, rank4 exact-shape guard and named max_pool2d tuple outputs.
- Actual native run6FC91C6E-1CD2-4A0B-A0BF-A144677E3868, request4b3f6895ecd11fe94f20056171270ee1a5a837a6457468fff83bca2c97922ad6, revision5, started2026-10-07T14:27:19.543Z, finished14:27:25.172Z. Three epochs,30verified artifacts,6computed/0reused preparation stages. Pool node/wires reported Captured. Bottom Results showed indices[8,2,3,2]int64 [6,4,6,12,16,12,2,2] and out float32 [0.614,1.539,0.614,0.223,1.071,0.223,0.887,0.887]. Selecting values wire displayed only out.
- Verified History samples matched. Reviewed/exported trained package to .local/KwipML-model-6FC91C6E. Standalone inference.py --example --batch-size2 exited0: logits[[-0.1133235767,-0.2415321916],[-0.8001078367,0.5898491144]], classes[0,1]. This is synthetic prepared-input inference; package excludes raw preprocessing, as its review states.
- Saved project, quit, relaunched and resumed from Projects. Reverified the same History run and exact value/index samples after restart. Quit again; process absence checked.
- Acceptance mapping: strict settings/native-worker shape parity→configuration tests; exact optional output types→graph/bridge/capture tests; numerical semantics→independent spatial oracle, gradients, noncontiguous input, CPU/MPS and all-padding checks; lifecycle→real checkpoint/cache tests and native training/export/restart; product→paired controls, connector toggle, code-before-run, bottom Results/History and explanation/example. Scoped delivery accepted. This does not close TORCH-01, owner ART, scored curriculum or any broad production/distribution gate. Next inventory unit: MaxPool3d.
