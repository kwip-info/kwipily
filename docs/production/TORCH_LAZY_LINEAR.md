# TORCH-01 · LazyLinear delivery contract

Status: scoped local delivery accepted, including packaged native example/edit/train/export/reopen evidence. Full framework coverage and production gates remain open. Owner: Codex/root. Bilinear scoped delivery is closed; this is the next linear-family unit, not completion of TORCH-01.

## Authored and inferred state

Expose output features (integer 1–512) and Boolean bias. Infer input features from the connected tensor's last sample dimension, preserving its batch and leading sample axes. The inferred width is derived evidence, never an editable or silently added configuration key. Bound inferred width at 4096 and resolved parameters at two million, counting bias, before allocation. Existing float32 CPU/MPS product execution boundaries apply; wider framework dtype/rank behavior must not be claimed from those bounds.

Changes upstream can change inferred shape and invalidate model state. A previously materialized layer cannot silently resize or replace its learned weights. Checkpoint compatibility includes authored graph and resolved shape evidence; reuse/resume must fail on incompatible width. Preserve old non-lazy checkpoint ordering and semantics.

## Initialization and transfer

Use actual nn.LazyLinear semantics rather than presenting an eager Linear under a lazy label. Eager modules initialize during construction; lazy parameters initialize afterward. Do not promise seeded equality between an otherwise equivalent all-eager model and a mixed model. Materialization must occur on the requested device/dtype before optimizer creation, parameter scans, gradient hooks and saved-state shape comparison. A bounded materialization pass must not consume user data or update unrelated stateful layers. Prefer visiting only lazy layers in validated execution order using small shape probes; verify this against the pinned backend before acceptance.

Generated architecture source must preserve lazy authorship and clearly explain materialization before optimizer creation. Exported trained state must load without random reinitialization. Validate actual saved weight shapes against inferred graph layout rather than relying solely on module metadata. The pinned 2.14.1 probe restored in_features correctly after forward. Never serialize uninitialized parameters as a completed model. Test exact CPU resume, incompatible-width rejection, fine-tuning, standalone reload/inference and RNG behavior explicitly.

Primary references: [LazyLinear](https://docs.pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html), [LazyModuleMixin](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html). Local pinned-runtime probe: `.local/lazy-linear-runtime-probe.log` checks both bias modes, deferred initialization, first-forward materialization, initialized-state reload and no RNG draw on reloaded forward. This probe is design evidence, not KwipML runtime support.

## Acceptance before local closure

Strict paired Swift/Python contracts; graph dtype/layout/trainability checks; deterministic registration and explicit materialization ownership; CPU/MPS numerical and gradient parity; generated source and trained export; actual lifecycle/cache/resume evidence; runtime capability discovery; informative configurable native node; example and learning material about inferred dimensions, initialization and parameter ownership; scoped native authoring/run/history/export verification. No palette availability until runtime integration is verified. Broader production gates remain separate.

## Configuration foundation evidence · 2026-10-07

Paired strict Swift/Python authored configuration and allocation-free shape resolution implemented. Python RED10missingmodulefailures; GREEN32tests including Bilinear contract regressions. Swift REDmissingAPI; initial GREEN corrected a test string literal to explicit JSONValue.string, then7tests pass (3LazyLinear4Bilinear),0skip/failure. Logs `.local/lazy-linear-contract-{red,green}.log` and `.local/lazy-linear-swift-{red,green}.log`. This does not expose a registry component or execute LazyLinear within KwipML yet.

## Graph and deferred runtime evidence · 2026-10-07

Swift graph integration supports inferred sample prefixes, float32 gating and aggregate trainable parameter accounting. Initial RED/GREEN exposed an invalid private Boolean field definition; corrected its nil bounds and replaced an invalid wide Input overflow fixture with legal projection/concatenation. Re-ran RED with only production integration removed: four tests/10 assertion failures (`.local/lazy-linear-graph-corrected-red.log`). Restored implementation:11 Swift tests pass, no skips/failures, including Bilinear regressions (`.local/lazy-linear-graph-accepted.log`).

Python RED9 failures/10 passes on private-registry tests; GREEN56 tests pass with no skips, including actual MPS execution and eager Linear/Bilinear regressions (`.local/lazy-linear-runtime-{red,green}.log`). Real nn.LazyLinear remains uninitialized at construction; explicit materialization probes only lazy layers on current dtype/device in validated order, preserves authored config and is idempotent without additional RNG draws. Higher-rank output/gradient parity, safe module keys, and standalone source retaining lazy authorship before/after initialization pass. Initialized-state reload forwards without random reinitialization. Product training/preview/frozen loading are the next integration checks; no palette or native acceptance implied.

## Lifecycle evidence · 2026-10-07

Training materializes before optimizer/parameter capture; preview before weight hashing/loading; frozen loading within its RNG-preserving context before expected-shape checks. Preview rejects incompatible state shapes before loading. New lazy-containing checkpoint schedules record lazy initialization and all parameter-node registration order; existing nonlazy contract keys remain unchanged.

Lifecycle RED7 failures (`.local/lazy-linear-lifecycle-red.log`); graph/lifecycle GREEN28 passes (`.local/lazy-linear-lifecycle-green.log`). Independent review found arbitrary first-forward width could bypass inferred allocation bounds in raw/exported models. Added two failing tests (`.local/lazy-linear-forward-width-red.log`) and width guards before lazy invocation. Both remain uninitialized on refusal.

Combined acceptance `.local/lazy-linear-lifecycle-accepted.log`:137 Python tests pass, no skips. Coverage includes both bias modes, exact CPU full-vs-resumed state/optimizer/RNG/metrics, gradient inventory, fine-tune reset, immutable parent, standalone reload, frozen RNG preservation, initialized/trained/cache reuse, incompatible architecture/state refusal and legacy checkpoint/preview regressions. One existing Embedding MPS renorm CPU-fallback warning, not a LazyLinear failure. Additional actual MPS train/initialized-preview/trained-preview check passes (`.local/lazy-linear-mps-lifecycle.log`:1pass,7deselected); MPS contract remains weights-only, no bitwise resume claim.

Shared registry, capability discovery, native controls, example/learning material and packaged native acceptance remain pending. All verification handles terminal; packaged candidate unchanged.

## Shared discovery and component integration · 2026-10-07

Added strict versioned lazyLinear worker descriptor with inferred input/output bounds and validatedShapeBeforeOptimizer policy, matched by exact bounded Swift decoding. Missing/altered/extra keys fail, and optional unavailable or invalid discovery clears stale support without blocking legacy graphs. Actual native worker probe installs support. Shared registry exposes Lazy Linear in Layers with only output width and Boolean bias; bridge input width is an inferred symbol, never an invented configuration field. Existing rose Linear icon family is reused pending owner ART pass.

RED: missing Swift APIs and Python descriptor; worker-emission test separately failed missing key. Initial GREEN caught inconsistent display title between declaration and JSON; standardized Lazy Linear, corrected title assertion. Final `.local/lazy-linear-discovery-final.log`:30 Swift tests pass0skip/fail (22Core8App), including actual worker discovery and existing native-boundary training regression. `.local/lazy-linear-catalog-green.log`:31 Python tests pass0skip/fail, including shared-registry execution/lifecycle. No rendered native acceptance yet. Prepared example and learning explanation are next; no nonexistent lesson/example IDs advertised.

## Prepared example and learning explanation · 2026-10-07

Bundled inferred-width example uses eager6 → reshape[2,3] → LazyLinear4 → flatten8 → LazyLinear2, including both bias modes in factory/compiler fixtures. Shared workflow source, split/fitted preparation, controls and history remain the existing modular path. Capability links actual prepared-lazy-linear-classifier example; no dedicated graded lesson is advertised.

Learn includes the runnable entry plus an expandable explanation of the affine sum, shared parameters across sample positions, 52 default parameters, initialization order and upstream-width experiments. It explicitly explains that these composed affine projections alone do not create nonlinear decision boundaries. Owner ART and selected first-release LEARN acceptance remain open.

RED `.local/lazy-linear-example-red.log` missing factory/store APIs. Initial GREEN reached completed training/verified history but exposed an outdated helper expectation for parallel-example provenance; added the new example to the existing nonparallel branch. `.local/lazy-linear-example-accepted.log`:17 Swift tests pass0skip/fail (14Core3App), including actual code-before-run, preparation, three training epochs, verified immutable history and saved source/capture checks. App integration3.117seconds. No rendered UI observation claimed. Next build a fresh canonical worker payload and replay the native example, controls and export.

## Packaged native acceptance · 2026-10-07

Built fresh canonical worker payload `.local/lazy-linear-runtime-f75eecd`, retaining the pinned interpreter/dependencies and replacing worker/registry/bootstrap plus rebuilt native watchdog. Verified source integrity before/after packaging and local signing (`.local/lazy-linear-package.log`). Candidate f75eecd.

Observed Learn’s expandable formula/52-parameter explanation and opened the inferred-width example. Native Build displayed first input[N,2,3], first output[N,2,4], flattened input[N,8], output[N,2]. Only output features and Learn bias were editable. Changed4→5: downstream[N,2,5] and[N,10] updated; Undo restored4/8. Disabled bias: untrained Code showed nn.LazyLinear(4,bias=False); Undo restoredTrue. Saved before running. One stale AX index after Undo produced elementHasNoFrame; refreshed inspector and continued without duplicate edits.

Actual packaged run **91B02E10-47D5-4357-8B7B-7A0FED041895**, request **ee81498a4002709edf9f790dcd81747a0645e0118781f90367452b2e314c0aa8**, started2026-10-07T09:21:51.804Z, finished09:21:57.012Z. Three epochs completed; History verified29artifacts,6preparationnodes computed/0reused, validationloss0.13062 and100%accuracy on this small fixture (not a generalization claim). Native trained export saved `.local/KwipML-model-91B02E10`; input-preparation exclusions disclosed. Independent development Python ran saved inference.py --example --batch-size2 outside app: finite logits and classes[1,1],exit0.

Found Code caption incorrectly said all weights initialize at construction. Corrected to distinguish eager construction and lazy shape initialization, commit8b4d196. Rebuilt with source-verified sameworker (`.local/lazy-linear-caption-package.log`); reopened saved example through project launcher, observed corrected caption/biasTrue generatedcode, reopened History and verified same immutable run. Candidate quit; no running test/build handles. Current-architecture Code keeps its existing Last trained button disabled for workflows; trained artifacts are accessed through History → Saved model source.

Closed scoped LazyLinear delivery. This includes explanatory example material, not a new graded lesson or full production evidence-complete numerator. Owner ART, selected release curriculum and broader native/accessibility/platform gates remain separate. Next inventoried family begins with Conv1d contract/shape/state review.
