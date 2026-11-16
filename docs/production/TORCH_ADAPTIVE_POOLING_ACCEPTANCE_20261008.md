# AvgPool2d and adaptive pooling acceptance · 2026-10-08

Frozen finite acceptance for `pytorch.torch.nn.AvgPool2d`, `pytorch.torch.nn.AdaptiveAvgPool1d`, `pytorch.torch.nn.AdaptiveAvgPool2d` and `pytorch.torch.nn.AdaptiveMaxPool2d`. Each owns forward, input-gradients and shape-contract modes. These stateless, parameterless blocks rely on surrounding trainable layers for checkpoint, continuation and trained export evidence. Framework-wide and release acceptance remain separate.

## Reviewed contracts

| Component | Configuration | Independently assessed lesson |
|---|---|---|
| `avg_pool2d` | Paired kernel, nullable paired stride, paired padding, partial final windows, padding-count policy, nullable signed nonzero divisor | `assessment.avg-pool2d-windows` |
| `adaptive_avg_pool1d` | Output length | `assessment.adaptive-avg-pool1d-bins` |
| `adaptive_avg_pool2d` | Independent nullable output height/width | `assessment.adaptive-avg-pool2d-bins` |
| `adaptive_max_pool2d` | Independent nullable output height/width; optional winning indices | `assessment.adaptive-max-pool2d-bins` |

Schema-2 production uses dense float32, a separate nonempty batch, sample rank2 for adaptive 1D and rank3 for 2D maps. Channels range1–512 and spatial/output sizes1–4096. Allocation is bounded at100million input/output elements; optional int64 indices count toward combined output allocation. Unsupported dtype/layout/rank, booleans, fractional/nonfinite or invalid settings, unknown fields and nonfinite evidence refuse. Execution is out of place with no learned weights or unpooling claim.

AvgPool2d shares reviewed MaxPool2d geometry with dilation1. Null stride follows kernel. Divisor override accepts signed integers from−2147483647 to2147483647 except0; null uses actual window counts. Ceil-mode tail windows trim at input+padding and count their intersection with the padded domain, including a genuinely partial last window. Adaptive bins use floor(start×input/output) through ceil(end×input/output), allowing overlap and larger outputs; null axes preserve their input size. Maximum ties select the first row-major flattened spatial index within a channel; indices are integer observations, not differentiable outputs.

Actual PyTorch2.14.1 MPS cannot directly execute some uneven adaptive-average bins. An explicit shape predicate performs those averages on CPU through differentiable transfers, then returns values and input influence to MPS. It does not catch arbitrary runtime failures. Started/completed evidence and saved training metadata retain node/operator/compute/return/reason disclosure; Results displays a bounded node-linked warning. Divisible averages and tested adaptive maxima use their direct path. This acceptance does not claim every operation computes natively on MPS.

## Actual tests and independent evidence

Final new Python slice:473 passed, no skips (271 strict configuration,79 real engine/export,56 independently graded lesson,67 authoring). Real CPU/MPS values and weighted influence are checked against independent scalar windows/bins, including partial divisors, signed overrides, nondivisible/kept/all-null axes, upsampling, exact tied winning positions, lifecycle, portable code and strict failures. Prior replay covers55 bound Python files:3,048 passed, no skips, with three existing documented warnings. All27 prior semantic bindings and16 frozen receipts remain unchanged apart from source-SHA reconciliation after actual replay.

Final affected Swift regression:189 tests,188 passed, one existing opt-in validation-timing benchmark skipped, zero failures (96Core,93App). Scoped38 checks had no skips; all four real guided/practice/private workers and fresh Workspace replay pass. Shared admission checks cover204 scenarios across17 operators, training/experiments and six dispatch/cancellation boundaries. The final regression enables the existing opt-in LayerNorm real runtime test. Logs distinguish obsolete exact count/title failures, genuine lesson-search and artwork mapping failures, and their repairs; prior semantic assertions are retained. `component_bundles.py --check` and `git diff --check` pass.

A separate read-only MPS probe checked eight adaptive kept-axis/all-null/upsampling variants against CPU, independently tied indices and exported Model: forward discrepancy≤1.1921e-7, input-influence discrepancy≤8.9407e-8; same-device portable parity exact0. It supplements rather than substitutes for production test cases. [Review](TORCH_ADAPTIVE_POOLING_CLOSURE_REVIEW_20261008.md).

## Lessons, artwork and app observations

All four approachable no-code lessons teach practical regions, sizes, observations and how a small input change affects the result. Required controls and canvas connections are explicit; optional technical reasoning is separate. Independent private fixtures and receipt identities are preserved. Assessment arithmetic is float64; production remains float32. Each component has its own searchable authored example and assessed lesson, with original generated sage artwork shared between palette and canvas. Generation prompts, unedited source files and image hashes are in `docs/design/component-icons-v2/ADAPTIVE_POOLING_RECEIPT_20261008.md`.

Native AvgPool2d lesson project329DAC01-9578-4F7B-9F78-0027799A66A5 was constructed with Add actions and explicit ports. Window3×3, stride2×3, padding1×1 and partial-window/padding-divisor policies produce sample[2,4,3]. Optional divisor enables1. Valid wrong divisor1 measured and failed both output and input influence. Zero refused with a nonzero-integer message and disabled assessment;−2 accepted. Returning to null passed guided and private assessment; private mode withheld measured fixtures. Initial discovery readiness was transient.

Native prepared AdaptiveAvgPool2d project8F885F61-AEAB-45F4-A19B-707F15A06E80 ran three epochs with explicit MPS and output height3/width null. Logical runE624A9AA-217B-4DC2-99EE-EB52A02E507D, request877fe18b438b587f1334608f7631580c2b61b0a9c8dcef815734ae286d964efd, revision1. Activity displayed the pool node’s uneven-groups CPU warning. History verified30artifacts and six freshly computed preparation stages. Saved model source retained output_size(3,None), the reviewed device policy, checkpoint keys and input checks. The native trained export saved `.local/KwipML-model-E624A9AA` with source, weights, strict inference helper, prepared-input contract and hashes. Source rows and fitted preprocessing remain excluded; callers must reproduce preparation.

Independent scalar saved-weight replay across all16 held-out rows: score error≤1.8032765036757326e-7; all class indices match. Captured pool prefix error≤1.9201781803479534e-8 and output prefix error≤1.5018351851026068e-7. Only eight retained scalar values per stage are captured, not entire arrays. Strict exported Model/helper parity exact0; batch1/16 discrepancy≤1.1920928955078125e-7. CLI example succeeds. Actual saved metadata identifies pool computeCPU/returnMPS. [Saved export evidence](TORCH_ADAPTIVE_POOLING_NATIVE_EXPORT_20261008.md) records all package and archive hashes. Teaching-set accuracy does not establish generalization.

Representative native observations do not claim all four members were manually trained/exported; automated actual-worker and portable checks cover all four. Final release theme, accessibility, clean-machine distribution and owner UAT remain open.

## Final packaged confirmation

Canonical worker payload integrity, copied-worker identity, local signing and deep strict package verification passed. Final executable SHA256: `7c9e42626af351de0a23180444eb834c34664656fb0e3bf2731ad81bdd7121d0`. Log `.local/adaptive-pooling-final-packaged-build.log`. No distribution signing or App Store submission is claimed.

Cold quit/relaunch of this final candidate restored the native AvgPool2d window/stride/padding settings, empty divisor and private receipt. Results displayed “Saved assessment verified from its task, submission and measured evidence” with both criteria passed. The final native Pooling ribbon displayed all four original generated icons beside the accepted four, and the AdaptiveAvgPool2d canvas used its matching artwork. Native Learn search “adaptive pooling” returned the assessed four-member lab and all three adaptive examples after the shared category/keyword repair.

All declared finite component checks are satisfied. Source-bound accounting must still preserve the prior27 units and16 receipts while binding these four to this frozen receipt; its result belongs in the separate accounting report. This receipt is immutable after creation. Future units, DATA-02, remaining curriculum and final UAT/DIST retain their own gates.
