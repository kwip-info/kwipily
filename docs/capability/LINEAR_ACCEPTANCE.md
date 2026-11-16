# Linear capability acceptance proposal

2026-10-05. Documentation review only. Capability `pytorch.torch.nn.Linear`, catalog `0.1.0`, remains proposed and incomplete. This document neither changes the denominator nor attests production completion. Owner: Trevor Ewert / KWIP. Implementation follows the current native milestone gate. The investment is depth in an existing component and a reusable backend acceptance pattern; it defers new operator breadth until one complete evidence package exists.

## Reference and version identity

The installed development interpreter reports Python **3.13.6**, PyTorch **2.14.1**, arm64, with MPS available. This is a local identity probe, not an execution/device parity suite or clean-machine acceptance. `requirements-runner.txt` pins torch 2.14.1. Native runtime probing must continue matching the packaged manifest before launch.

The official [PyTorch 2.14 Linear reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html), inspected on 2026-10-05, describes an affine map with optional bias and preservation of leading input dimensions. Its documentation version is **2.14**, not a patch-specific 2.14.1 guarantee. The installed constructor was independently inspected: `(in_features: int, out_features: int, bias: bool = True, device=None, dtype=None)`. Installed `torch/nn/modules/linear.py` SHA-256: `1961ad1a1f650e104b3c7f78d19af2d21bf977e7af6c3fdf2420aad563c9e202`.

Local source inspection confirms weight shape `[out_features,in_features]`; disabled bias registers `None`. Default reset initializes weight and enabled bias within bounds determined by input width. This corroborates the inspected constructor/reset behavior, not all framework API or binary compatibility. The catalog's minor-documentation/patch-runtime alignment remains an explicit review item.

## Proposed bounded product contract

This is the next acceptance slice, not a proposal to silently reduce the framework-wide objective. Keep broader ranks, dtypes and devices visible as unimplemented domains. Before any catalog scope change, record its versioned diff and approval independently of this document.

| Surface | Current behavior | Required next slice |
|---|---|---|
| Configuration | Only `in_features`, `out_features`; integral numeric values 1–512 | Preserve widths; add explicit boolean bias setting, default true for migrated old documents |
| Ports | One `in`, one `out`; rank-2 float32 tensor | State `[batch,in_features]` → `[batch,out_features]`; preserve row order and batch size |
| Bias | Implicitly true in runner and generated code | True and false must change parameter ownership, count, export and checkpoint identity correctly |
| Parameters | Each node owns independent weight/bias under stable ModuleDict keys | Preserve keys; bias false omits bias key; no implicit parameter sharing |
| Devices | Worker CPU/MPS selection | Validate actual availability; record requested/resolved device; no silent explicit-device fallback |
| Dtype | Existing data path creates float32 | Explicit float32 contract; reject unsupported tensor dtypes before execution rather than casting silently |
| Application task | Two input features, two output classes; intermediate widths may vary | Retain application limits; accepting a Linear width does not imply arbitrary dataset/model-task support |
| Resource bounds | 128 nodes, 256 edges, 2m parameters; registry widths 1–512; intermediate tensors bounded by graph checks | Bias-aware parameter count and preallocation checks; do not raise limits incidentally |
| Initialization | Fresh run initializes with model seed; trained weights are distinct | Same seed/device/version behavior tested; no cross-device bitwise promise |
| Mode | Linear has no dropout/running statistics | Train/eval output equality for fixed weights; training still creates gradients and optimizer state |

Proposed rejection domains: scalar/rank-1/rank-3 inputs at this adapter boundary; wrong feature width; sparse/ragged/nested/quantized tensors; float16/bfloat16/float64/complex/integer inputs; CUDA/ROCm/meta device requests; zero or excessive widths; malformed boolean/configuration; nonfinite input/weights under a finite-data execution policy. These are product limits, not claims about PyTorch limitations. Finite-value and tensor-domain enforcement must be tested in the adapter: graph configuration validation alone cannot prove runtime tensor properties.

## Migration and cross-cutting work

The registry currently represents numeric fields only. Do not encode a boolean as an undocumented width field. Decide a shared typed configuration representation with native Codable/inspector, Python validator and migration fixtures together. Old graphs lacking bias retain true; newly versioned documents persist explicit bias. Reject unknown fields and malformed values; migration must not globally weaken strict validation. Preserve integral-number compatibility for existing width JSON.

`BuiltInCapabilityBridge` explicitly accepts only the existing two Linear fields and phases initialize/forward/export. It must evolve alongside the schema rather than merely accepting an extra registry field. Runner construction, parameter accounting, generated code, graph hashing, composite snapshots, preview signatures, checkpoint comparison and learning graph construction all need coordinated review. A bias change invalidates dependent activation caches and exact-resume compatibility. Do not relabel an old bias-enabled checkpoint as compatible with a bias-disabled graph.

Generated code must explicitly represent disabled bias, retain readable aliases and original ModuleDict parameter keys, and load matching legacy bias-enabled state dictionaries. Exporting code is not executing/training a model. Changing bias must be undoable and survive save/reopen; temporary inspector text cannot mutate a valid configuration accidentally.

## Minimum numerical and boundary test matrix

The following are required tests, not recorded passing results. CPU and actual MPS cases are separate. An unavailable-device skip is not device acceptance.

| Cases | Required oracle / observation |
|---|---|
| Bias true/false × `(in,out)` = `(1,1),(2,3),(3,2),(512,512)` × batch 1, 7 | Independently assigned parameters; compare forward to `X @ W.T + b` (omit b when false); correct shape/count/state keys |
| Same cases on CPU and MPS float32 | Same-device reference parity; initial tolerance proposal `rtol=1e-5, atol=1e-6` CPU and `rtol=1e-4, atol=1e-5` MPS; report maximum errors and justify any revised tolerance |
| CPU and MPS backward, nonsquare shapes, noncontiguous dense input | For arbitrary upstream gradient G compare dX=GW, dW=G.T X, db=sum(G,axis=0); disabled bias has no parameter/gradient |
| Branched fan-out/add/concat | Shared activation receives sum of downstream derivatives; independent node parameters remain independent |
| Seed/reset | Same seed and exact environment gives reproducible initialization; fresh reset does not silently preserve learned state; reset requires explicit lifecycle action |
| Bounds and types | 0, -1, 513, fractional widths, bool widths, NaN/Inf, string/numeric/null bias, missing/extra fields; migrations separated from malformed new-document rejection |
| Tensor/resource rejection | Wrong rank/width/dtype/device/layout, nonfinite data, parameter-limit crossing; reject before allocating model parameters where knowable |
| Export/persistence | Both bias modes; duplicate/code-like labels; output and gradient parity; correct state-dict round trip; legacy enabled-bias keys preserved |
| Training/evaluation | Real bounded classifier improves on fixture; fixed partition and seeded recipe; evaluation makes no optimizer/parameter changes |
| Checkpoint | Uninterrupted versus CPU exact resume; fine-tune fresh optimizer; incompatible bias/width/runtime/partition rejected; original artifacts immutable |
| Cancellation/failure | Cancel before/within execution and around artifact commit; no completed manifest or success event; foreign output artifacts preserved |

Do not require every cross-product to be a separate full training run. Primitive numerical cases are cheap; lifecycle fixtures should target distinct state transitions. Add meaningful failing tests before implementation. Record actual red/green outputs without treating this proposal as tests.

## Lifecycle adapter boundary

The next adapter should expose explicit operations over validated immutable requests, using the existing worker process and artifact protocol. It need not introduce a new executable plugin system.

- **Discover/validate:** exact backend/package version, supported operation modes/configuration, devices/dtypes, shape/resource limits and actionable node/port failures. Validate before model allocation.
- **Build:** private per-node parameters, deterministic initialization recipe and semantic configuration identity. Return parameter inventory without claiming trained state.
- **Forward/predict/evaluate:** distinct inference mode, bounded observations, output schema and weight provenance; evaluation consumes declared held-out partitions without fitting transforms or updating parameters.
- **Train:** owning experiment controls Adam/loss/batches/RNG, rather than pretending a Linear layer independently owns an optimizer. Report actual weighted metrics and counts.
- **Checkpoint/resume:** preserve the existing explicit CPU exact-resume guarantee; verify graph, data partitions, optimizer/runtime and RNG/epoch state. Fine-tune is a separate weights-only start, available on supported CPU/MPS. A descriptor must not generalize this to every optimizer/device.
- **Export:** readable standalone architecture plus exact state/configuration identity. Graph-only export requires no dataset or training. No claim of ONNX, TorchScript or torch.export support from Python source generation.
- **Failure/cancellation:** stable structured cause/node/context where available; unsupported contracts distinguished from process failure. Completion is staged until successful worker exit; partial/cancelled results never become successful model artifacts.

## Evidence gate and next decision

Both proposed training/inference modes need reviewed configuration, execution, export, validation, diagnostics, lesson, example, UI, persistence and failure-handling evidence with exact catalog scope and file hashes. The affine lesson can supply part of conceptual evidence after native acceptance, but does not prove bias-disabled behavior, general training proficiency, or every mode. Diagnostics must avoid unsupported gradient/learning claims when required measurements are absent.

Smallest next implementation: typed bias configuration plus backward-compatible migration, runner/count/export/checkpoint/cache integration and focused numerical fixtures, followed by native edit/save/reopen/export UAT. Then bind reviewed evidence and formal adapter lifecycle tests. Keep all catalog units incomplete until their full gate is satisfied; this document alone cannot earn completion credit.
