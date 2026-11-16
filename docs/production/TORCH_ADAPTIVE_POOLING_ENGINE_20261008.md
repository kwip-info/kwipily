# Adaptive and average 2D pooling: engine integration

Implemented scoped integration, 2026-10-08. This document records automated production-worker evidence; it is not native acceptance, curriculum closure or coverage credit. Owner root coordinates packaging/native checks and frozen acceptance.

## Changes

`runner/graph_semantics.py` adds schema-2 normalization for AvgPool2d and the three reviewed adaptive operators. `runner/engine.py` admits their static registered function IDs, generated shape resolvers and exact float32 graph types. Authored adaptive output-height/width/null settings remain in graph identity and model configuration; one `_pooling_kwargs` translation produces the framework `output_size` argument for execution and export.

The four new operators use `_execute_reviewed_pooling`: exact nonempty batched sample shape, dense float32 values, finite inputs and bounded logical input/output allocations checked before calling the pool function. The actual batch is bounded to 100M logical input elements and 100M combined output elements (float output plus optional index output). Stride-zero views cannot bypass the logical budget. Maximum indices retain the generic named-output route and exact int64 dtype. Pooling adds no learned state or checkpoint keys.

MPS adaptive averages use an explicit conservative shape predicate: if any input spatial extent is not divisible by the resolved output extent, compute the pooling on CPU and copy the result to the original device. Both copies retain differentiation. A RuntimeWarning states the CPU fallback and gradient return to MPS. There is no catch-all exception handler and no silent fallback for unrelated errors. Divisible adaptive-average bins and AdaptiveMaxPool2d use their normal device path. This records mixed device execution; it does not claim the unsupported MPS kernels execute natively. Actual training computes a bounded per-node disclosure once: `operationDevices` is included in started/completed events and the saved manifest runtime only when nondivisible MPS adaptive-average nodes require fallback. Entries identify node/operator, CPU compute, MPS return and the shape reason. Each node emits one schema-1 `warning` event with code `pooling.cpuFallback`, worker run ID and bounded message. CPU, divisible MPS and old graphs preserve the absent-field representation. Root wires this recognized warning to the native console; the manifest provides retained provenance independently of stderr.

Standalone generated Python includes the reviewed adapter source verbatim, plus standard-library math/warnings imports, rather than importing KwipML. It preserves boundary checks, null-axis resolution and the explicit device policy. Existing four pool paths retain their prior function/configuration/export behavior. New exported models load the original model state strictly and contain only the graph's learned projection/head keys.

## Actual TDD and verification

Interpreter `.venv/bin/python`, PyTorch 2.14.1, host CPU and available MPS. Initial production tests failed **26**, passed **4** before implementation: all new GraphModels were rejected as unsupported. Log `.local/adaptive-pooling-engine-red.log`. The four passes were legitimate schema-refusal controls, not missing-test skips.

After implementation and generated real bundle registration, first graph checks passed **30**, zero skips. Expanded independent tests exposed one test-builder error when a deliberately invalid null 1D size was used to calculate the fixture head width; the fixture was corrected to mutate an already valid graph, with no production relaxation.

Final new suite: **79 passed, zero skips**, `.local/adaptive-pooling-engine-green.log`. Combined actual new engine and prior AvgPool1d/MaxPool1d/MaxPool2d/MaxPool3d graph plus configurable-activation regression: **379 passed, zero skips**, `.local/adaptive-pooling-engine-regression.log`. No warnings were reported in this final pytest run; deliberate fallback warnings were captured and asserted. `git diff --check` for owned paths passed.

Exact commands:

```sh
.venv/bin/python -m pytest runner/tests/test_adaptive_pooling_engine.py -q
.venv/bin/python -m pytest runner/tests/test_adaptive_pooling_engine.py runner/tests/test_avgpool1d_graph.py runner/tests/test_maxpool1d_graph.py runner/tests/test_maxpool2d_graph.py runner/tests/test_maxpool3d_graph.py runner/tests/test_configurable_activations.py -q
```

Each member checks actual production graph construction, independent scalar outputs and weighted input gradients, stateless pooling state inventory, standalone generated Model strict state loading and exact CPU full-graph forward/backward parity. Max additionally verifies exact first row-major tie indices, typed output contracts and disabled-index stale edge refusal. AvgPool2d checks asymmetric windows/padding/partial boundaries and signed negative divisor, explicit zero refusal, and actual CPU/MPS full-graph parity.

Adaptive variants cover global pooling, nondivisible/overlapping bins, upsampling, independent kept axes/both kept, negative ties, return-indices false/true and full portable model execution. Boundary controls cover schema 1, bool/null/zero/fraction/string/infinity/unknown settings, runtime float64/sparse/empty/wrong shape/nonfinite tensors, and huge logical batch refusal before allocation/call. Available MPS checks execute all four full models and standalone exports, compare CPU values/gradients, assert fallback warnings only on the known unsupported nondivisible average shapes, and separately prove divisible bins emit no fallback warnings.

The structured training provenance addition had its own actual RED: **2 failed, 6 passed**, with the two MPS fallback runs missing retained policy; log `.local/adaptive-pooling-engine-provenance-red.log`. Its eight actual training controls then passed: both nondivisible MPS averages, both divisible MPS controls and all four CPU operators. Each fallback run persists exactly the same one-node list in started, completed, returned result and manifest, and emits exactly one warning event. Nonfallback runs omit both list and warning. No epoch-level disclosure arrays are produced.

## Remaining closure

This scoped integration does not verify packaging, Swift typed configuration, native inspector enablement/persistence, actual History training exports, lessons, original icon assets or catalog accounting. Those remain parent batch gates. Preserve prior 27 accepted units and frozen receipts. Expand an applicable old-worker regression only when root's batch changes or final binding criteria justify it.
