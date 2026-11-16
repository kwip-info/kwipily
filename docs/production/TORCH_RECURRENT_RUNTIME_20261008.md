# Recurrent runtime implementation evidence — 2026-10-08

Scoped implementation proof for `rnn`, `lstm`, `gru`, `rnn_cell`, `lstm_cell`, `gru_cell` under `TORCH_RECURRENT_CLOSURE.md`. Baseline scheduler commit `9d6b199` accepts 59/88 capabilities. This receipt grants no acceptance credit and does not represent native application UAT or a distribution gate.

## Actual terminal results

Host `.venv/bin/python`, installed Torch 2.14.1, macOS arm64 CPU and actual MPS. No signed bundled runtime, Swift build or native application operation was executed by this worker.

```sh
.venv/bin/python -m pytest -q runner/tests/test_recurrent_runtime.py
```

Final terminal: **215 passed, 0 failed, 0 skipped, 3.80 seconds**, `.local/recurrent-complete-final-green.log`.

```sh
.venv/bin/python -m pytest -q runner/tests/test_checkpoint_lifecycle.py runner/tests/test_frozen_checkpoint.py runner/tests/test_tensor_routing_runtime.py runner/tests/test_stateful_normalization_runtime.py runner/tests/test_optimizer_runtime.py runner/tests/test_scheduler_runtime.py
```

Adjacent accepted runtime terminal: **831 passed, 0 failed, 0 skipped, 10.06 seconds**, `.local/recurrent-adjacent-prior-green.log`. This is a bounded adjacent regression, not the final catalog-bound prior regression that root coordinates separately.

## Implementation and independent checks

Strict shared normalization, conditional input ports, observation-first named output layouts, native parameter order/count and resource estimates are implemented in `recurrent_configuration.py`. Graph semantics canonicalizes defaults, so empty and explicitly default configurations retain the same semantic identity. Actual authored contracts, generated callable tables and worker discovery are independently compared for all six published entries; genuine shipped registry publication contains 69 definitions. The generated worker table is imported in its real standalone module environment, with its own `tensor_layout.BatchedLayout`, rather than mixing incompatible package class identities in a test fixture.

GraphModel registers native learned modules through the existing collision-safe registration seam. Supplied sequence states cross the native layer/direction boundary explicitly; zero state uses native default initialization. Named results are evaluated once and every output is checked, including unconnected states. Cell `out` and `hidden` share the same tensor. Repeated inference has no retained hidden state; explicit finite graph connections carry hidden/cell results between nodes. Resource checks bound sources, combined physical outputs, conservative gates and work before recurrent execution. Projected LSTM refuses MPS before model construction/state mutation; portable forward repeats the actionable guard and never silently falls back to CPU.

Expected values use a plain Python scalar oracle, independently implementing RNN tanh/relu, LSTM input/forget/cell/output gates plus learned projection, and PyTorch GRU reset placement after the recurrent affine term including its bias. Weighted objectives assess result, hidden and cell outputs. Central finite differences independently measure every input, supplied state and learned parameter coordinate. CPU/MPS cases include all six, bias enabled/disabled, zero/supplied state, multilayer bidirectional sequences and batch-preserving genuine graphs. Numerical tolerances are bounded explicitly in tests; runtime-versus-portable forward and gradients require exact equality when executing the same saved module on the same device.

All six train and export through genuine schema-2 classifier graphs. Self-contained exported code imports no runner adapter, preserves native learned key order and named aliases, and independently checks strict state before copy. All six CPU models prove uninterrupted-versus-checkpoint exact model/optimizer/RNG state and metrics. Three multilayer dropout sequence modules prove training/eval behavior and CPU RNG continuation. Projected multilayer bidirectional LSTM additionally proves exact CPU AdamW and StepLR continuation, optimizer ownership/projection keys, optimizer update counts, scheduler/RNG state and byte-identical metrics/rate trace. Actual MPS training and fine-tuning cover all six; three multilayer bidirectional dropout models preserve the existing nondeterministic MPS policy and exact-resume refusal. Fine-tuning retains actual parameters and resets the established lifecycle state.

Negative controls reject malformed configuration, missing/disabled state ports, wrong shape/rank/dtype/batch, nonfinite input/parameters, incompatible device/projection and resource bounds before native forward. Strict composed state checks reject missing/extra keys, wrong shape/dtype, sparse and nonfinite tensors before copying any parameter. Preview named-state caches round-trip all six and resume from the captured flatten boundary. Explicit carry and strided input paths work on CPU/MPS and portable modules without input mutation or mutable state.

New checkpoint schedule metadata is conditional on recurrent graphs: recurrent execution order and real learned parameter order; stochastic execution order includes multilayer recurrent dropout together with existing dropout nodes. Old non-recurrent branches and lifecycle contracts are preserved. Recurrent layers add no persistent hidden buffers.

## RED → GREEN and honest fixture corrections

- Helper tests first failed because the new helper did not exist: **40 failed**; implemented shared controls/layouts gave **40 passed**.
- Independent native/scalar tests failed because the execution helper did not exist: **12 failed, 40 deselected**; execution implementation gave **52 passed** in the then-current suite.
- Genuine shipped-registry tests exposed missing canonical default storage in GraphModel: **12 failed, 130 deselected**; adding recurrent normalization to the existing canonical-storage set gave **12 passed, 130 deselected**.
- Strict-state tests found a real portable/direct module gap: Torch strict loading accepts a floating dtype conversion. New recurrent-only state admission refuses that before copying. The saved engine checkpoint path already refused wrong dtype; its negative test initially expected the wrong diagnostic substring. The two-failure RED log therefore combines one production gap with one fixture diagnostic mistake and is not evidence that old engine checkpoints lacked strict dtype protection.
- Earlier fixture-only failures used a nonexistent export function, omitted required Flatten/Linear fields, or imported the standalone generated table with package-layout classes. These were corrected using existing contracts; they did not weaken admission or expected numerics. Preserved attempt logs remain distinct from the final GREEN.

## Limits and remaining coordinated gates

This proof covers the frozen dense float32 family, explicit states and finite DAG carry. It does not claim packed/ragged sequences, mutable streaming state or projected MPS support. Native canvas/lesson/History/export checks, independently emitted canonical example proof, lesson private/cold grading, artwork, full bound prior regression and guarded accounting are separate root/team gates. No catalog, tracker, existing acceptance receipt or signed runtime was modified by this worker.

## Final source and log hashes

| Path | SHA-256 |
|---|---|
| `runner/recurrent_configuration.py` | `346ed5bb20d9a5e931946175ca3debd6356a075b2a8219153de798124aeac89c` |
| `runner/engine.py` | `6c92dfdde3fb1e5613f17d8121fd5a01bbceffdfae1c28eff061844b16670850` |
| `runner/graph_semantics.py` | `37f78ed56626fbf363e5371f0ce9cadb569e36fb52d3898ea1d7be49bf4872b9` |
| `runner/checkpoint_lifecycle.py` | `07c2c93e0fdcfab214b049a7f1bb63fb441600a879068ee89e105c9478c10e2a` |
| `runner/tests/test_recurrent_runtime.py` | `8bf5afb9a8d5fccae9852bf5a8fa2bf1bca27b87fff81e2eb1d71ce494d5ba9b` |
| `runner/recurrent_adapters_generated.py` | `c8a3bb65e7aa2f6f370358e8e88de5140128ffeda185fc45cfc837ffce5a13de` |
| `.local/recurrent-helper-red.log` | `a4b5ff1b8888291da7a34df9a448329044999475e0b1be90e6019a62bbd32cd4` |
| `.local/recurrent-helper-green.log` | `d2c88d7a588caa8f8f4113ccc3a22ba4687bf22f8594e1455c358cc62693baf4` |
| `.local/recurrent-native-api-red.log` | `d3fbe8c7b602d8e35fb8b65c945cd1360479d051bac0a2c7bb9c9a838a856789` |
| `.local/recurrent-native-green.log` | `ee70ca650db390c414b8bd90928b011e56a4a1ea43d04b7d452b24f59b953786` |
| `.local/recurrent-genuine-registry-fourth.log` | `fee5db46f4d3a16b54556dd62f07097c0bff45b377bd3789255cfccd07422d5a` |
| `.local/recurrent-genuine-registry-fifth.log` | `a72d534f86a64e21bf317dbaf07315e6ad8d8a46ef60835af53cf56f28a78b07` |
| `.local/recurrent-strict-state-red.log` | `9c6619cef93894765c3c217d914f4ff86c7da257cf5e9ec67fbfa91880317928` |
| `.local/recurrent-complete-final-green.log` | `bdd60407df0862c96ced097be574b9c5ee3da46891c7bdcc819e5cdab59637c5` |
| `.local/recurrent-adjacent-prior-green.log` | `974c34a4f5bf14efa7a512686502043d0442bdf01dd1b8a70fa4467e8f5ab093` |
