# Attention runtime evidence — 2026-10-08

Scoped implementation evidence for the five frozen attention units and the supporting `attention_mask` utility. This records host execution, not native UI acceptance, catalog credit, machine coverage or release readiness. Root owns final acceptance and the unchanged prior receipts.

## Finite delivered contract

Dense observation-first float32 CPU/MPS attention uses strict canonical controls, exact native learned parameter names/order/shapes, bounded tensor/score/mask/work admission and self-contained portable source. Masks are int64, one blocked/zero allowed, with exact observation geometry and no hidden observation broadcast. Combined masks must leave an eligible original source for every query, including when MHA appends extra positions. Utility mask includes configured padding columns and separately emits padding; rank-four reference head counts are bounded at 32. Utility state/gradients are empty and it receives no reviewed capability credit.

MHA supports unequal key/value widths, native learned/zero augmentations, optional returned weights and averaging. Every declared output is checked even if unused. Transformer variants preserve native parameter initialization/order and disable nested optimization. SDPA owns an empty-state module whose parent mode disables inference dropout, without invented parameters or gradient leaves.

### MPS positive-dropout policy

Installed public SDPA refused MPS training dropout for inputs without gradient tracking. This is a real native limitation; forcing the public math backend did not repair it. Approved runtime and portable source use device-local public `matmul`, blocked-score masking, `softmax`, `torch.nn.functional.dropout`, then `matmul`, exclusively for MPS training with positive dropout. CPU and other modes use public native SDPA. No CPU fallback, private primitive or fake gradient leaf is used. Scores/probabilities and all returned values remain checked for finite values.

Exact descriptor field `sdpaMPSDropoutExecution` is `deviceLocalPublicOps` for SDPA and `notApplicable` otherwise. Actual applicable runs additionally disclose ordered `attentionExecution` entries `{nodeID,operator,computeDevice,policy,phase}` on started/completed events and manifest.runtime. Values identify MPS/training/public-ops; no tensor contents are placed in this policy disclosure. Existing pooling operationDevices remains unchanged.

### Checkpoint and export

Only attention graphs add ordered `schedule.attentionNodeOrder`. Learned four modules join parameterNodeOrder; empty-state SDPA does not. Positive-dropout attention joins stochasticNodeOrder alongside existing recurrence/Dropout. No-attention contract projections retain their old fields. Exact CPU continuation restores model/optimizer/RNG and matches uninterrupted metrics and learned state. MPS explicitly supports weights-only fine-tuning and refuses exact continuation.

Export embeds the reviewed pure geometry/configuration source and checked execution helpers, with ordinary PyTorch imports and no repository import dependency. Runtime/portable composed state loading rejects extra/missing names, wrong shape/dtype, sparse or nonfinite values before copying any parameter, even when strict=False. SDPA separately rejects foreign owned state.

## Actual TDD and verification

Host command:

```
.venv/bin/python -m pytest runner/tests/test_attention_runtime.py runner/tests/test_recurrent_runtime.py runner/tests/test_checkpoint_lifecycle.py runner/tests/test_preview.py -q
```

Final terminal: **434 passed, zero skips, five upstream constructor warnings, 17.05 seconds**. This is 190 new attention checks and 244 adjacent prior checks, including all 215 accepted recurrent runtime tests. Log `.local/attention-runtime-adjacent-green-attempt.log` ended GREEN despite its historical attempt filename.

Meaningful RED milestones retained: missing configuration helper (13); missing executor (6); padding-union and mismatched native constructor checks (4); corrected MPS nongrad-input dropout (1 failed/1 passed); absent lifecycle order (5); missing live/manifest execution disclosure (1 failed/4 passed); unsupported utility head count reaching mask allocation (1). Initial 20 graph checks were blocked by unpublished real metadata and are not treated as semantic engine RED. All final graph checks use the actual published registry75 and actual generated adapter table, never a synthetic registry bypass.

Fixture mistakes are retained honestly: an initial scalar-gradient test incorrectly used tensor equality/list lookup; later dropout fixture treated scalar_leaves entries as index paths; early export fixture mistook a provenance literal for an import; lineage fixture used an invented nesting/mode; descriptor fixture incorrectly expected runtimeContract inside registry and a callable map; preview fixture omitted the actual outputs wrapper. These were test corrections, not product semantic failures. Intermediate failing logs remain local and are not reported as passed.

### Independent numerical evidence

Plain Python loops compute stable softmax, learned projections, residual paths, LayerNorm statistics, relu/gelu and all five attention operations. Coordinate central differences check every floating input and owned native parameter with a mixed weighted objective including returned MHA weights. CPU/MPS, both residual normalization orders and unequal lengths/value widths are exercised. Additional masked MHA cases cover heads, averaging, unequal widths, learned/zero augmentation and bias off; masked SDPA checks binary inversion and automatic/custom scale.

The MPS public dropout proof draws only an authoritative dropout multiplier from `F.dropout(ones)` at a captured seed, independently evaluates scalar attention using that multiplier, then checks all query/key/value central differences with the fixed mask. Both nongrad-input and gradient-enabled input cases are tested; nongrad input remains nongrad output. Runtime/portable RNG replay is verified on CPU and bounded numerical MPS replay is observational evidence, not an MPS exact-resume guarantee.

Actual 20 masked/unmasked CPU/MPS graphs verify named output shapes, all parameter/input gradients and discovery descriptors. Ten separate host Python subprocesses load saved real composed state with no PYTHONPATH repository assistance and check train/dropout/RNG output and gradients, then deterministic eval. All five modules train and exact-resume on CPU; all five train/fine-tune on MPS and refuse exact resume. A named MHA weights classifier and collision-safe nn.Module registration are checked. Real preview cache reload preserves typed utility mask outputs and MHA weights; semantic empty/default controls canonicalize identically.

## API/source pins

`normalize_attention(operation,config)`, `configured_input_ports(operation,config)`, ordered `parameter_shapes(operation,config)`, `constructor_options(operation,config)` and `resolve_attention(operation,config,layouts)` are Torch-free. `create_attention_module`, `execute_attention`, `validate_attention_state` and `ScaledDotProductAttention` own checked native execution. Generated tables bind these genuine helpers and all six literal descriptors exactly.

| Source | SHA-256 |
| --- | --- |
| `runner/attention_configuration.py` | `d6767e847d023abb2bfe1b48a5b1d0f608fdd47610c0fa4875d8804800dd60f1` |
| `runner/attention_execution.py` | `117daf2bcb3fdf49e984c6db2e5eeabffac2e8d2e5359fca520d3904abb01506` |
| `runner/engine.py` | `8f07be7d3820480f51137422b3dafcef5ec3d2a13caa12d22edcba3c0558a778` |
| `runner/graph_semantics.py` | `e3727c630e623b0587c487034e5aeb361b330170b127f5388e6bdcfeda469df1` |
| `runner/checkpoint_lifecycle.py` | `906df2a138191beaddaed57b43a6500f491c52d10c889928c284e3e53d9572f8` |
| `runner/tests/test_attention_runtime.py` | `57f84d398d86edb82c4a484a4e43b9230f310c3207b3e8244b91c9ea9485e190` |
| `runner/attention_adapters_generated.py` | `8984d1be031cc17c5de4ddc6c6afad439e25de1a243c827bab735a162f4a6c6a` |
| `components/attention/bundle.json` | `74522227d58a988d18867c9c9fbf600cb2d6304ddc06d4f52c6b698c95e1fc8e` |

| Log | SHA-256 |
| --- | --- |
| `.local/attention-helper-initial-red.log` | `c19f03f29bf09dcef94cf690238c15f4f147b59dfe5606d6976af183a4573939` |
| `.local/attention-helper-initial-green.log` | `2517e9eae50646d7bbe12680a6b4bab62b845ea0743476059c8b765356d3b1f7` |
| `.local/attention-execution-initial-red.log` | `f9fb4eeb664302d5e150773d0a4e5cdf27cf8cddb6275e44947eb3d8ad4d3f2a` |
| `.local/attention-execution-initial-green.log` | `3de8600d5f9c36198f06d42dfbe513d36cdc7c0c425bfb62e872ef282b3d9e12` |
| `.local/attention-mask-constructor-semantic-red.log` | `af6acecd06c4a8126d7db049817a80470ca795445c8d825d2a622e3172dc389e` |
| `.local/attention-mask-constructor-green.log` | `fbe8d037eb91bb1f03ed4aae3426241f76785f102845af76b4a95d07b3882d73` |
| `.local/attention-public-dropout-corrected-red.log` | `4eae549f47fbc0a761cb4f66cd57f2087cd242e596c9d2ec5e39be4540d310c9` |
| `.local/attention-public-dropout-green.log` | `58a3d5948840e7972c917f6d3f2bb1b2a606b1a592e44b57a674b44798847c53` |
| `.local/attention-graph-integration-red.log` | `ef5d53b91a95e00e81673e38fd4c6bcfa85f6bbda25dd8576efee6f9f2d9fb96` |
| `.local/attention-lifecycle-order-red.log` | `b5fee46e851dc3912e37ecbd99e1c7249417f52ef1ee71381e93ad8ead989545` |
| `.local/attention-provenance-semantic-red.log` | `25aceb5c96c7dcd046de79a4baee3e06bc5b1be3c5b118facbe0ee3737e25bcd` |
| `.local/attention-utility-head-bound-red.log` | `e5be5aff75bb4f154e8573448045b5047465348d305ca966e50544c5ccac170a` |
| `.local/attention-runtime-adjacent-green-attempt.log` | `03e448f60c007f4e5a817ba7e336ff7c377c29951c65c1d753f505d42c638790` |

## Explicit remaining gates

Native Core/App admission, learning store delivery, genuine canonical examples and immutable artifact/History/export verification are owned by other agents/root. This receipt does not assert those have passed. Native UI observations, original artwork, five lessons and complete catalog per-mode evidence remain required by the frozen closure; final source binding/accounting must follow their actual terminal results. No Swift, packaged/signed Python, UI, catalog mutation or commit was performed for this runtime receipt.
