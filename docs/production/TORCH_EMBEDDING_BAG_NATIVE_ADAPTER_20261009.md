# EmbeddingBag checked native adapter — 2026-10-09

Scoped backend foundation, not bundle acceptance. Only NEW runner/embedding_bag.py, runner/tests/test_embedding_bag.py and this receipt are owned/changed. Frozen configuration and packet helpers are reused, not edited. No shared engine, discovery, export, registry, Swift, native UI, catalog or old acceptance changes.

## Actual interface and ownership

`CheckedEmbeddingBag(config, *, device='cpu')` subclasses real `torch.nn.EmbeddingBag`. It owns its conventional `weight` Parameter and state_dict key `weight`; a future composition ModelPrimitiveAdapter supplies the sole `.operation` prefix. No duplicate registration, fake module, detached floating input, implicit device transfer or adapter CPU fallback. `forward(packet)` accepts the actual EmbeddingBagPacket, validates its current tensor values and exact configured layout/terminal flag/weights presence, then calls native forward. Native max_norm's selected-row mutation is preserved on admitted CPU configurations.

Constructor normalizes the frozen exact configuration, rejects table product above general2000000 before native allocation, and accepts explicit available CPU/MPS float32 placement. Shared aggregate model admission must still count every layer before construction; this per-table limit does not authorize multiple tables beyond that aggregate. Selected initialization separately imposes200000 whole-model parameters. Freeze uses actual weight.requires_grad_(False), since installed PyTorch2.14.1 EmbeddingBag has no `_freeze` constructor keyword.

Forward rechecks actual native controls and table shape/dtype/dense/device/finite ownership, packet int64 IDs/offsets, finite float32 source weights and data budgets. Finite output and table values are checked after native execution. Invalid values are refused without truncation or substitute inputs. Config is stored as a fresh scalar-only dictionary; no packet/storage aliases are retained. Input item-weight derivatives remain live even with a frozen table. Sparse gradients are actual sparse COO; no sparse optimizer support is granted here.

## Discovered native limits and approved device admission

Actual host installed PyTorch2.14.1, MPS available. Initial read-only execution probe mistakenly passed Embedding's `_freeze` keyword to EmbeddingBag; all TypeErrors in `.local/embedding-bag-native-preflight.log` are setup-only, not backend failures. Corrected probe uses actual weight.requires_grad, retained in `.local/embedding-bag-native-preflight-corrected.log`; basic CPU/MPS forward/backward dense/sparse/weighted/frozen cases ran.

The first checked-adapter numerical run produced **52 passed,1 failed,1 warning**. Independent repeated-index sum fixture `[1,1,1]` with upstream `[2,-1]` requires row gradient `[2,-1]` under inverse-frequency scaling. CPU matched; MPS returned `[6,-3]`, the unscaled result. This genuine failure is preserved in `.local/embedding-bag-adapter-green-attempt.log`, with explicit measured values in `.local/embedding-bag-native-limitations.log`. Expectations were not weakened.

Native MPS max_norm emitted an unsupported `aten::embedding_renorm_` warning and CPU fallback even with PYTORCH_ENABLE_MPS_FALLBACK unset. Root therefore froze explicit refusal: **MPS max_norm requires selecting CPU**, before construction and again at forward after `.to()` changes. No fallback was implemented or credited.

For frequency scaling, root froze refusal on MPS whenever actual weight.requires_grad is true. Constructor checks intended freeze before allocation; forward checks actual flags, so moving a CPU layer or unfreezing an admitted MPS table cannot bypass it. **Frozen MPS frequency configuration is admitted**: actual CPU/MPS independent weighted outputs and item-weight gradients passed, with no table gradient. Remedy text names CPU or keeping the table frozen. Configuration remains device-independent; runtime admission is narrower based on actual measured behavior.

This is not blanket backend parity: dense sum/mean/max, weighted sum, sparse sum, frozen weighted input derivatives and tie routing were tested on both devices. Frequency-scaled trainable tables and maximum norm are CPU-only in this adapter. No scheduler/optimizer, full-engine continuation, packaged native or supported-machine claim follows.

## Actual TDD and independent arithmetic

Importable native-default passthrough subclass baseline produced **53 failed in1.41s** (`.local/embedding-bag-adapter-subclass-red.log`), including wrong reduction/options/gradients, missing freeze and missing validation. An earlier wrapper-baseline log is historical only; final ownership is the native subclass.

After actual native-limit findings, new refusal/changed-device/trainability tests produced **6 failed,53 passed,1 warning in1.10s** (`.local/embedding-bag-adapter-policy-red.log`) before device guards. Frozen-frequency positives were already independently measured rather than assumed.

Final actual command:

```
.venv/bin/python -m pytest -q runner/tests/test_embedding_bag.py runner/tests/test_embedding_bag_configuration.py runner/tests/test_embedding_bag_packet.py
```

Terminal success: **214 passed in1.02s**, zero skips/warnings (`.local/embedding-bag-adapter-final-green.log`). Scope:59 adapter cases,81 configuration cases,74 packet cases. CPU and MPS both executed; no MPS skip on this host.

Literal review arithmetic supplies sum/mean/max outputs and full table gradients for fixed and packed layouts, terminal/nonterminal offsets, true empty bags and padding-only bags. Weighted sum verifies both table and actual floating-weight gradients, including padding exclusions and frozen table. Repeated-index frequency CPU arithmetic, first-maximum tie routing under reversed IDs, max_norm2 with norm orders1/2, selected-row-only mutation and frozen CPU renormalization are checked. Sparse sum compares real sparse gradient converted for inspection to independent dense expected values; zero-item packed bags yield actual zero outputs. One real plain SGD rate.1 step matches independent learned table values, followed by strict native state reload and identical output.

Negatives include mismatched packet configuration/weights presence, out-of-range IDs, nonfinite weights/table/output, changed table dtype/native controls, unsupported device, and preallocation table/device-control refusal via constructor mock. No huge allocation probe. Floating weight gradients are never fabricated for discrete indices or offsets.

## Initial foundation source and evidence pins (historical after consumer refinement)

| Path | SHA256 |
|---|---|
| runner/embedding_bag.py | aa96fa26437b4d4b07434098670c1b24cf401f14c5fd0d893da5290f8067a9dc |
| runner/tests/test_embedding_bag.py | bdab9d038319fe127fb810c80d2c6a9a2337f9f8f92f2460e2d0a2a50c621267 |
| runner/embedding_bag_configuration.py | d101c2b882c7435cb8a761fec16b626de532dde2fbded7ce97d439097e89f1d6 |
| runner/embedding_bag_packet.py | 8b8bf4e5c4fe17852dd7a0ee2648e4e45d4ea2218144e66010a59f572fa0a669 |
| .local/embedding-bag-adapter-subclass-red.log | aa922dba4e6e1246fb8bf89eb28de4c39196f4cbd0310980b7bab4a68d870992 |
| .local/embedding-bag-adapter-green-attempt.log | cda86268d829378a1032699acd8370dabc8240db872fff5f8d3b78143bf28acd |
| .local/embedding-bag-adapter-policy-red.log | 1a61fcf36586a438efe7887a7a3faa161e63d9ffe69fe02b732390a04230fd32 |
| .local/embedding-bag-adapter-final-green.log | 01bb4cb736fe60f19a5ad46f6b076732add3baf13c69c0628901a4a73ba4ca5b |
| .local/embedding-bag-native-limitations.log | 2dec11fc211ce22095668db86207ee791481e618828694205ae8541a7b644bb3 |

Remaining integration: synchronized prepared packets/batching, native engine allocation/ownership/discovery, sparse optimizer policy, initialization/trainability/requested derivatives, portable export and checkpoint continuation, Core/App authored configuration/admission, lessons/examples/original art/native UI and prior-bound regression. Original four modes remain required; this adapter receipt neither narrows them nor awards coverage.

## Consumer-alignment refinement — final seal

Root identified two concrete missing consumer paths before seal: copied loose-worker import and legitimate separate trainability overrides. Meaningful additional tests first produced **3 failed,59 passed in1.46s**, `.local/embedding-bag-adapter-consumer-red.log`: both initial-freeze override directions refused and actual isolated copied-worker import failed.

The adapter now uses the existing __package__ import pattern for package and copied loose files. The test copies only embedding_bag.py, embedding_bag_configuration.py and embedding_bag_packet.py outside the repository, runs host Python with -I, imports the loose adapter and executes real literal sum inference; no installed runner or engine import is used.

Configuration freeze is **initial native state**, not an immutable constraint on a separately selected trainability policy. The unconditional equality check between actual requires_grad and initial freeze was removed. Native controls, table shape/dtype/finite ownership, packet and device checks remain strict. Real dense CPU tests prove initially trainable→frozen and initially frozen→trainable overrides, retaining literal outputs, source-weight gradients and applicable table gradients. MPS frequency refusal still uses actual weight.requires_grad and still refuses a later unfreeze, as independently tested; this is not a native-control or weight-tamper concession.

Final actual same three-file pytest union: **217 passed in3.85s**, zero skips/warnings (62 adapter +81 config +74 packet), `.local/embedding-bag-adapter-consumer-green.log`. The previous214 result and prior source hashes remain historical evidence; current source pins supersede them for consumer integration.

| Current path/log | SHA256 |
|---|---|
| runner/embedding_bag.py | 0fc72cf36de523db009860c2a850492b96c60a77917022ca7102005a6d3718ae |
| runner/tests/test_embedding_bag.py | bb7cfda78cf1190922c5ee410109e999017061fd19afe9c1552816edcb1166da |
| .local/embedding-bag-adapter-consumer-red.log | 0b5ed0cf31afbfc7bd2979189ea24ef1e2e582aa8d9c77848e6bcd5fa2f1eca3 |
| .local/embedding-bag-adapter-consumer-green.log | 3ad805ad12dd650bf93db4d9aa4480119a239150da17ec4cccbc46ae69c3c362 |

No shared integration, device-policy expansion or acceptance credit. All prior explicit MPS limitations and pending bundle exits remain.
