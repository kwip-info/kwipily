# Attention canonical workflow host proof — 2026-10-08

This is independent host evidence for the five genuine Core prepared-classifier factories. It is not a native UI export, capability acceptance, App Store release or broad supported-machine claim. No production source, catalog, frozen receipt or artwork was changed.

## Actual factories and prepared data

Root's serialized `AttentionCanonicalExampleTests` emitted actual original workflow, compiled model, dataflow and settings under `KWIPML_ATTENTION_CANONICAL_FIXTURES` into `.local/attention-canonical-factories-20261008`. This agent waited for the real files; it did not construct stand-in fixtures or run Swift. Original workflow is schema1, compiled graph schema2, compiler6. Each model has one primary input, independent learned source/query/key/value branches as appropriate, explicit typed mask utility and a learned classifier head. Supplied masks and causal target masks follow each actual factory. Unconnected MHA returned weights remain present and checked.

Sealed preparation archive:

`/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/056FF2E4-2C6A-4E0E-81B5-4EC1A1B6A7B0/F3F3695C-71E6-453A-ADB4-58A09EB6F928`

The loader verifies its pinned manifest SHA/dataflow digest/qualified training and validation outputs/feature and target columns, float32 input contract and two-feature shape. Every actual emitted factory dataflow must equal this sealed preparation digest. Actual partitions contain **48 training and 16 held-out observations**. All captured settings are epochs3/batch32/learningRate0.03/seed42; the prepared partition is consumed directly without resplitting. Factories use dropout0 and implicit Adam with no scheduler; runtime active-dropout and optional recipe matrices remain separate evidence.

## Independent reference and checks

The plain Python oracle copies `scalar_attention` and `scalar_module` from runtime-test source SHA `57f84d398d86edb82c4a484a4e43b9230f310c3207b3e8244b91c9ea9485e190`, then extends the copied functions with independent supplied/causal/padding mask application for Encoder/Decoder/full Transformer. Copied-and-extended functions SHA is `b6c71e4491aeb1b3803fbaea5d8bccfedd64c49ba60a1a76d8ae829b2a0de9a1`. Plain scalar graph composition computes learned affine branches, reshapes, binary masks and padding, native learned projections, head softmax comparisons, residual paths, LayerNorm statistics, activations and the learned final classifier. No attention native operation, runtime adapter or portable helper computes expected values. The oracle file is frozen and pinned rather than importing mutable test-source functions.

Each factory trained for its real captured settings on **CPU and MPS: ten completed training-device cases**. Saved weights are evaluated against scalar logits for all 16 held-out rows; classifier choices must match. Runtime full stages are independently compared, including every projection, mask/padding, attention out, unconnected MHA weights, flatten and head/output. Integer mask roles match exactly.

Every completed package verifies model snapshot/hash, checkpoint SHA/bytes, metrics SHA, contract equality and copied preparation manifest. Portable model source has no product runtime import. Every saved export executes on CPU and MPS for full and one-row batches against scalar logits/classes. Saved application's genuine inference CLI executes batches1 and16 and matches scalar logits/classes. A tampered checkpoint must be refused after successful prior loads; original bytes are restored in these host-created proof packages.

An additional read-only exported-helper supplement executes the actual saved portable `execute_attention` helpers on CPU and MPS for all ten trained packages: **20 helper cases**. It evaluates utility mask/padding and every named attention output against the same independent scalar graph traces, including unconnected MHA weights. Portable masks feed the portable attention helper; it does not substitute runtime mask outputs. All required integer roles match exactly. These measurements supplement, rather than replace, standalone portable classifier/CLI results.

Every CPU full run matches one-epoch checkpoint plus exact continuation in model weights, optimizer state, RNG and complete metric history. Every MPS exact continuation refuses; each explicit fine-tune records weights-only lineage, initial epoch0 and optimizer/RNG/epoch/metric resets. These default-dropout canonical checks supplement the runtime's independent active-dropout/gradient/state tests; they do not imply MPS exact continuation.

## Actual terminal and chronology

- First local attempt trained the first genuine MHA CPU graph, then failed because the independent utility reference incorrectly assumed an omitted default `padding_keys` had been explicitly serialized. Actual Core mask configuration legitimately omits that empty default. Only the ignored verifier was corrected to the frozen default; no production source or expected numerical tolerance was changed. The failed output directory/log remains preserved.
- Corrected fresh-directory replay: **terminal exit0, all ten canonical CPU/MPS cases passed**. Final source-freeze guard passed; every pinned runtime/helper/factory/verifier input remained unchanged.
- Exported named-helper supplement: **terminal exit0, all20 cases passed**, matching exact source and canonical-receipt pins.

Maximum absolute scalar error across classifier full/micro/CLI, runtime stages and exported named helpers: **7.7809123633088006e-07**. Actual PyTorch `2.14.1`. Existing upstream odd-head/norm-first Transformer nested optimization warnings occur during native constructors; actual execution disables nested optimization per the frozen contract. No case was skipped.

| Evidence | SHA-256 |
| --- | --- |
| `.local/attention-canonical-host-20261008-r2/receipt.json` | `b2a6d7138e43499425a056b9d59b32b078cfa51df5ddf95958bdf863a6fb6835` |
| `.local/attention-canonical-host-20261008-r2/exported-named-receipt.json` | `56d52b56e4cb1f150b5a5a938d2141592b709ecb128915f5d1ede187f60b1f15` |
| `.local/attention-canonical-host-20261008-r2.log` | `683035e3e51a597df62cff2819095af2a5a46f3dc67f69458dbc474c4724a3c7` |
| `.local/attention-exported-named-stages.log` | `7c38c6e1e588ea4595760e8109260228366aab2b725140d0c9138875372d3f16` |
| `.local/verify-attention-canonical-workflows.py` | `49bbb9c8780bd38eef1e7073cf5f717b574231c602d451f28cd81195f8858fd0` |
| `.local/verify-attention-exported-named-stages.py` | `c715b55dc38f9fb38e3a7ee648437d2e19e86f21cbd4b2ee94be1a5a2ba5d667` |
| `.local/attention-canonical-host-20261008.log` | `79a221a1259ee157c5ce438a4c1e0e226a69f63cdb79b23f09538e4aea0b63bf` |

| Component | Training device | Actual run ID | Checkpoint SHA-256 |
| --- | --- | --- | --- |
| `multihead_attention` | `cpu` | `feefab9d-dbf6-4123-8874-0756b68453b2` | `922b91ac7b0f379a2b1e7ba667304c99cfa6e7287235ceaffca4b8bfcc37aed8` |
| `multihead_attention` | `mps` | `01ad2f6a-a56a-4492-9bfc-cd5d89578952` | `62d67a8842d888bf5cf7691d15eca194d0ec6e569658fa0f30412c6316a79c9c` |
| `transformer` | `cpu` | `1dc5ec7b-c049-4af9-b781-c3428c35ae70` | `7a40aa95cbdee3f8889af56fafb4430a10b071e0d1a6d01f40e16d399fbb0a49` |
| `transformer` | `mps` | `a5e2a95c-de0a-4f87-84dd-d7bcbb4fe4d9` | `1bcca364f11e37d5bbd91a69bab5f162c6bd0504b3fb85a9e7eb3a455f24f390` |
| `transformer_encoder` | `cpu` | `54df06c2-6b44-4c4f-b548-668bebadd306` | `2bb77ac997ec34306d476e3bfdcb07ac782f1fe7672718acf13f7ccb60189453` |
| `transformer_encoder` | `mps` | `865905fe-5b3c-4d7f-86c3-31d3a56e45ad` | `7573e9e0189cb5a1407fad096d7564b85105ee80acf37f0d34c8fdf4d348fc99` |
| `transformer_decoder` | `cpu` | `c4a9f334-1780-489d-bc77-8e9bedada7ee` | `0f1731dc5377800c09e5a341cee61ee0dae9fbccff2c3bfa3db53d0491350fb3` |
| `transformer_decoder` | `mps` | `94acacdb-5137-498a-9667-92d7a5e7f306` | `7c3a6b6d9e08d2953ba78a0596cc85783f1b01747734ade98bd11568da26a88d` |
| `scaled_dot_product_attention` | `cpu` | `12be3052-d6ec-4df7-9c7d-4f223ab57f20` | `dec67d3973dcb4eb78a2abb05801abe1e3217c06ac33ef09b5af8daa05153745` |
| `scaled_dot_product_attention` | `mps` | `2bc79118-df61-4ac8-ba01-6222ab8832b9` | `30402657db450fe4ab520fe0f2a96e52617d7df49c7c382cb5ddac47121e2965` |

The JSON receipts retain every fixture hash, current source hash, graph/model/checkpoint/manifest identity, native learned name inventory, per-device numerical differences and continuation manifest. This report does not silently refresh any accepted binding.

## Commands and remaining boundary

```
.venv/bin/python .local/verify-attention-canonical-workflows.py \
  --fixtures .local/attention-canonical-factories-20261008 \
  --archive '/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/056FF2E4-2C6A-4E0E-81B5-4EC1A1B6A7B0/F3F3695C-71E6-453A-ADB4-58A09EB6F928' \
  --output .local/attention-canonical-host-20261008-r2
.venv/bin/python .local/verify-attention-exported-named-stages.py
```

Host `.venv` only. Packages are host-created from genuine worker outputs, never represented as saved native UI exports. Native learned-window admission, live activity, formal learning delivery/private grading, cold History, native saved export and final accounting remain root-owned acceptance gates. No signed/bundled Python, UI, Swift, package build, catalog mutation or commit was performed by this proof.
