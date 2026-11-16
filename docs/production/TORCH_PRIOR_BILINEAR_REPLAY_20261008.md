# Prior-unit reconciliation after Bilinear authoring · 2026-10-08

Read-only acceptance audit of the ten previously bound units, plus one Python authoring replay. This document supplements original frozen acceptance receipts; none of those receipts, the catalog or generated coverage reports were edited. Bilinear native closure and new-unit binding remain root-owned and pending at this observation. Audit base HEAD `6549867386bb480c6e3bf087fb204feeddd9c879`; working source changes identify the current Bilinear slice, not a claim that HEAD contains those edits.

## Exact affected bound dependencies

Every current verified catalog dependency was compared with its stored SHA256. Seven prior units have the following stale dependencies; ReLU, Tanh and Sigmoid have none at this observation. The full current hashes below are candidates for updates only after the stated verification is accepted. Update each repeated per-mode binding for its existing dimensions; retain mode lists and frozen receipts.

| Prior unit | Changed dependency | Bound dimension | Current SHA256 |
|---|---|---|---|
| torch.nn.Linear | `components/layers/bundle.json` | configuration | `8b3829d5d4108295c12984b797401ba89363010d3e144e2894597ceeb8bd5260` |
| torch.nn.Linear | `scripts/component_bundles.py` | validation | `4dc1b285d96b62a1eebe5e99c3c0b8a6fa4491b68c3dca0122f489d01e1f0886` |
| torch.nn.LayerNorm | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `aaa393cf8baca48f4de104e9bdc61eab04407746e786195ca13d3b79335f7c59` |
| torch.nn.GELU | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | validation | `aaa393cf8baca48f4de104e9bdc61eab04407746e786195ca13d3b79335f7c59` |
| torch.nn.Softmax | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `aaa393cf8baca48f4de104e9bdc61eab04407746e786195ca13d3b79335f7c59` |
| torch.reshape | `scripts/component_bundles.py` | configuration | `4dc1b285d96b62a1eebe5e99c3c0b8a6fa4491b68c3dca0122f489d01e1f0886` |
| torch.reshape | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `aaa393cf8baca48f4de104e9bdc61eab04407746e786195ca13d3b79335f7c59` |
| torch.flatten | `scripts/component_bundles.py` | configuration | `4dc1b285d96b62a1eebe5e99c3c0b8a6fa4491b68c3dca0122f489d01e1f0886` |
| torch.flatten | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `aaa393cf8baca48f4de104e9bdc61eab04407746e786195ca13d3b79335f7c59` |
| torch.transpose | `scripts/component_bundles.py` | configuration | `4dc1b285d96b62a1eebe5e99c3c0b8a6fa4491b68c3dca0122f489d01e1f0886` |
| torch.transpose | `Sources/KwipMLApp/WorkflowWorkspaceSession.swift` | failure_handling | `aaa393cf8baca48f4de104e9bdc61eab04407746e786195ca13d3b79335f7c59` |

No other stale asserted dependency was found. Whole-file hashing intentionally makes a shared manifest/generator change visible even when a prior component's own record is unchanged.

## Scoped preservation evidence

- The prior Linear manifest record and the current Linear record compare exactly equal as parsed JSON. Bilinear is appended; there is no alteration to Linear widths/defaults/runtime/version/schema/learning/example contract.
- Generator changes select exact reviewed definition/runtime contracts by layer ID and add Bilinear factory/declaration mapping. Tensor-layout validation/runtime metadata remains unchanged. New mutation controls reject Bilinear schema/broadcast/dtype/parameter/version/port drift without weakening Linear's controls.
- Actual replay `.local/prior-bilinear-authoring-replay.log`: **151 passed, no skips, 0.12 seconds**. This executed the common authoring suite and activation/normalization/layers/tensor-layout/Bilinear tests. No Swift build or native operation was performed by this audit.
- Root-run `.local/bilinear-scoped-swift-green.log`: **33 Core + 34 App tests passed, zero failures/skips**. Log inspection confirms Linear/TensorLayout metadata, Linear/ReLU/Tanh/Sigmoid catalog registration and tensor-layout admission checks, plus the opt-in actual Bilinear workflow launch. This is root-run automated evidence, not this audit's native observation.
- The only shared admission source change appends `bilinear` to the guard. Selected compiled graph validation and the existing TensorLayout operation condition are retained. No existing operation's runtime decoder or executed numerical path changed.

Root supplemental shared-admission replay: `.local/bilinear-prior-admission-replay.log` records **12 App tests passed, no failures/skips**, covering GELU, LayerNorm and Softmax selected/queued optional descriptors, valid dispatch, cancellation and disconnected controls. No numerical implementation changed.

## Bounded replay and update plan

Linear's manifest and generator hashes can be reconciled using exact record preservation and the authoring/Core metadata tests; no new Linear numerical training is needed for that additive change. Reshape/Transpose/Flatten generator bindings use their unchanged exact authoring controls and current generator drift proof. Their admission bindings, plus LayerNorm/GELU/Softmax bindings, use the unchanged admission logic with root's scoped admission regression; perform only any native catalog/admission observations root judges necessary for the shared entry point. Existing numerical/lifecycle/lesson receipts remain frozen.

Root should append this reconciliation receipt to affected dimensions with each unit's own existing mode scopes, then refresh only the changed source hashes listed above. Do not replace an original frozen receipt or automatically refresh unrelated dependencies. Current catalog evidence hash failure before reconciliation is expected and must not be described as complete coverage. No release gate closes through this document.

## New Bilinear dependency map

Bilinear's four unchanged modes are inference, training, input-gradients and state-roundtrip. The new immutable `TORCH_BILINEAR_ACCEPTANCE_20261008.md` must record distinct current/historical observations and be bound for all four modes in all ten evidence dimensions. Actual IDs: primary `assessment.bilinear-with-bias`, additional `assessment.bilinear-without-bias`; example `prepared-bilinear-classifier`.

| Dimension | Actual dependencies to bind alongside the frozen receipt |
|---|---|
| configuration | `components/layers/bundle.json`, `BilinearComponent.swift`, `BilinearConfiguration.swift`, strict worker `runner/bilinear_configuration.py` |
| execution | `runner/engine.py`, `runner/bilinear_configuration.py`, `runner/tests/test_bilinear_axes_independent.py`, existing Bilinear graph tests |
| export | engine export implementation, independent axes export/gradient tests, lifecycle strict saved-state tests |
| validation | `BilinearCapabilityBridge.swift`, exact runtime decoder, configuration/Core metadata tests, authored generator |
| diagnostics | actual assessment criteria and independent gradient/shape refusal tests; no new calibrated diagnostic rule is implied |
| lesson | `BilinearAssessment.swift`, reference, deliberately distinct supplied-parameter canvas, worker policy/tests, real store/privacy/reopening tests |
| example | `BilinearExample.swift`, generated Layers launcher, prepared example dispatch, actual workflow launch test |
| ui | bundle presentation, Learn catalog/registration tests, `bilinear-v2.png` and recorded generated original provenance, root native receipt |
| persistence | draft/receipt/store tests plus actual Bilinear lifecycle resume/fine-tune/cache/strict checkpoint tests |
| failure_handling | selected/queued admission tests, runtime/config/shape refusal tests and actual cancellation/privacy controls |

New binding RED `.local/bilinear-binding-red.log` is root-observed: five tests/seven failures against the incomplete catalog. The staged regression requires actual worker/numerical/manifest/assessment/bridge dependencies; an unchanged narrative receipt cannot hide a stale source. Owner consumer denominator remains 88, minimum 80; this reconciliation neither grants Bilinear credit nor claims broader framework API coverage.
