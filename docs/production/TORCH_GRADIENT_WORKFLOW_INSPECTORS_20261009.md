# Workflow gradient policy inspectors — 2026-10-09

The general workflow canvas now routes gradient-policy and trainability payloads to retained wrappers around the existing shared editors. The wrappers reuse existing Configuration/Information frames, flat controls, current/busy authority guards, expected-configuration Apply transactions and Undo. Their callbacks use the existing general Boolean pending-edit mechanism. This slice does not claim cold raw-draft persistence in the general workflow inspector; lesson persistence is a separate integration.

`GradientModelInspectorResolver` requires exactly one outgoing typed recipe connection to the matching input of one actual Train block. Missing, malformed or multiple connections refuse resolution; it never falls back to another Train/model. It resolves that Train's unique complete Model definition under the captured model/data registries, selects explicit-objective shape context only when that Train has an objective connection, and derives endpoint dtype/sample shape from configured per-port output contracts. Genuine input/index-input boundaries are distinguished from intermediate outputs. Native parameter choices reuse InitializationOwnerResolver with the exact selected Train and actual expanded model.

The gradient callback validates proposed typed configuration by compiling a scratch workflow at contract9 with the selected Train and current captured registries/objective. Trainability commits use the same scratch-compile guard. Neither validator writes the document; the existing authority performs the expected-source transaction. External source/selection changes retire authority callbacks, while own successful commits renew their lease. The wrappers observe authority for fresh busy/current presentation and release their own pending Boolean on disappearance. No capture permission is changed.

## Actual TDD

Root serialized `.local/gradient-consumer-editors-lifecycle-red.log`: **four new methods / eight assertion failures (one thrown rejecting scaffold)**. Tests exercise real captured endpoint/parameter shapes, Apply/Undo and pending state, trainability transactions/retirement, disconnected and multiple Train refusal, and actual objective-scale mismatch. They remain unchanged after implementation.

Implementation is frozen for root's GREEN run. No Swift/build/native command was executed by this authoring agent. Native routing/editor interaction, publication and acceptance coverage are not claimed here.

## Source and RED identities

- `Sources/KwipMLApp/WorkflowWorkbenchView.swift` — `860215103c1013fb10f56b9cdaa2d1ea5227aacc17187c3ea3170fe76413ba71`
- `Sources/KwipMLApp/GradientModelInspectorResolver.swift` — `f6df8781622b2f63cec222be49dd749bba60387a6580f9de8733511bb9aaf593`
- `Sources/KwipMLApp/GradientWorkflowInspectorViews.swift` — `511286780233a2e09d7396f34ba3a84e870b68c8cefbf1bcc2c5297aada4b32b`
- `Tests/KwipMLAppTests/GradientWorkflowInspectorTests.swift` — `d8d6799e782d84550d5404f4444987b7047fafc7beb84a0483f59599a783d86c`
- `.local/gradient-consumer-editors-lifecycle-red.log` — `08f61ed57904653cb0dd43a7193c7260bf410281ee4bdd6aa5a5436cf5feff67`

## Root automated verification

`.local/gradient-consumer-row-lifecycle-red-editors-green.log` compiled and ran the installed editor tests: workflow inspector four methods, lesson inspector two App methods and pending-draft two Core methods all passed with zero failures/skips. The same command separately recorded the lifecycle rejecting-baseline and ordered-row RED; those failures do not belong to the editor checks. Native presentation and overall paired-bundle acceptance remain pending.
