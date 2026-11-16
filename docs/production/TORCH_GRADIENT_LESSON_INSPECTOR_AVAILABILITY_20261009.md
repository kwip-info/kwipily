# Gradient lesson inspector availability — 2026-10-09

## Scope and cause

The root observed packaged candidate `8b487bb` completing the selected-influence lesson's public and private checks successfully, with Run enabled but the selected Gradient inspector still disabled. Both Gradient and Trainability lesson editors cached a context by selected node and source configuration, but additionally refused cache reuse while that context was busy. A view refresh during a check replaced the cached context. The still-visible editor's identity-based authority then remained stale after completion.

The bounded repair removes only the busy condition from these two cache-reuse checks in `TensorLearningCanvasAdapter.swift`. The existing live busy closure still blocks editing, applying and discarding during execution. Node selection, source configuration and draft-session identity checks remain unchanged. This introduces no new editor or notification framework.

## Test evidence

Root ran `GradientLessonInspectorAvailabilityTests` in its serialized Swift lane. The first log, `.local/gradient-lesson-inspector-availability-red.log`, contained four initial context-unwrapping failures before reaching the target behavior. The test helper had released its local workspace, which the lesson store intentionally references weakly. The fixture was corrected to retain the workspace; that setup-only failure is not semantic regression evidence.

The corrected baseline run `.local/gradient-lesson-inspector-availability-semantic-red.log` terminated with exit 1: four methods, eleven assertion failures, zero unexpected failures. Both busy→refresh→idle positive methods failed; both stale-configuration/selection/retired-session negative methods passed. Tests exercise genuine lesson adapters and live busy closures, without fabricating worker completion evidence.

The production repair and test are frozen pending root's serialized GREEN and subsequent native recheck. No Swift, build or UI command was executed by this worker. No capability acceptance or backend claim follows from this slice.

## Current pins

| File | SHA-256 |
| --- | --- |
| `Sources/KwipMLApp/TensorLearningCanvasAdapter.swift` | `5a248227e722bd105d37059777e41a9fe22511fc8075a7ef206c776d5df184ac` |
| `Tests/KwipMLAppTests/GradientLessonInspectorAvailabilityTests.swift` | `755a2b99d336ec557db9a9cb1694f05682ce3641ce920958c58915062e23fa01` |
| setup-only RED log | `9c7ee3e7c1b5111ebc924a53fe7062c535b8a52a83a3e29a1c4ffcf71efc5f5b` |
| semantic RED log | `4148d437131fff2ffa69bd5deb4c2ac5c23ffc21cb0c59240cf651033b5d60c1` |

Root final scoped GREEN: `.local/gradient-lesson-inspector-availability-green.log`, terminal0,24App methods, zero failures/skips. Covers both busy-lifecycle regressions, changed config/selection/retired-session controls, both structured contexts, pending lesson edits, public comparison presentation and all four actual lessons/all modes/cold receipts. Native repair replay remains pending.
