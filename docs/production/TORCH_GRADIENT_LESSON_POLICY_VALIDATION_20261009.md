# Gradient lesson control validation — 2026-10-09

## Native finding and bounded correction

After the busy-context repair, root used the Keep settings lesson and changed Gradient policy's Small groups per adjustment from 1 to 2. Apply refused with a misleading source-changed message and no Undo transaction. This was a genuine explanation/validation defect: `GradientLearningConfiguration` intentionally keeps the gradient policy fixed in Keep settings and Next run, and keeps trainability fixed in the two gradient lessons. The lesson inspector previously validated only the general production recipe before Apply; the stricter lesson guard was reached only during draft commit.

The lesson adapter now validates each draft against the actual task and current companion recipe before Apply. Fixed-gradient changes explain: “This lesson keeps Gradient policy fixed. Restore it and configure Trainability instead.” Fixed-trainability changes provide the converse next action. Missing task/companion context refuses with a reopen instruction. This reuses the existing partial lesson configuration validator, without requiring complete starter connections.

`TrainabilityInspectorContext` gains an optional validation callback, defaulting to a no-op for existing generic workflow callers. Its existing validationMessage/canApply/apply path invokes that callback. Production workflow freedom, expected source checks, busy restrictions, pending draft retention and Undo remain unchanged. No Core lesson restriction was relaxed.

## Evidence and limits

Root's serialized `.local/gradient-lesson-policy-validation-red.log` terminated with exit 1: four methods, ten expected assertion failures across the two fixed-policy methods. The two editable-control methods passed. This establishes missing pre-Apply lesson validation rather than a generic commit failure. The tests cover both fixed-gradient tasks, both fixed-trainability tasks, retained invalid drafts/discard, no misleading source-change message, and the editable controls applying on partial starter graphs.

Root's serialized `.local/gradient-lesson-policy-validation-green.log` terminated with exit 0: 14 App methods, zero failures/skips, covering the new four methods, availability repair, lesson inspectors and generic Trainability context. `.local/gradient-results-policy-validation-packaged-build.log` terminated with exit 0, including registry/runtime verification and strict local candidate signing. Native cold replay confirms the fixed-setting explanation, preserved draft and disabled Apply; Discard restores the original. In the editable Combine lesson, Apply creates Undo, Undo restores graph grouping2, and explicitly discarding the retained unapplied draft reveals the restored value. See the native receipt. The implementing worker executed no Swift/build/native command. The earlier native source remained unchanged after rejected Apply; no saved edit or new acceptance credit is claimed. The separate busy-context fix remains intact.

## Current source pins

| File | SHA-256 |
| --- | --- |
| `Sources/KwipMLApp/TensorLearningCanvasAdapter.swift` | `ac5a17f6755292657ebb058dca0d537f4114b686cab7443353f540f4c7a2ba24` |
| `Sources/KwipMLApp/TrainabilityInspectorContext.swift` | `8f56267fcd3a78896631fd26bd246a803bbb211d76d0c0a5c5443836b868da0e` |
| `Tests/KwipMLAppTests/GradientLessonInspectorPolicyValidationTests.swift` | `64ae590c594ff15b04313f705cf709dcb3930831683f69d8c8cb012920ca88b5` |
| semantic RED log | `25e9daf0d0b58cd7ba61dcf418a916d0b1b4d7d0df50337c6d27d808f0f7b46e` |
