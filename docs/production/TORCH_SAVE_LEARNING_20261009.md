# Checkpoint learning: scoped implementation evidence

2026-10-09 · TORCH-LIFE.1 · provisional: actual App lifecycle worker and final four-method Core replay/native/coverage remain root-owned gates. This records implemented learning behavior, not capability acceptance.

## Bound implementation

Two assessment IDs: `assessment.checkpoint-restore-weights` and `assessment.checkpoint-continue-training`. Strict `checkpointLearning` schema1 accepts only taskID and typed trainingMode/parent/epochs/batchSize. Default drafts start fresh with no parent; incomplete control choices persist for repair. Old lessons omit the optional draft field. No forward node, workflow schema, arbitrary path, external checkpoint or new training engine was introduced.

Four fixtures per assessment execute real preparation, `engine.run` and the existing pinned checkpoint loader. Restore performs eight real training runs (four parents plus four fine-tuned children), restores an independently instantiated model, compares different held-out predictions with scalar affine math and compares the full new epoch/settings/momentum against independent MSE/SGD loops. Continuation performs twelve real training runs (four parent/uninterrupted/resumed groups), including nonzero SGD momentum, StepLR, MSE and dropout; compares full parameter/buffer/optimizer/scheduler/objective state, complete history, RNG and actual executed epoch boundaries. Every parent file's initial bytes are compared after the child. No learner success flags are accepted.

Measured keys and shapes are exact task-specific contracts; generic old receipts retain their old exact-key validation. Public projections include saved/restored/updated tables only after current receipt verification. Missing criteria or formative tables refuse replay; private attempts and private current mode withhold projections.

## Actual RED and GREEN

- `.local/checkpoint-learning-python-red.log`: initial test setup syntax failure, not semantic evidence.
- `.local/checkpoint-learning-python-semantic-red.log`: 7 failed/7 passed against the rejecting implementation scaffold.
- `.local/checkpoint-learning-python-green-candidate.log`: 13 passed/1 failed. The wrong fresh continuation numerically matched uninterrupted final state, proving final-state equality alone was insufficient.
- `.local/checkpoint-learning-boundary-red.log`: 2 failed/12 passed for missing actual executed-epoch boundary evidence. Real `started.initialEpoch` and epoch events now distinguish resume `[2,3,4]` from fresh `[0,1,4]`; no submitted mode flag is graded.
- `.local/checkpoint-learning-dispatch-red.log`: 2 failed/19 passed for missing worker registration.
- `.local/checkpoint-learning-partition-red.log`: 4 failed/21 deselected. Original combined/random preparation did not preserve the supplied eight training/four held-out groups. The repaired path asks the existing split operator for its position order, arranges the supplied rows, then performs genuine dataflow preparation and strict typed loading. Independent tests check both complete row/target groups and disjoint lineage.
- `.local/checkpoint-learning-python-perturbation-green.log`: final 30 passed, no failures/skips, 4.27 seconds. Includes two real four-fixture isolated worker dispatches, direct actual lifecycle runs, strict controls/fixtures, incompatible resume refusal, pre-preparation cancellation, independent hand-calculated affine update, exact partition roles and actual child artifact perturbations of model/optimizer/RNG/scheduler/history that fail their comparisons.
- `.local/checkpoint-learning-prior-dispatch-green.log`: 123 passed, no failures/skips, 30.30 seconds across existing generic learning evidence, optimizer and scheduler tests.
- Root `.local/checkpoint-learning-app-semantic-red.log`: raw-ID App start failed for both tasks before shared integration; initial compiler setup failure remained separate.
- Root `.local/checkpoint-learning-app-controls-green.log`: 3 Core + 9 App methods passed, no failures/skips, before the last exact-parameter-tolerance/private/default-draft test extension. This is not yet final four-method Core proof or actual App multi-mode worker proof.

Final-source standalone benchmark `.local/checkpoint-learning-budget-final.json`: four-fixture restore 0.503 seconds/eight actual engine runs; four-fixture continuation 0.307 seconds/twelve actual engine runs. Each records maximumTensorElements5056, below unchanged assessment bounds60 seconds/1,000,000 elements. These are local measurements, not clean-machine or general performance guarantees. Cold receipt validation itself runs no training; a new Check operation performs a fresh measured assay.

## Limits and pending root gates

CPU lifecycle learning is explicit; this lesson evidence does not claim MPS exact continuation. Production CPU/MPS and loader security matrices belong separate runtime receipts. Genuine prepared examples, publication metadata, final App guided/practice/private/cold workers, native action review and final accepted catalog binding remain root/authoring work. No accepted catalog, original receipt, engine or checkpoint loader was edited in this learning slice.

## Frozen learning source snapshot

- `Sources/KwipMLCore/CheckpointLearningTask.swift` · `d33f224a33915312dd087230b586fc0848cb5eee53c27170b08f0b29643697ae`
- `Sources/KwipMLCore/CheckpointLearningConfiguration.swift` · `3f055cbff5637206d54b211f6d78c2e091458445abfac5f993793435986feb15`
- `Sources/KwipMLCore/CheckpointLearningSubmission.swift` · `9fb6554baf813bc06d47bd49c74d07026622c1c5c80b69e7c066125dce53d6f2`
- `Sources/KwipMLCore/CheckpointLessonReference.swift` · `97bf3d0bc8e7e2581cb80aa0bc9a4077e0e09df8201e1c5d54b11cb9f29ea49d`
- `Sources/KwipMLCore/CheckpointFormativeProjection.swift` · `ce80c27cc7b3062fe454660d639145bb5e78a9679a868a600d6fc99a61e14cd0`
- `Sources/KwipMLCore/TensorLearningTask.swift` · `184e51e849dee7570caea23691dd282d3348e9e78196930aa54c1a8909c420f9`
- `Sources/KwipMLCore/TensorLearningDraft.swift` · `9016579b378993ac9bc1a0eca41c11edb108c51c1a85630929732ff0914d3fd6`
- `Sources/KwipMLCore/TensorLearningReceipt.swift` · `e5fe1e97f79b72b42e128a3849ba0074eebdabb2dd82d1666b7b235922f4c4ea`
- `Tests/KwipMLCoreTests/CheckpointLearningTests.swift` · `718c97226acb609630aaf738e4e84c5b2d7f2a81dd5c6fe0115ae44e60352721`
- `runner/learning_checkpoints.py` · `ba69acc06fb953d3a9d4a2f84d4e50f4a3009f14a68faf40acf8b1e1bbef98a4`
- `runner/learning_evidence.py` · `1134a86ea952209bd04b1791dab96c9d3c40de17fb2ca4057b83951fbd7e59fe`
- `runner/tests/test_learning_checkpoints.py` · `a19c12769fde92a5386e070b3bd2512a1fa910635cf8625e5debed223395943b`

## Actual private-fixture collision and bounded repair

Root's `.local/checkpoint-learning-integrated-green.log` is a retained terminal RED despite its filename: 7 Core methods passed, 16 App methods ran with seven assertion failures caused by one actual random/private fixture's `Selected features overlap across prepared partitions at float32 precision` refusal. The earlier small quantized draw space could create duplicate two-value rows. Production leakage admission was correct and remains unchanged.

The generator now adds row/1024 to the first feature for all twelve supplied rows before calculating targets. Base draws are integer multiples of1/16, while distinct offsets range0…11/1024, so different rows cannot have equal first features; all resulting values are exactly float32-representable. This is a deterministic construction, not a retry-until-passing private seed. New `CheckpointLearningUniqueRowsTests` checks512 seeds × two tasks × four fixtures, twelve distinct first features at production precision and target formulas computed after adjustment. Actual serialized Core execution of that new regression is root-owned/pending at this append.

`.local/checkpoint-learning-duplicate-row-refusal-green.log`: one actual duplicate training/held-out feature-row refusal passed,30 deselected. `.local/checkpoint-learning-python-final31-green.log`:31 passed, zero failures/skips. The existing engine/loader and all leakage limits remain unchanged. This refinement changes fixture digests, so old provisional receipts cannot silently replay as the new assessment.

Updated paths after the previous source snapshot:
- `Sources/KwipMLCore/CheckpointLearningTask.swift` · `412a6fbc9ca7c6f97088b2c1abb7106f9736351823a0ac99c93647037520d06a`
- `Tests/KwipMLCoreTests/CheckpointLearningTests.swift` · `718c97226acb609630aaf738e4e84c5b2d7f2a81dd5c6fe0115ae44e60352721`
- `Tests/KwipMLCoreTests/CheckpointLearningUniqueRowsTests.swift` · `6dad7a4d84da722f55dbae0090291fb649e5d509ebcb25735cc4d7c2044b977b`
- `runner/tests/test_learning_checkpoints.py` · `21f5f07baf751293fa54282fcdbe451553e79a12bff139e2868cd7edc7573765`

## Genuine packaged registry propagation repair

Root native packaged restore failed with a nonexistent `KwipMLRuntime/Sources/KwipMLCore/Resources/components.json` lookup. Development worker success did not establish packaged execution. The launcher already verifies bundled `components.json` and forwards its exact registry into learning evidence; the checkpoint branch dropped that argument before real training/restored-model construction.

`.local/checkpoint-learning-packaged-registry-red.log`: two actual isolated packaged-layout cases failed,31 deselected, reproducing the same missing-checkout path for both tasks. The bounded fix forwards the existing verified registry through the checkpoint dispatch and optional worker argument into every engine run and the restored `GraphModel`. No resolver fallback, source-tree shadowcopy, integrity relaxation, dataflow CLI, engine or checkpoint-loader change was made.

`.local/checkpoint-learning-packaged-registry-green.log`:33 passed, no failures/skips,8.85 seconds. Both tasks execute four actual lifecycle fixtures using a copied worker/runtime layout launched with `-I -B`, no checkout `Sources` directory and no Python/KWIP environment overrides. The test also rejects an unsealed registry tamper and, after genuine inventory resealing, refuses a registry whose actual linear control bound excludes the lesson model. `.local/checkpoint-learning-prior-packaged-registry-green.log` separately passes the existing generic packaged launcher discovery/execution/registry restriction regression.

Current Python source snapshot after this bounded repair (supersedes earlier Python hashes only):
- `runner/learning_checkpoints.py` · `a695792cb0291cf6aa240ca2c4c713adce9ca1af0cc0529c77b687ac8f097ea0`
- `runner/learning_evidence.py` · `454f3622c59d53e5fc19b1ad07552c307a28567ba9a4c3fb03421ade88cc67f4`
- `runner/tests/test_learning_checkpoints.py` · `de3da991dff0a901ecd52ae97d213be43d65a0a6f13dcd26891f47b2b7815980`

Root rebuild and fresh native lifecycle execution remain required; the earlier native failure is retained and is not converted to a success by these automated checks.
