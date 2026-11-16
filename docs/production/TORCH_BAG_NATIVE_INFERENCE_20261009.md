# EmbeddingBag portable native-export inference — scoped host proof

The owned implementation is `Sources/KwipMLApp/Resources/inference.py`; the only new test source is `runner/tests/test_bag_native_inference.py`. No engine, packet, adapter, registry, Swift production, or acceptance catalog source changed in this slice.

## Contract and execution

Schema3 is an additive packet contract agreed with the native saved-export owner. Exact top-level keys are schemaVersion/input/output, plus objective only for an explicit objective. Input retains kind=bagPacket, packet schema1, actual sourceID, complete four-field source configuration, enabled role/dtype ports, BagInputMapping, minimum reachable numEmbeddings, and preprocessing=alreadyPrepared. It has no invented dense input shape. Binary output is the existing two-logit/class-index contract; objective output retains actual float32 observation-first shape and the existing recipe/source/targetMapping/scalarization metadata.

`load_model().predict(records, batch_size=...)` returns a CPU float32 tensor. Each record contains indices and exactly the configured paired per_sample_weights. Fixed groups have exact capacity; packed groups can be empty and have at most capacity items. Caller offsets are prohibited: each batch constructs genuine offsets from complete group lengths, respecting include_last_offset. Every record is checked before tensor allocation or model invocation. Limits remain 4096 groups, 1,000,000 items, 2,000,000 output elements and capacity128. IDs are exact in-vocabulary integers, weights finite float32 values. Class-index metadata refuses booleans.

The helper retains the bounded regular-file inventory, exact file hash/length checks, checkpoint weights-only CPU loading, strict model-state restoration and normalization-buffer checks. Training helpers are inspected as bytes but never executed for inference. Dense schema1 and explicit-objective schema2 remain available through their existing tensor API. This helper is CPU inference; it makes no MPS convenience-helper claim.

## Actual TDD

- Initial importable existing helper RED: `.local/bag-native-inference-red.log`: **9 failed, 8 passed, zero skipped, 20.76s**. Eight trained fixed/packed × weighted/unweighted × binary/objective positives and the nonexecution positive reached the dense-only contract rejection. Eight malformed contracts already refused.
- Initial additive implementation: `.local/bag-native-inference-green.log`: **17 passed, zero skipped, 26.31s**.
- Exact-integer metadata control: `.local/bag-native-inference-integer-red.log`: **1 failed, 1.73s**. Python equality admitted `[False,True]` as `[0,1]`; the exported-code sentinel demonstrated the missing early refusal. The bounded repair requires actual integer class identities.
- Final command:

```sh
.venv/bin/python -m pytest -q runner/tests/test_bag_native_inference.py runner/tests/test_portable_inference.py runner/tests/test_objective_portable_inference.py
```

`.local/bag-native-inference-final-green.log`: **75 passed, zero skipped, four upstream KL mean-reduction warnings, 36.21s**, terminal exit0. New suite19 plus prior dense/objective56.

The new positive cases train through the actual CLI with sealed CSV/dataflow preparation and captured independent registry definitions, copy actual model/checkpoint/objective helper into a separate package, then delete the original trained run and prepared source. Expected predictions independently sum actual saved learned table rows, padding excluded, multiplied by caller-supplied weights when selected. Full and whole-group micro batches1/2/3 agree. Actual subprocesses run `python -I <relocated>/inference.py --input <records> --batch-size 1` outside the checkout and compare binary logits/class presence or objective numeric output. Empty packed groups are tested as standalone micro batches. Negative controls cover wrong role dtype, control bounds/types, malformed mapping, unknown fields, wrong binary output shape, bool IDs, vocabulary overflow, excess capacity, mispaired/nonfinite weights, caller offsets, fixed empty groups, unexpected unweighted weights, group-budget overflow and invalid batch size before allocation. A correctly resealed poisoned objective helper confirms no training-helper execution.

This is real host portable-package evidence. It does not claim native UI export, native History acceptance, or complete bundle acceptance. The native export owner separately tests archived-preview agreement against actual saved packages. Production and test sources were frozen during the final replay and remain frozen for root's serialized Swift/native checks.

## Frozen terminal pins

Full supporting source and log snapshot: `.local/bag-native-inference-final-pins.json`.

- `Sources/KwipMLApp/Resources/inference.py`: `7589d8f9c2e5c4c6b184406eb41f87ffec1b5b7c25f87d9d1cae2bd0ca5f7cbb`
- `runner/tests/test_bag_native_inference.py`: `ad3dee1f56c84179a67b73c7db6b678b0b74c1ea6de74aff268f8e778a2c9854`
- `.local/bag-native-inference-red.log`: `68abc18e6aa9ec0699c9139f97317e11d34e14c22c068050db75dfd966fe4aa4`
- `.local/bag-native-inference-integer-red.log`: `833b1f39c7fc3dd6b31412bcdbefcb148ba08b7f6adb9064cb4997e97bf5d800`
- `.local/bag-native-inference-final-green.log`: `4d4285395a7f2d3dafe66b730388187b3f2763ef0985fc3ae7af5dd7e0933265`
