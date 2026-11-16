# Scheduler-parent checkpoint transport · 2026-10-09

A verified scheduled parent requires its pinned `rate-trace.json` as well as manifest, weights and metrics. Native staging and standalone training previously admitted only the three base files. This blocked actual scheduled-parent continuation despite correct numerical worker resume behavior.

## Bounded change

The shared checkpoint inventory derives the fourth-file requirement from the verified parent manifest. It checks the exact dependency, bounded bytes, SHA256, scheduler recipe, timing and record count before staging or execution. Three-file legacy parents remain supported. Archive, snapshot, operation admission, storage accounting and file writing consume the conditional inventory. The trace limit is4MB; the model-training limit becomes307MB while the global330MB cap remains unchanged. Child scheduler settings cannot erase a parent dependency. Fine-tuning may deliberately omit the child scheduler. No checkpoint engine or Torch deserialization policy changed.

## Actual RED and GREEN

- Standalone pre-deserialization RED:2 failed/4 passed, zero skips. Four-file parents were refused and scheduled parents missing their trace were incorrectly admitted. GREEN:38 passed, zero failures/skips,6.54s across the new six cases and affected training/scheduler contract checks.
- Actual App RED created a genuine CPU MSE+SGD+StepLR parent but staging omitted the trace. The same serialized run verified all seven historical count repairs and three artwork checks (ten passing methods).
- The affected-consumer replay executed96 App methods with two preexisting performance skips and one unexpected failure. All ten workspace-session methods passed, including genuine prepare/train/verified History. The remaining failure identified the final checkpoint-writer three-file allowlist. The original failing log remains RED despite its filename ending in `green.log`.
- After the narrow writer correction,10 App methods passed with zero failures/skips in19.590s. The genuine scheduled-parent test verifies four-file staging/archive, missing/tampered trace refusal, exact CPU continuation compared with uninterrupted training (including model, optimizer, RNG, objective, scheduler and rate trace), fresh fine-tuning without a child scheduler, and unchanged original parent History. Seven DiagnosticPersistence methods and two legacy checkpoint-snapshot methods also passed.

These are actual host application/worker integration checks. Earlier native MSE/CE observations remain separately documented; this receipt does not claim a newly performed native UI continuation check or final UAT.

## Frozen source and log identities

- `Sources/KwipMLApp/WorkflowCheckpointInventory.swift`: `2fb197a3144df8d24b398d46a3b7426e4ded5fb8cac04aa21348156e97a4f63a`
- `Sources/KwipMLApp/WorkflowCheckpointSnapshot.swift`: `62b035ad38d8c2892df759589432f9b090e9a528d18239a1b8958bf75d9c7e39`
- `Sources/KwipMLApp/WorkflowHistoryReader.swift`: `8756b0d299025df7ca2db2c7e219b6275158554a80c9569aa19c4c116f207a5b`
- `Sources/KwipMLApp/WorkflowLifecycleParentRecord.swift`: `dccd2220d9926e546c00dcdce276866f0e8e59f366b4b7df52709e3b2da9b778`
- `Sources/KwipMLApp/WorkflowArchiveContracts.swift`: `e397cce16c2b71846c5488fed5f5d59cb143f5770632ec6fa6fd81a16b6ee947`
- `Sources/KwipMLApp/WorkflowExecutionStorageAdmission.swift`: `8b7981081c02f73b815fc911d5dad5b82c5db9ac702980a946821e0e254365ee`
- `Sources/KwipMLApp/WorkflowModelTrainingToolContract.swift`: `7e0251f756b77fd66189fe2960578a1ae5b249684bc98f1324ac42b903fbbda9`
- `Sources/KwipMLApp/WorkflowModelTrainingExecution.swift`: `8550f039349528cf87ba7f072613a607723139bbaebc21e6238ed2af894ce1cb`
- `Sources/KwipMLApp/WorkflowPreviewCapture.swift`: `04cfb3fde2c2c70d57d42226c3b70d5ab30789f9c0e6f53c231aaf1ad8b88bec`
- `runner/model_training_operation_contract.py`: `5541bab4dbf9f433caf3ea2e25da91aeccdcef1037884f9c3fc465ef37f4f34b`
- `Tests/KwipMLAppTests/SchedulerCheckpointTransportTests.swift`: `de142f119e2971e7e1923d85b330c7aa676a5899839e5d1600725465c8fcedf7`
- `runner/tests/test_scheduler_parent_transport.py`: `a48d30461babd357576556594d0cd9327562509e322b71edac5377e41fa9b238`
- `.local/scheduler-parent-transport-python-red.log`: `ef8050bb5d7fb0a3d0c942c4d03991171441057785cf9963ac181e8ad1c1a544`
- `.local/scheduler-parent-transport-python-green.log`: `4c3bf07ccce206e151dbe91cf88b6282973e48c88aa3107cabf045e8f17ead4d`
- `.local/scheduler-parent-transport-swift-red.log`: `10e86154e8e55d2f6311d2dd4183cfc166d94122eb9ed42c9c2f03fc24258921`
- `.local/scheduler-parent-transport-swift-green.log`: `baf68d95b5636dfa44d5f2b9e30a50a2646019647482f1b4142f079288c33c62`
- `.local/scheduler-parent-writer-swift-green.log`: `5405a70e891d796f138d54c4c3a5b62799093694b02305ec741bb6063a42099a`
- `Sources/KwipMLApp/DiagnosticFileIO.swift`: `92422ae95b340698a45c5ffc9f98ea16eb782e2008cf81a763b41f0297b9d23e`
