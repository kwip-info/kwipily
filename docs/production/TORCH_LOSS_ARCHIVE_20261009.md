# Typed loss archive prerequisite · 2026-10-09

Bounded capture-format implementation under `TORCH_LOSS_CLOSURE.md`; not completed History, preparation, worker, checkpoint/export or native acceptance.

## Contract

Workflow schema 8 pairs with compiler contract 7. A selected objective stores canonical `lossRecipe` with exactly `source`, `configuration` and `targetMapping`, plus typed `objectiveScalarization`. A selected objective has no dummy `targetColumn`; an implicit objective retains the historical target column and omits both new optional fields, including under compiler 7. Schemas 1–7 retain their exact version pairs and encoding.

The wrapper enforces bounded qualified control identity, complete canonical configuration controls, the 32768-byte recipe budget, exact typed mapping/role keys, supported versions and objective identity. Capture admission checks the mapping's input columns against captured feature columns and distinct source identity against Train, optimizer and scheduler. Scalarization must exist exactly with a selected objective. Recompiled-plan comparison pins source, configuration, mapping and scalarization. Model/prediction geometry belongs to recompilation against the captured registry; this decoder does not substitute a built-in registry or assert observed data correctness.

Scheduler capture admission includes compiler 7; an objective can coexist with its genuine selected optimizer and scheduler. No training outcome or completed artifact is fabricated by these changes.

## Actual verification

Initial new test setup builds failed on whitespace and an attempted mutation of immutable `WorkflowExecutionPlan.targetColumn`; these are retained in `.local/loss-workflow-archive-red.log` and `.local/loss-workflow-archive-semantic-red.log`, not classified as semantic RED. Target injection remains exercised through actual archive JSON. The historical schema-1 fixture uses a real compiler-2 plan projected to archive contract 1, matching the existing historical convention.

Final semantic RED before production mutation: `.local/loss-workflow-archive-final-semantic-red.log`, terminal 1, five methods with six failures (five unexpected). It proves the unsupported compiler-7 projection and schema-8/compiler mismatch. Seven deterministic historical capture fingerprints were recorded under the unmodified archive implementation in this run, then frozen as assertions and the temporary printing hook removed.

Initial implementation GREEN: `.local/loss-workflow-archive-green.log`, terminal 0, 21 App methods. Final GREEN after the approved combined control test: `.local/loss-workflow-archive-final-green.log`, terminal 0, 22 App methods, zero failures, unexpected errors or skips. Suites: six new loss archive methods; five optimizer projection; five scheduler archive; six generic archive codec. No Core methods or actual worker runs were selected.

New tests cover all eight actual loss example factories compiled into selected canonical recipes; capture decode/reencode and recompilation equality; nontrivial reordered L1 target columns; changed configuration/source/scalarization mismatch; 23 malformed capture cases including foreign controls, versions, target/source/mapping/scalarization and qualified-source aliases; an actual 4096-value configured weight vector exceeding the canonical recipe byte budget; implicit/new optional omission; and a genuine StepLR + SGD + CrossEntropy workflow with distinct qualified controls in schema 8/compiler 7.

## Preserved historical canonical fingerprints

Fixed project/directory/run identities make these byte proofs repeatable. They were observed before the archive source changes and all pass unchanged in the final run.

| Workflow schema | Capture SHA256 |
|---|---|
| 1 | `51ce36147b3c001a963eb79d29199e5e2ea7a5004fcef64d1353b64337991f04` |
| 2 | `13fe88c722e795ca34acb679bc238052cac46afac09dc897d257689b7c9f347f` |
| 3 | `226dda87d01359596033a0d6e6c2140bcac8177d7d9013e59e1f0bbadf3963a0` |
| 4 | `ed0da9510d085d1e0b7630a9504c4c760e2519e0d3e92ada78046cf1b959d82f` |
| 5 | `45fb520d427e3f8cbd73310989adc6fb5debc57b34ea062353f5a94892e5baab` |
| 6 | `07802dce9c549621f50f1913efb9c5154e105ad544c1ef33c5ebb2d99fd5ef20` |
| 7 | `fe064145865d3591080149144776854fabb694b1420464d4ae7cd3043ed19b4c` |

## Final scoped identities

- `Sources/KwipMLApp/WorkflowArchiveContracts.swift` · SHA256 `31972a5cdad3980771463d7dec0ef6eb570cbf8cd4b7c59c1ef6ee568ef54662`
- `Tests/KwipMLAppTests/LossWorkflowArchiveTests.swift` · SHA256 `04dbf8b9085c28ee87a70bc5bd4d946a5e9bfb53a3ad027b563ab97af45b5472`
- `.local/loss-workflow-archive-final-semantic-red.log` · SHA256 `ceda3175f797e967acafd7168b5367e7d6c25c5f06aec69a929c8e8fe75b8e50`
- `.local/loss-workflow-archive-final-green.log` · SHA256 `dcd90d3bc285692c33e290fb43cf4bb756c7b0d92c1d2f88de0a92c8d7fc9c5e`

Prepared selection/PhaseArtifacts/History and runtime-session admission remain root/runtime-agent-owned integration gates. Native and full loss acceptance are not claimed.
