# Bag activation and live completion verification — 2026-10-09

Scoped native consumer implementation; no bundle, live Session, cold History, or UI acceptance credit. Root alone executed serialized Swift builds/tests. Shared bag preparation and schema3 row verification are recorded separately in TORCH_BAG_NATIVE_ARTIFACTS_20261009.md.

## Actual RED and GREEN

- Initial `.local/bag-phase-activation-red-derivative-green.log`: four activation methods failed while decoding fixture nodes without required GraphNode name/x/y. This was a setup failure, not semantic RED. The fixture now supplies GraphDocument.name and every node's name/x/y before actual engine execution.
- Corrected `.local/bag-native-combined-red-green-2.log`: four activation methods, three failures (three unexpected), one negative passed, 5.295 seconds. Genuine CPU/empty/objective fixtures reached the existing tensor-preview contract refusal. This is the meaningful RED.
- `.local/bag-native-phase-integration-green.log`: four BagActivationArtifactTests passed, zero failures/skips, 5.600 seconds. Actual CPU/MPS coordinate activations and completion, all-empty packed coordinates, objective omission/context binding, and mutation/authority/ordinary-zero negatives passed. Two Phase and five bag derivative methods also passed in this mixed run.
- The mixed command was not an overall GREEN: six prior DerivativeResultsTests refused missing KWIPML_TEST_DEVELOPMENT_ROOT before their worker fixtures; Core roundtrip fixture assertions were also being repaired by root. These do not establish a source regression or adjacent-suite clearance. Full correctly configured adjacent replay remains pending.

## Implemented contract

The optional bagContext parameters retain defaults on verifyPresentation, verifyActivationPreviews, and saved activation decode. Absent-context dense paths retain their prior implementation. Bag paths require the captured registry and a distinct packet context. The helper independently rederives the actual BagWorkflowInputRequirement and output contracts from that captured graph/registry.

Coordinate previews use the first min(8, validationCount) complete verified prepared bags. Fixed item/weight geometry is [bags, capacity]. Packed item/weight geometry is [sum of those bag lengths], with offsets [bags + terminalOffsetFlag]. Enabled named roles, int64 coordinates, float32 weights, exact flattened prefix values, byte-pinned artifact identity and independent activation capture status must match. A zero item extent is admitted only when those actual prepared bags contain no items. It does not allow zero ordinary activation dimensions. Normal computed outputs retain positive [bags] + captured sample geometry, exact named ports/types, bounded prefix values, and binary integer attention-mask checks.

Live completion requires exact run/device/evaluation count, no legacy dense prediction array, and captured permission. Schema3 rows are verified through WorkflowBagArtifacts using a bounded synthetic envelope; its locally computed hash is explicitly not an archived-file integrity claim. Objective bags require the actual supplied WorkflowObjectiveEvaluationContext with matching recipe/source, output shape, verified counts, captured scalarization, device and privacy. The shared decoder verifies measured aggregate score and only eligible classification accuracy. No fake dense input or invented epoch horizon is introduced.

Tests generate their real preparation/training/completion and saved activation artifacts during execution using the host .venv and explicit captured registry fixtures. No cached native artifacts, signed runtime, or UI automation were used by this agent.

## Frozen source/test pins

| Path | SHA-256 |
| --- | --- |
| `Sources/KwipMLApp/WorkflowBagActivationEvidence.swift` | `07381aa01bb65c3d5f52ff60c2451de6ab1583887cfc97da329fcce7c5579bf8` |
| `Sources/KwipMLApp/WorkflowModelTrainingExecution.swift` | `7c1554bc53dd5d7924ab8883ea2302ea2070829703101cde276c76ff2faa8d01` |
| `Sources/KwipMLApp/WorkflowActivationArtifact.swift` | `25e53238330ac0a46a2033638a28c552812e55c84ed5ab22f1f7699d1d9f25e2` |
| `Tests/KwipMLAppTests/BagActivationArtifactTests.swift` | `e442f098644e619336705ff985b80d9298c5e0e1978bde863cd4f46211e8160d` |

## Log pins

| Log | SHA-256 |
| --- | --- |
| `.local/bag-phase-activation-red-derivative-green.log` | `3314d30e1e7e208bd763dd7f80bf5054101749bae6a4d7e69083a1778aaf4e90` |
| `.local/bag-native-combined-red-green-2.log` | `2bdc41b195a7995e6a6fa98cc49390a725f2f37d8878fed613b937c18ac07a18` |
| `.local/bag-native-phase-integration-green.log` | `d8156b3ca63d0a1f6caab345d606da9c7469783500b0be8ad5a0c73e772fdd7d` |
