# Bag prepared evidence and objective artifact readers — 2026-10-09

Scope: native Swift artifact verification and input-independent objective result decoding. This is a component foundation receipt, not bundle acceptance, Phase/Session/History integration, or native UI evidence. Root executed all Swift commands in the serialized Xcode lane; this agent executed only the host Python fixture setup checks.

## Actual verification

- `.local/bag-native-artifacts-red.log`: five App methods, four failures (four unexpected), 6.366 seconds. Actual prepared CPU fixtures reached the intentionally rejecting prepared-evidence scaffold. The changed-byte negative passed.
- `.local/bag-gradient-artifacts-native-green.log`: earlier mixed replay passed two Core BagGradientContextTests plus seven App methods (six BagArtifactTests and one BagGradientArchiveTests), zero failures/skips; App terminal 10.187 seconds. This was the earlier nine-method proof, before shared objective evaluation was implemented.
- `.local/bag-objective-evaluation-red.log`: four methods, four failures (four unexpected), 5.216 seconds. Genuine fixtures reached the intentionally rejecting bag objective-context factory.
- `.local/bag-objective-native-green.log`: final 16 App methods passed, zero failures/skips, 15.860 seconds: six BagArtifactTests, four BagObjectiveEvaluationTests, six existing WorkflowObjectiveArtifactTests. Root reported terminal exit 0.
- The intermediate `.local/bag-gradient-native-red.log` was a compile-only attempt during implementation (explicit JSONValue string spelling); it is not semantic RED evidence.

The new tests generate fresh sealed CSV preparation and actual engine training through the host `.venv` during each test. They do not depend on cached local model artifacts. Packet readers exercise CPU binary and CPU/MPS MSE objective runs, empty packed observations, ordered input/target cells, four independent capture-permission combinations, and repinned metadata mutations. Objective reuse exercises actual CPU/MPS MSE reduction-none with captured sum scalarization, CPU index CE with measured classification, saved epoch records, and independent activation omission. Incorrect identity, shape, aggregate arithmetic, role, privacy and fabricated MSE accuracy refuse. Existing objective tests replay the dense consumer API.

## Implemented boundary

`WorkflowBagPreparedEvidence.preparedEvidence` reuses anchored inventory/table reads. It verifies the selected completed manifest pin, table bytes/length/SHA, complete mapped cells (including unused packed suffixes), separate int64 coordinates and finite float32 weights, whole-bag lengths, column schemas, bounded observation/output counts, ordered unique/disjoint partition IDs, source/split/fitted-state lineage, and referenced fitted states restricted to training observations. Only the first 200 typed observations are retained for row comparison; every mapped table row is checked. Fitted state and selected table reads retain anchored path and size protections. Existing shared-source changes only expose preparedInventory/preparedTable internally.

`WorkflowBagArtifacts` checks source/configured named-role descriptors, runtime/device, manifest/preparation/checkpoint identities, exact partition counts/order/lineage, independently recomputed membership and split hashes, independent capture status and exact bounded schema3 row count. `BagObservationInput` represents genuine empty bags with empty coordinate arrays rather than a fabricated dense tensor layout. Captured inputs and desired values must equal the prepared cells; numerical outputs remain immutable hash-bound worker results.

`WorkflowObjectiveEvaluationContext` contains no input tensor contract. Bag context creation uses the exact captured PreparedObjectiveSelection.scalarization and verified partition counts. Existing dense evaluation/epoch/metrics signatures delegate to the same arithmetic validation; dense prediction input validation remains separate. No binary class index is invented for objectives without eligible measured classification.

## Integrity limits and pending consumers

The prepared manifest and selected artifact byte hashes are independently checked. Complete source/mapping/target identity fields, lineage and observation order must match those actual prepared tables; changing both reported manifest and checkpoint metadata cannot bypass those comparisons. Python `tableDigest` and `mappingDigest` are required to be well-formed and identical between manifest provenance and checkpoint bag sources, but are **not independently recomputed** by native Swift. Torch binary tensor digests are similarly reported/pinned identities rather than native tensor recomputation. No general cross-language canonical JSON scanner or pickle interpretation was added. Worker prepared-data validation remains authoritative for its full numerical execution semantics.

This receipt does not claim live Session publication, Phase completed-artifact routing, cold History retention/export, package/native UI verification, all-eight-objective bag workflow coverage, or accepted capability credit. Those are separate integration gates.

## Frozen source and test pins

| Path | SHA-256 |
| --- | --- |
| `Sources/KwipMLApp/WorkflowBagPreparedEvidence.swift` | `6cfc3f53f70affbcb5ba8e3bed0bf8f3f11b100787028f48b3b9f9df524b4598` |
| `Sources/KwipMLApp/WorkflowBagArtifacts.swift` | `3e6b7e3f1bc276f0a613ec0cb1b25104acfbdd054179cd8ad2fb06afde8e9210` |
| `Sources/KwipMLApp/WorkflowObjectiveEvaluationContext.swift` | `d3fbcee43a186f07213615159339d8b442e21989765763f6f97bbd74a10ec035` |
| `Sources/KwipMLApp/WorkflowObjectiveArtifacts.swift` | `bd6c8bd2bac8d298c8da53df7cf220771f742d50a45fc3e70b4f4236ad0bd7ee` |
| `Sources/KwipMLApp/WorkflowFailedGradientEvidence.swift` | `6b7cb8899b209fbe2692168b973f1305bf58e2fc167cf2a68618c73c75946eae` |
| `Tests/KwipMLAppTests/BagArtifactTests.swift` | `22438052d6f7c269b6d63a9e68bffb3ed6c79b015389dc358230c45a522aed0e` |
| `Tests/KwipMLAppTests/BagObjectiveEvaluationTests.swift` | `cf082259802627408fdb45710d9bc36231462eee177f065a0f60683bc4786c76` |

## Log pins

| Log | SHA-256 |
| --- | --- |
| `.local/bag-native-artifacts-red.log` | `abe463e8d29baa6d18ac69396cd1549cb57729cc2141511d3ed8153f0f0159e6` |
| `.local/bag-gradient-artifacts-native-green.log` | `a6e6388f9755053948bf914f28a247f78bfd1578bdf3675df4e175ae3b469bfd` |
| `.local/bag-objective-evaluation-red.log` | `3680cef8ced01d2e4e941884d5e8deea31eba84049cfb70916ee4fc8ae68cf97` |
| `.local/bag-objective-native-green.log` | `43834c499879de1c1ab17b6a42f18221499dbe4a7c6fcc8aa5d822094c5625e7` |
