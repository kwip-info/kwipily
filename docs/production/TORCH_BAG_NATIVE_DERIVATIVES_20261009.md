# Native bag derivative verification — 2026-10-09

Status: the bounded native decoder/geometry slice passed its five dedicated tests. Adjacent ordinary regressions, Phase/Session/History integration, native UI acceptance and complete component admission remain root-owned; this receipt does not claim their completion.

## Implemented boundary

`VerifiedBagPartition.itemCounts` retains the independently validated number of items in every observation (at most 4096), populated during the existing prepared-table cell verification. Full observation values remain limited to the existing first 200-row preview. Derivative ranges can extend beyond row 200; no range is silently clamped or rejected merely for exceeding that preview.

`WorkflowBagDerivativeContext(requirement:preparation:)` checks full count/row-ID cardinality, source configuration, fixed-layout capacity, preview alignment and partition disjointness. Its actual Core requirement supplies source identity, controls and enabled roles. It constructs the exact worker schema 1 `bagPacket` modelInput descriptor and selects counts using the requested partition/range.

`WorkflowDerivativeGeometry.resolve(..., bag:)` independently compares the requirement against the validated captured graph and resolves coordinate targets through Core's separate `outputBagContracts`. Fixed IDs/weights are `[rows,capacity]`; packed IDs/weights are `[sum(selectedCounts)]`; offsets are `[rows + terminal]`. Output and native parameter geometry keep their positive-dimension rules and existing budgets.

`DerivativeResults.verify` requires artifact/manifest schema 2, exact modelInput and per-request bagItemCounts only when supplied with typed bag authority. Preparation pins and all row identities must match that authority. Only independently established empty packed coordinate shape `[0]` permits zero elements; actual float-weight derivatives may be observed `[]`, while integer coordinates remain unavailable with `integerInput`. Parameters and outputs cannot acquire zero shapes. Native owner lookup now includes EmbeddingBag's actual weight shape/name, and frozen-weight flags recognize both Embedding and EmbeddingBag while retaining explicit trainability overrides.

Ordinary schema 1 exact field sets remain selected by the default nil bag context. Existing geometry/decoder call sites retain their defaults. The worker schema 1 wire format was not changed by this slice. Dedicated tests do not independently recompute learned derivatives; reported numeric values retain the same trust boundary as ordinary schema 1 results.

## Actual root-executed evidence

1. Semantic RED, `.local/bag-native-derivative-red.log`: five methods failed unexpectedly in 6.688 seconds because genuine generated fixtures reached the existing native ownership mismatch. This exposed the missing EmbeddingBag native owner/flag path before implementation.
2. First implementation replay, `.local/bag-phase-activation-red-derivative-green.log`: four derivative methods passed; the forged-count case failed to throw because its hard-coded `[0,0]` was exactly the genuine selected count vector. The test now derives genuine counts and changes the first entry modulo capacity+1, guaranteeing a different vector. No decoder validation was weakened.
3. Corrected GREEN, `.local/bag-native-combined-red-green-2.log`: **all five BagDerivativeResultsTests passed**, zero failures/unexpected failures, 9.531 seconds. Other selectors in this combined log remained intentionally RED for their separate unfinished implementation; the combined command is not claimed globally GREEN.

The five methods exercise actual Python-generated saved derivative artifacts and actual verified prepared tables, an independently derived Core requirement/registry, CPU and MPS, fixed and packed layouts, terminal-offset on/off, empty packed weights, selected training rows 201–202 beyond the 200-row preview, exact native weight name/shape, inherited/frozen/trainable overrides, repinned version/source/count/shape/status forgeries, and refusal without typed bag authority.

Root replay `.local/bag-expectation-red-adjacent-green-3.log` also passed the five bag methods (9.119s), six ordinary `DerivativeResultsTests` (1.069s), and four `WorkflowDerivativeGeometryTests` (0.028s). Two unrelated phase-wrapper tests were intentionally RED, so that combined command is not globally GREEN. Root owns serialized Swift execution and the forthcoming gradient-lifecycle/phase integration replay. This worker ran no Swift build or native UI check.
