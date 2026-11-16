# Gradient/trainability standalone Core configuration · 2026-10-09

TDD staging only. New GradientPolicyConfiguration and TrainabilityConfiguration expose real default canonical recipe/config values with intentionally incomplete default-only validation. Their eleven new test methods independently exercise nondefaults, exact malformed keys/types, integer versus bool/number, request/target uniqueness, conditional upstream geometry and float32 values, Unicode/canonical owner order, portable size, objective compatibility and strict complete recipe decoding. Root owns serialized Swift execution; no RED/GREEN result has yet been claimed. Native graph owner resolution is explicitly outside this standalone configuration slice.

Proposed root command: `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter 'GradientPolicyConfigurationTests|TrainabilityConfigurationTests'`. Missing symbol/compile fixture errors must be recorded as infrastructure separately; intended semantic RED is against the baseline default-only implementations.

## Actual semantic RED and selected implementation

Root serialized semantic RED terminated39388 exit1: **11 methods,111 assertion failures,0 unexpected failures** in `.local/gradient-core-configuration-red.log` (SHA256 `e5466e8fe687fd196d7a86560289daa4c175b1be39b9f88473d7bc7620a5edb3`). Types/defaults compiled; selected nondefaults and missing strict validation failed actual assertions. No missing-symbol result is counted as semantic evidence.

One positive boundary fixture was corrected after that terminal: row_start Int64.max with row_count1 cannot fit the checked endpoint; Int64.max-1/count1 is the positive standalone boundary and max/count1 now explicitly refuses. No other assertion was weakened. Full configuration implementation is now frozen pending root GREEN. It resolves only source-independent schema/geometry/portable bounds; actual captured graph port/native parameter identity, returned derivative geometry and differentiability remain later admission responsibilities. No production acceptance or numerical runtime success is claimed.

## Root scoped configuration GREEN

Root joint invocation99480 terminated exit1 because separate, intentionally rejecting learning/workflow scaffolds remained RED. The standalone configuration selectors are **11 methods passed,0 failures,0 skips**:7 GradientPolicyConfigurationTests and4 TrainabilityConfigurationTests. Log `.local/gradient-config-green-learning-workflow-red.log` SHA256 `33a3c6add2a4d22514805f6976227df24615eadeb8187c300e0c9136d721357a`. This is scoped GREEN, not a claim that the combined invocation or runtime execution passed. No child Swift build was run.

Frozen configuration/test source pins:

- `Sources/KwipMLCore/GradientPolicyConfiguration.swift`: `d1c775bd12160c4d408fa35425cff1e14e79d3093cc4c1cb6b41cb2090061260`
- `Sources/KwipMLCore/TrainabilityConfiguration.swift`: `95f79d7d2e508b6a5e97e7a8b5e2c7d93c3e75e21da90aed217f19f857d89d27`
- `Tests/KwipMLCoreTests/GradientPolicyConfigurationTests.swift`: `2737003ab055012fc7bc920f130eb0b7d37f8faba72dc96ddb172fafbfc16aa2`
- `Tests/KwipMLCoreTests/TrainabilityConfigurationTests.swift`: `d22412797ed3e7ddf601fccd4d182c8e865d810329c394752f324fd0b6eb75d8`

All four sources remain frozen. Shared workflow registration, actual native resolution, runtime derivatives/update arithmetic, portable/native proofs and catalog acceptance retain separate unfulfilled gates.
