# EmbeddingBag native authoring configuration · 2026-10-09

Status: BAG.2 configuration foundation verified, not complete component acceptance. New Core files only; no registry, graph-source, compiler, runtime admission, palette or export publication. The full [source integration contract](TORCH_EMBEDDING_BAG_SOURCE_CONTRACT_20261009.md) retains both layouts, real weights and original four modes.

EmbeddingBagConfiguration reuses existing EmbeddingConfiguration for table dimensions, padding canonicalization and initial learned-array controls. It adds the frozen configuration1 fields for reduction, fixed/packed layout, terminal-offset policy and real item-weight input availability. Twelve fields must be present exactly. Float32 norm underflow/overflow, missing/unknown fields, invalid categorical/Boolean/numeric types, table product bounds and native-invalid max/sparse/frequency/weighted interactions refuse. Canonical values and conditional named input order match the Python contract. Fixed-row terminal policy remains explicit and preserved, although native fixed-row execution ignores it.

Static shape algebra remains100M table elements and grants no allocation. The engine's general aggregate2M and selected initialization200k limits remain separate. A layout/configuration alone does not claim prepared-source support, learned state, CPU/MPS parity or an input-gradient mode.

Actual TDD: the first attempt stopped at compile-only JSONValue string-literal setup; no semantic RED claim follows from it. After correcting the fixture/scaffold to explicit JSONValue strings, the compiling passthrough baseline executed **five methods/63 expected assertion failures/zero unexpected errors**. Minimal implementation then passed **eight Core methods**, zero failures/skips: five new and three existing Embedding configuration methods. The App target matched zero methods; no App/native UI coverage is inferred.

Tests check all fixed/packed/weighted port variants, nondefault padding and numeric canonicalization, preserved flags/input immutability/roundtrip, every missing field, malformed types, static exact/overflow boundaries, float32 norm edges and invalid interaction rejection even with initial freeze. Existing ordinary Embedding semantics remain unchanged; its three methods pass.

| File/log | SHA256 |
| --- | --- |
| `Sources/KwipMLCore/EmbeddingBagConfiguration.swift` | `778f2fa1c75862d0d7d370146916beeff2ed9b12f781dea2707414045f2d859d` |
| `Tests/KwipMLCoreTests/EmbeddingBagConfigurationTests.swift` | `2de19839aed72e9bd3510f3488c6bbe79f18d42f3b503d655eaf22ef0809731f` |
| `.local/embedding-bag-core-configuration-red.log` | `987f87e931cb695a91075fd836f48ee3b513499d73bb3b809b0fa9a1ac80bf70` |
| `.local/embedding-bag-core-configuration-semantic-red.log` | `ac7ec3ef8184a154f0adb8da585d4608ba7dfaf15f91f51cea51bd2b7890efc5` |
| `.local/embedding-bag-core-configuration-green.log` | `f24bd3e432a8f3b74f040c7ab8ccb0a53abd70ebf35b5f77ceff2d98bcf8cf79` |
