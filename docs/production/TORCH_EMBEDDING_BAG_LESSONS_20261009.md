# EmbeddingBag paired lesson implementation — bounded evidence

Status: scoped implementation complete; root Core GREEN pending. Not component publication or bundle acceptance.

## Owned scope

New `EmbeddingBagLearningSubmission.swift`, `EmbeddingBagLessonReference.swift`,
`EmbeddingBagAssessment.swift`, `EmbeddingBagExample.swift`, new Core tests,
`runner/learning_embedding_bag.py`, new Python tests and this receipt only.
Shared lesson enums, dispatch/store/catalog, registries and bundle metadata remain
root integration work. Existing generated bag icons are reused by publication;
this slice does not generate or register artwork.

## Approved contract

Submission schema1 `embeddingBagGraph` contains only the user graph. It requires
one real Bag Input, one EmbeddingBag and one Output. Enabled source roles connect
to identically named bag roles, then `out` connects to Output `in`. Names, IDs,
positions and array ordering carry no assessment meaning. Complete current source
and operator configurations are checked, including layout, terminal offset and
weight role agreement.

Private fixtures represent actual rectangular source table rows as `sourceRows`,
plus `lengths` for packed inputs and `sourceWeights` for weighted inputs. Every
cell is checked, including unused cells. Row lengths collate genuine packed
indices/offsets; empty groups produce no fake ID or sentinel. Weight slicing stays
in the differentiation graph, so dormant source-cell influence is zero through
real preparation. `table` and per-bag `upstream` are fixture-owned. Integer IDs
and offsets have no derivative available and never receive fabricated gradient
measurements. Floating table/weight derivatives are measured when enabled;
frozen table gradients are omitted rather than replaced by zeros.

The Python executor checks submitted connections before executing the actual
`CheckedEmbeddingBag` float32 product adapter and `execute_bag_source`, not a
lesson-only lookup implementation. The independent Swift reference uses explicit
row loops and dtype-aware assessment tolerances; it never calls PyTorch.

Six approved tasks: fixed sum, packed mean, packed max, fixed weighted sum,
packed weighted sum, padding plus norm. Copy introduces groups, category IDs,
vectors and influence in ordinary language. No authored answers or executable
code are accepted. Private seed-varying fixtures must not appear in handouts.

## Actual test observations

- Initial Python collection: exit2, missing executor module. This is scaffolding
  evidence, not numerical RED.
- After a rejecting executor scaffold: 22 failed, exit1, 0.47s. Tests cover actual
  repeated/empty bag arithmetic, three modes, both offset forms, weights/table
  derivatives, padding/norm/freeze, invalid roles/cells, cancellation and budget.
- Implementation: 22 passed, exit0, 0.41s.
- Added verification for all-empty mean/max, seeded independent fixed row-loop
  forward checks, renamed/reordered graph equivalence and sparse measurement.
- Combined command:
  `.venv/bin/python -m pytest -q runner/tests/test_learning_embedding_bag.py runner/tests/test_embedding_bag.py runner/tests/test_embedding_bag_packet.py`
  produced 169 passed, exit0, 1.53s.

Core tests were staged with rejecting compile scaffolds and handed to root for
serialized Swift semantic RED. Root reported actual terminal evidence in
`.local/bag-native-combined-red-green-2.log`: 3 methods, 2 unexpected failures
(reference and six-plan scaffolds), 1 negative pass, 0.316s. The four approved
Core implementations replaced those scaffolds only after that signal and were
frozen for root GREEN. No Swift build or native UI command ran in this agent.
First root GREEN candidate passed independent arithmetic and source-cell refusal,
but six roundtrip assertions exposed integral `.number(2.0)` re-decoding as
`.integer(2)`. After root terminal, submission admission was normalized locally
with exact Int64-safe integral conversion (strict upper bound below 2^63); Boolean
and nonintegral controls remain unchanged. Global JSONValue equality and tests
were not relaxed. The corrected files were frozen for root GREEN. Essential verification was then
added: all-six assessor pass/incomplete/wrong-influence checks, independent mean
and coordinate-max arithmetic, padding/norm mutation and frozen-gradient
unavailability. Root replay71853 found conditional test arrays inferred as Any;
this was a compilation failure, not test execution. Both arrays received explicit
[Double] types. Approved first-use gradient/offset definitions and plain-language
norm2 wording were applied after the terminal, then sources were frozen again.
Corrected root replay `.local/bag-expectation-red-adjacent-green-3.log` passed all six Core methods (0.020s) and `EmbeddingBagWorkerBridgeTests` (0.591s). The bridge generates all six private Swift plans (four fixtures each) and sends only source fixtures plus the submitted graph to Python. Actual production float32 adapter outputs, table mutations, table gradients and source-weight gradients match the independent Swift reference at absolute3e-6/relative3e-5. Expected answers are never sent to Python. This is direct cross-language execution evidence; shared managed dispatch and native Learn publication remain separate.

The independent reference deliberately covers these six lessons with frequency
scaling disabled; it makes no full native frequency-policy oracle claim. All
production switches remain available and actually honored by submitted native
execution. Private expected table/weight influences use explicit row arithmetic,
including first-winner max ties and padding suppression, not copied Torch output.

## Explicit remaining acceptance

Root must integrate capability admission and lesson dispatch, audience guidance,
restricted palette/draft/reopen paths, bundle registration, prepared runnable
examples and generated artwork references. Native graph construction and managed
learning evidence must be verified after those integrations. No standalone
executor test establishes managed evidence authority, Learn UI availability,
release coverage credit, device parity or UAT completion.
