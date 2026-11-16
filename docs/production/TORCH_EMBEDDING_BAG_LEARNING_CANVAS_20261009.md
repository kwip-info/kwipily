# EmbeddingBag typed lesson canvas

Status: scoped Core canvas checks passed; source frozen. Publication remains separate.

Owned files: new `EmbeddingBagLearningCanvas.swift`, existing
`TensorLearningCanvas.swift`, existing App `TensorLearningCanvasAdapter.swift`,
new `EmbeddingBagLearningCanvasTests.swift`, and this receipt. No lifecycle,
worker, store/readiness/catalog, authored bundle, generated registry or artwork
changes are authorized in this slice.

The new helper filters caller-owned actual Bag Input and EmbeddingBag definitions
and adds a fixture-bound Output. It must refuse unavailable or duplicate bag
definitions rather than invent a parallel lesson declaration. The existing
nonthrowing palette route may return an empty palette until root publishes the
actual bundle; inspection must explain the unavailable definitions. Ordinary Input
and unrelated tensor tools must never be a fallback for bag lessons.

Inspection reuses the shared GraphValidator with explicit-objective output
geometry and strict incomplete/complete bag submission checks. Source role
contracts remain `outputBagContracts`, with no fake dense source tensor layout.
Reduced results are real observation-first float32 vectors. Config-derived
role tooltips describe fixed rows versus packed items, true offset boundaries,
empty groups and paired real weights. They never populate measured evidence from
an inferred declaration. Partial construction remains editable; complete checking
requires matching controls and all enabled named roles.

Tests use explicit field-bearing registry fixtures because the authored bag bundle
is not yet published. These fixtures are not admission or full Learn UI evidence.
They cover exact restricted palette, missing/duplicate publication, all six typed
constructions, partial source-role propagation, meaningful disabled-role and
integer/weight derivative tooltips, and shared no-code connection commands.

No Swift or native command runs in this agent. Root owns serialized semantic
RED/GREEN and later native application checks. Full actual-palette insert/configure
and lesson UI acceptance require the separate root-owned publication unit.

## Actual observations

Root semantic Core RED in `.local/bag-results-canvas-red.log`: five methods
executed, 16 failures (4 unexpected), 0.325 seconds (suite wall0.326). These were
actual rejecting-scaffold/unknown-tooltip failures. A later unrelated App test
crashed on its test-side forced unwrap; no combined/global RED count is claimed.
Implementation began only after root released terminal8711.

The completed adapter dispatches the bag helper for validation and port
presentation, permits at most one of each tool, and reuses existing field
configuration/insertion/connection commands. It uses config-derived group
summaries and withholds private assessment values. Bundled bag palette projection
is cached for the immutable app-process lifetime so port rendering never repeatedly
decodes the entire registry JSON. The injectable helper accepts actual caller
registry definitions; it does not cache mutable caller state.

Root Core GREEN in `.local/bag-session-results-red-canvas-green.log`: five methods
passed, zero failures, 0.021 seconds. The tests used the explicit caller-owned
definition fixtures described above. No Swift/native command ran in this agent.
No full actual Learn palette or native insertion claim is made before authored
publication. Root owns adjacent checks and commit; a separate session-discovery
fixture issue is not a failure of these scoped canvas methods.
