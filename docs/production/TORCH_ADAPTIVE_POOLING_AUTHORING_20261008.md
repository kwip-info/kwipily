# Additive averaging and adaptive pooling authoring · 2026-10-08

State: authored metadata and mutation checks pass; integrated runtime, lesson, export and native acceptance are still open. This document does not award capability coverage.

## Finite scope and ownership

Four new entries extend the existing declarative Pooling family: `avg_pool2d`, `adaptive_avg_pool1d`, `adaptive_avg_pool2d` and `adaptive_max_pool2d`. The authoring agent owns the bundle, shared generator, scoped authoring tests and this receipt. Root owns generated original artwork, native bridges/configuration, examples, assessed lessons, native testing and final acceptance; the runtime agent owns worker execution and numerical tests. No additional pooling operations, framework-wide coverage or release acceptance is inferred.

Each new row has its own type identity, green generated artwork reference, configuration, schema-2 float32 transport descriptor, example factory and assessed lesson identity. The four previously accepted rows are preserved without mutation. The bundle remains schema 1/version 1; adding these entries does not change any existing per-component configuration or adapter version.

## Reviewed controls and boundaries

- AvgPool2d exposes independent height/width pairs for window, stride and padding, partial final window handling, padding contribution and an optional custom divisor. Empty stride follows the window; empty divisor uses the window count. The nonzero signed divisor range is −2147483647…2147483647. The shared nullable field declares `enabledValue: 1`, so enabling the control selects an admissible initial value.
- AdaptiveAvgPool1d exposes output length, an integer from 1 through 4096.
- AdaptiveAvgPool2d exposes output height and width independently. Each accepts an integer from 1 through 4096, or empty to retain that input dimension.
- AdaptiveMaxPool2d adds a winning-position switch and a separate conditional int64 `indices` port. Positions refer to the flattened original grid within one channel, with first-position tie selection.

Every descriptor preserves observations and channels. Channels are bounded to 1…512 and spatial input dimensions to 1…4096. Adaptive outputs are additionally limited to 100000000 elements before allocation. The generated worker table calls the dedicated AvgPool2d resolver and binds each adaptive ID explicitly to the shared adaptive resolver and capability function; none aliases an existing MaxPool component.

The generator pins the reviewed definitions and runtime transport descriptors independently of the runtime modules. It rejects changed defaults, bounds, types, rank/schema/dtype claims, invented controls or outputs, unreviewed conditions, mismatched icons/themes, unknown lesson/example identities, and missing or escaping source references. Published entries require learning content and actual artwork references.

## Learning authoring

Each entry explains its practical controls, output geometry and input influence in plain language, defining windows, stride, divisor, adaptive regions and indices at first use. Required text avoids advanced unexplained symbols. Adaptive text explicitly explains overlapping regions and that larger outputs do not create new measurements. Constructed table-feature grids are identified as examples rather than claims of natural spatial or temporal meaning.

The matching assessed identities are:

| Component | Lesson |
|---|---|
| AvgPool2d | `assessment.avg-pool2d-windows` |
| AdaptiveAvgPool1d | `assessment.adaptive-avg-pool1d-bins` |
| AdaptiveAvgPool2d | `assessment.adaptive-avg-pool2d-bins` |
| AdaptiveMaxPool2d | `assessment.adaptive-max-pool2d-bins` |

Matching runnable examples use distinct `preparedAvgPool2dClassifier`, `preparedAdaptiveAvgPool1dClassifier`, `preparedAdaptiveAvgPool2dClassifier` and `preparedAdaptiveMaxPool2dClassifier` factories. Actual independent grading and example execution remain root acceptance requirements.

## Observed TDD evidence

Command: `.venv/bin/python -m pytest scripts/tests/test_adaptive_pooling_bundles.py -q`.

- RED: 62 failures, zero passes; new identities, controls and generated factories were absent. Log: `.local/adaptive-pooling-authoring-red.log`.
- GREEN: 67 passed, zero skipped. The original 62 checks are joined by four exact actual-worker capability comparisons and one semantic freeze check covering all four accepted rows. Log: `.local/adaptive-pooling-authoring-green.log`.

The 56 transport mutation cases first validate the unmodified metadata against an explicitly temporary reference tree, then demand rejection of the individual mutation. That isolates authoring policy from concurrently arriving source/art files; it does **not** establish that production references exist. Whole-repository generation and `--check` must pass against the actual repository before final acceptance. No Swift build, worker execution, native interaction or artwork generation was performed by the authoring agent.

After the root's example/configuration/art files and the engine agent's actual `runner/tests/test_adaptive_pooling_engine.py` arrived, `.venv/bin/python scripts/component_bundles.py` completed against the real repository and `.venv/bin/python scripts/component_bundles.py --check` passed. Generated registry membership is 37 primitives, with eight Pooling entries; this is a registry count, not a coverage claim. All actual source/artwork references now exist.

Eleven unaffected-family authoring suites also passed: 337 checks, zero skips, log `.local/adaptive-pooling-prior-authoring.log`. The existing whole-generator/old-pooling tests contain explicit historical count/table assumptions which root is updating to recognize the additive entries; their pending update is not represented as a passing replay here.
