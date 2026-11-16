# Core implementation evidence

> **Archived snapshot — reconciled 2026-10-09.** This records an earlier implementation or plan, not the current catalog, layout, release status or work queue. Retained at this path for existing evidence links. Start with the [documentation guide](README.md), [current tracker](production/TASK_TRACKER.md) and current source before using its historical details.

Owner: graph_core agent. User-facing work date: 2026-10-04 (America/Chicago).

## Delivered

- Public Codable/Equatable graph document, node, edge types with explicit identity, generic user label, separate implementation type, numeric configuration and canvas layout.
- A versioned shared component registry at `Sources/KwipMLCore/Resources/components.json`. Swift loads its bundled copy; the Python runner consumes the same source registry. Seven supported primitives: Input, Linear, ReLU, Tanh, Concatenate, Add, Output.
- Public validation result with deterministic issue identities, actionable messages, per-node feature widths, and topological order.
- Validation of schema version, component/edge identities, finite positions, known fields, required integer dimensions, port existence, one incoming edge per input port, input/output cardinality, cycle detection, complete input-to-output reachability, linear dimensions, add equality, concatenation widths, and fixed POC dataset/classification boundaries.
- Reference parallel classifier with eight components and eight connections. Two independent 2→8 Linear branches use ReLU/Tanh, merge to 16 features, then project to two output logits. Generic user-facing names are deliberately independent of component types.
- Invalid drafts remain serializable; validation gates running, not saving.

## Actual TDD evidence

Tests were written before validator implementation. Initial repository test attempts were blocked by package setup: Core had no Swift source, then the application target was still empty. Those are setup failures, not functional red evidence.

To avoid waiting on another agent's application target, copied only the owned Core sources/resources and test sources into `/tmp/kwipml-core-tdd`, with a minimal Core-only Swift package. No production package settings were changed.

Command: `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --package-path /tmp/kwipml-core-tdd --filter GraphTests`.

Observed red: seven tests executed, 18 assertion failures against a placeholder validator returning an empty result. Failures included malformed graph acceptance and missing propagated widths/order.

Implemented validator, copied the updated validator into the same temporary package, repeated the exact command.

Observed green: seven tests executed, zero failures. Coverage includes registry/reference model, Codable invalid-draft roundtrip with custom name/layout, duplicate identities, unknown components/schema versions, wrong ports/missing nodes/multiplicity, disconnected nodes/cycles, linear and Add shape rules, fractional dimensions, unknown fields, and fixed classifier input/output constraints.

## Scope and limitations

The feature axis is the only modeled tensor dimension. Batch size remains implicit. This is a feed-forward POC validator, not arbitrary tensor algebra, sequence-state semantics or third-party extension installation. The bundled two-moons data requires exactly two input features; classification requires exactly two output logits. Both restrictions are surfaced as actionable issues, not silently coerced values.

The shared JSON registry supplies UI metadata and port/field contracts. Shape behaviors still require matching Swift and Python implementations; introducing arbitrary new modules requires code plus tests. Registry metadata alone does not claim full plugin support.

Reference graph positions occupy about 1105×384 canvas points; UI may fit/zoom or scroll to show the full graph depending on panel width.

## Contract review and second red/green cycle

Reviewed Python validator against Swift after first integration. Found substantive drift: missing fields silently defaulted on Python, unknown fields ignored there, parameter-free graphs accepted by Swift, and resource/identity limits only enforced by Python. Communicated these directly to runner agent; runner is making supplied configuration strict rather than changing the graph's explicit saved settings.

Added tests before Swift changes. Actual repository command `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter GraphTests.testParameterFreeGraphRejectedBeforeRunning` produced two failing assertions. Added limits/budget tests; full GraphTests run executed ten tests with seven assertion failures before fixes. After implementing the corrections, the same repository `swift test --filter GraphTests` executed ten tests with zero failures.

Swift now rejects parameter-free execution with an actionable instruction to add a Linear component. It also matches Python's limits: 128 nodes, 256 connections, 128 Unicode-scalar node-ID characters without dots, 2 million parameters, and intermediate width 4096. Width propagation stops above the cap to prevent integer overflow through repeated concatenation.

Remaining intentional envelope differences: Swift Codable requires serialized name/layout/config keys and typed strings/numbers; runner can execute graphs without presentation metadata. Runner will validate finite positions if supplied. Both allow finite negative positions (layout concern, not model semantics). Swift Codable rejects boolean numeric values, while Python requires explicit boolean rejection because bool is a Python int subtype; this was called out for schemaVersion and configuration. Defaults belong to node construction, not silently changing execution configuration.

## Project-library extension following owner steering

Added `ProjectLibrary.swift` with the requested synchronous public API: list, save with optional stable project ID, load by ID, independent import, and document URL access. Projects are UUID-named `.kwipml` files; the recent-project index is derived from actual documents and modification dates. Saves use atomic file replacement. This avoids a graph/index consistency failure and preserves other projects. Corrupt documents remain on disk, are omitted from the healthy-project list, and report errors on explicit load.

Document-level checks intentionally allow invalid model drafts. They require schema 1, at most 128 nodes/256 connections/2 MB serialized data, finite numeric data, and regular files instead of symbolic links. UUID validation prevents path traversal. No deletion API or migration is introduced. Store is documented single-writer; concurrent multi-process edits are outside this POC.

Actual TDD: six ProjectLibrary tests ran against stub APIs, five failed with thrown unimplemented-save errors. Implemented behavior, then added execution-settings tests before the corresponding graph API. The execution tests produced actual compilation failure for missing ExecutionSettings/graph.execution. Added the API and reran `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter ProjectLibraryTests`: eight tests passed with zero failures.

Added optional `GraphDocument.execution` holding the requested Codable/Equatable device, epochs, datasetCSV, featureColumns, and targetColumn. Old graphs without execution decode with nil. Project save/reopen preserves custom CSV settings and user labels exactly; the app remains responsible for selecting the default ExecutionSettings on a legacy graph. CSV paths are references, not copies of private data. Moving/removing a referenced CSV may require user repair and must not silently substitute the bundled dataset.

## Independent integration review

Read-only review of WorkspaceStore, ProjectActions, DataActions and views identified and reported these concrete integration issues to root:

1. Canvas summary converted arbitrary Double configuration directly to Int. A finite oversized imported invalid draft such as `in_features=1e100` passes document-level preservation checks but traps while rendering; validator rejection alone cannot prevent that UI crash. Requested safe formatting/checked conversion.
2. CSV result freshness compared configuration/path but not source content. Editing a CSV in place could leave older metrics marked current. Manifest already includes sourceHash; requested content verification or explicit unverified status.
3. Autosave errors were swallowed while edits persisted in memory; addNode then overwrote status. The 129th node exceeds library limits and can be lost on project switch. Requested bounds and preserving visible save errors, with failed-save switching protection.
4. Project switch did not reset dataRowCount/isInspectingData/eventLog, allowing prior source inspection state to remain visible in the next project.

These are source-review findings, not computer-use observations. Root owns integration fixes and UAT verification.

## Pre-training source generation and native code presentation

Owner-requested Code tab correction now generates Python for a valid current graph before training or data selection. CodeActions launches the shared real runner export path against an immutable local snapshot; token and graph-signature checks prevent an old asynchronous result replacing a newer graph's source. Temporary request directories are removed after process completion. Invalid graphs show actionable validation; runtime failure is explicit.

Current architecture and Last trained source are separate views. Native AppKit read-only NSTextView supports selection, themed syntax colors, horizontal/vertical scrolling, copying and `.py` save. Export bytes remain unchanged by highlighting. Checkpoints remain separate from model-definition code.

Syntax tests were written first against an empty tokenizer: three tests produced nine actual assertion failures. After implementation, `swift test --filter 'PythonSyntaxTests|CodeGenerationTests'` executed six tests with zero failures. Token tests cover keywords, numbers, comments, callable names, strings containing hash/keywords, escaped quotes, Unicode range safety and multiline strings. This is a presentation lexer for generated Python, not a full Python parser.

Generation tests verify actionable invalid-graph state, a deliberately delayed native-process fixture rejecting stale generation, and real Python code generation with a nonexistent CSV path before any training. The real integration produced source with no metrics or run artifacts, confirming code generation does not require opening data. Native display/copy/save interaction remains root computer-use UAT responsibility.
