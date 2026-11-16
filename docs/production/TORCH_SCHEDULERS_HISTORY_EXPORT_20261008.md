# Scheduler History and trained export — 2026-10-08

This bounded change retains verified scheduler evidence when a completed run is reopened or exported. It does not grant coverage or release acceptance.

## Boundary

Owned production files: `Sources/KwipMLApp/WorkflowHistoryReader.swift` and `Sources/KwipMLApp/WorkflowTrainedExport.swift`. Tests: `Tests/KwipMLAppTests/SchedulerHistoryExportTests.swift`.

The existing cold History path recompiles the captured workflow and requires equality of the independently verified phase inventory and saved outcome inventory. The scheduler artifact verifier belongs to the runtime integration worker. This exporter reopens the pinned `rate-trace.json` bytes after that verification, bounds them to 4,000,000 UTF-8 bytes, and inventories the exact original bytes in the portable package. It never creates a trace from current UI state.

The portable manifest includes the optional canonical source-qualified scheduler recipe. When absent, its optional encoding is omitted, preserving the old export manifest shape. Existing unscheduled package files and README remain unchanged. Scheduled README text distinguishes actual used/next rates from quality scores, constructor initialization from ticks, fresh state from checkpoint restoration, batch versus completed-validation timing, and the unused terminal OneCycle next rate.

`inference.py` is outside this lease. Root owns admitting the optional trace with the same 4 MB bound and hash verification; inference must not execute it.

## Actual RED and implementation

Root serialized run `.local/scheduler-artifact-green-history-export-red.log` completed actual CPU and MPS scheduled training and independent cold History verification. Both new methods then failed at the missing exported `rate-trace.json` inventory entry: **2 failures, 0 unexpected**, approximately 7 seconds. The same candidate's 10 artifact methods and 5 archive methods passed. This is semantic RED, not a compile or synthetic-registry claim.

The narrow implementation is complete; `git diff --check` passes. Serialized GREEN is pending root's run. No Swift, packaging or native UI command was run by this worker.

## Test criteria

Two real-worker methods require the explicitly enabled development runtime, including actual CPU and MPS availability. They retire the live host before reading History, compare the captured recipe/source and exact trace pin, and require exported bytes, pinned inventory, canonical scheduler identity and explanatory README.

CPU additionally checks changed bytes, a 4,000,001-byte replacement, omitted outcome inventory, changed staged bytes before publication, and a repinned capture whose scheduler origin no longer reproduces from the saved graph. These negative checks are reached only after the positive portable-export requirement succeeds. The worker, not the native exporter, validates Torch binary tensors, optimizer counters and scheduler state.
