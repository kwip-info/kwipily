# Background History capture validation — 2026-10-09

Root observed the packaged app quit while reopening completed native History run `EED397BA-6051-48ED-BF1B-D811321C57A3`. Diagnostic report `/Users/developer/Library/Logs/DiagnosticReports/KwipML-2026-10-09-115407.ips` identifies stack exhaustion in `WorkflowArchiveCodec.validate(WorkflowArchiveCapture)` at its prologue, reached through `decodeCapture`, `WorkflowHistoryReader.load` and the actor-backed History loader.

## Exact stack evidence

Adaptive's inspection of the compiled debug candidate measured these retained frames:

- capture validator: `0x70410` = 459,792 bytes plus 48 saved bytes;
- `decodeCapture`: `0x2080` = 8,320 bytes plus 32 saved bytes;
- `WorkflowHistoryReader.load`: `0xff10` = 65,296 bytes plus 80 saved bytes.

Their total is **533,568 bytes**, already above the 524,288-byte (512 KiB) worker stack before actor overhead. The native crash occurs in `___chkstk` at validator+36 before its body. This is not recursive capture data or a reason to loosen archive validation. Adjacent outcome and bag-identity frames are much smaller; no broad reader redesign is proposed.

## Regression staging

`WorkflowArchiveBackgroundStackTests` first added two explicit 512 KiB Thread methods for direct capture decode/re-encode and malformed packet/version/unknown-field refusal. Root process 53492 executed both successfully in 0.050 seconds. **That is positive adjacent evidence, not semantic RED:** direct codec calls do not retain the larger History reader frame and do not reproduce the full native call chain.

A third test now creates a genuine completed, guarded managed-worker bag archive using the existing `BagWorkflowSessionTests.capture` fixture, then calls the actual `WorkflowHistoryReader.load` on a 512 KiB Thread with locked result handoff and a bounded wait. It requires verified completed packet History, including its real prediction preview. Its execution is pending root; no production implementation has been changed yet.

The finite repair, after actual RED, is to split the capture validator mechanically into sequential `@inline(never)` logical helpers, preserving every check and its order. Increasing global stack size, disabling checks and changing archive contracts are outside this scope. Root owns all Swift/native runs and final native crash reproduction/recovery.

## Actual caller RED and bounded implementation

Root's `.local/bag-background-history-red.log` built successfully (3.63 seconds), started the genuine completed History regression, then terminated with signal 10 / exit 1. This reproduces the native crash through the real History caller on the explicit 512 KiB worker; it is the actual semantic RED.

After root release, the capture validator was mechanically divided into eight ordered `@inline(never)` sections: format/permissions, identity/model provenance, gradient/initialization recipes, optimizer/objective recipes, continuation lifecycle, cache provenance, file/source inventory and plan bounds. The outer function only calls these sections in the original order. Every original check and message remains verbatim. The only repeated binding is the local `plan = value.plan` needed by the final section; no authority, version, shape, allocation, privacy, pin or path checks are removed or relaxed. Reconstruction of the concatenated original section bodies was asserted byte-for-byte during the edit; its original body is retained as ignored `.local/bag-capture-validator-original-body.txt` for review.

`git diff --check` passes. Root GREEN and native recovery remain pending; no Swift/native run was performed by the subagent.

## Accepted scoped GREEN

Root's terminal `.local/bag-background-history-green.log` executed **66 App methods, zero failures, 33.166 seconds**. The three background-stack methods all pass (4.417 seconds total): actual completed managed-worker History replay on 512 KiB passes in 4.378 seconds; direct exact-byte capture replay passes in 0.028 seconds; tampered packet/version/unknown-field refusal passes in 0.011 seconds. The adjacent archive, bag History/export, legacy schema and History-store checks also pass in that union.

This establishes the real caller regression fix without a larger global stack or relaxed validation. Packaged native recovery and final bundle accounting remain root-owned and pending at this receipt update. Source and tests are frozen; this update changes no runtime bytes.

## Final compiled-frame and native recovery checks

Read-only disassembly of the rebuilt binary measures outer validate192 bytes and largest helper115,600 bytes, including saved registers. The inspected retained History→decode→validate→largest helper chain is189,520 bytes, down from533,568. The remaining334,768 bytes below512KiB is prologue accounting, not an exhaustive dynamic high-water measurement; actual512KiB History replay supplies runtime proof.

Root final packaged build exits0 and native selection of the original completed runEED397BA-6051-48ED-BF1B-D811321C57A3 now shows Verified archive, whole-group tables and saved model source. Actual native export saves successfully and its clean-import standalone inference smoke exits0. [Native receipt](TORCH_BAG_NATIVE_20261009.md). This closes the observed background History crash; broader release stress/UAT remain separately scoped.
