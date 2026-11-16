# TORCH v1 completion and LBFGS deferral

Status: owner-approved release scope revision, 2026-10-09.

Trevor explicitly directed: save LBFGS for another release, add it to a backlog table, call TORCH-* complete and move to LEARN-01 closure review. This supersedes the2026-10-08 all88 delivery requirement for v1, not the original inventory or prior accepted evidence. TORCH implementation closes at87 delivered reviewed units. Coverage remains87/88(98.86%); LBFGS is not counted as implemented. No capability, original mode or existing lesson is removed or reclassified as delivered.

Deferred original unit: `pytorch.torch.optim.LBFGS`. Rationale: owner prioritizes release learning/product closure over the last optimizer. Future complete-bundle acceptance retains configurable controls, actual repeated-closure behavior, supported objective/device/state and failure boundaries, portable/checkpoint behavior, generated artwork, example, approachable assessed lesson and meaningful tests. No v1 palette/execution support is implied.

This closes TORCH task work and P17 for the revised release scope. Gate D's backend boundary review, Gate G, UAT, clean-Mac packaging and distribution remain separate. Canonical coverage retains the88-unit denominator and records the named owner deferral; pending boundary review is not silently passed. Current LEARN work is bounded in [the seven-item closure checklist](LEARN_RELEASE_CLOSURE_20261009.md). DATA-02 remains authorized after this review; its lessons ship with its components.

## Verification of the scope revision

The original preservation guard first rejected the changed top-level deferral (one actual failure, `.local/torch-deferral-preservation-red.log`). Its only permitted new top-level delta now asserts the exact named owner-approved LBFGS deferral; all original unit semantics, modes, order, denominator, prior receipts/artwork and other catalog policy remain guarded. Exactly14 digest bindings to that updated guard were refreshed across the three newest bundles; no prior unit evidence was repinned. Final targeted acceptance/coverage-reporter union: **76 tests and10 subtests passed in28.50s**, `.local/torch-deferral-preservation-green.log`. JSON/Markdown reports regenerated; release gate remains unpassed because boundary review remains pending. Tracker/new checklist relative links and `git diff --check` passed. No app implementation changed or new native run claimed.
