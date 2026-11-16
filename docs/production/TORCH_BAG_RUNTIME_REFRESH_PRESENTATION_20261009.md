# Bag lesson runtime refresh presentation — 2026-10-09

Status: actual semantic RED recorded; bounded predicate fix implemented and frozen for root GREEN. No native observation by this agent.

Root native review found a concrete missed seam: TensorLearningStore permits refreshing Bag and Gradient lesson runtime discovery, but TensorLearningView.results maintained a second task list excluding both. Its readiness message promised an action that was absent.

Finite fix: reuse the store's existing canRefreshLessonRuntime predicate for the actual Results refresh presentation; preserve the framed container, action and disabled policy. No discovery/capability grant or alternate task list.

The new EmbeddingBagRuntimeRefreshPresentationTests method covers all six Bag task identities, all four Gradient tasks, an already-supported Initialization task, an unopened lesson and leaving for an ordinary project. The actual view branch now uses an extracted showsRuntimeRefresh presentation seam preserving existing behavior until RED. Root serializes all Swift execution. No production behavior fix has yet been applied.


## Actual RED and scoped implementation

Root terminal `.local/bag-final-gates-red.log`: the refresh presentation method executed with **20 assertion failures in 0.127 seconds**. Separate artwork/export test failures in that invocation are outside this receipt.

After root released the view lease, `showsRuntimeRefresh` was changed to delegate directly to `lesson.canRefreshLessonRuntime`. The actual Results branch consumes this property; the duplicate task list is removed. The existing button still calls `lesson.refreshLessonRuntime()` and uses the same store eligibility for disabled state. Active-project and busy-state checks are retained by the store. No capability grants, discovery changes or unrelated lesson behavior were added.

GREEN remains pending root's serialized replay. This agent ran no Swift builds or tests.

## Actual scoped GREEN

Root terminal `.local/bag-final-gates-replay-green.log`: refresh presentation method passed in **0.130 seconds**; generated artwork lookup method passed separately in **0.021 seconds**. Full selected App invocation: **10 methods passed**, zero failures, in **49.080 seconds**. This closes the automated refresh presentation regression. Root separately reported an all-bag Python union of **658 passed in 65.81 seconds**; that broader execution does not count as native observation for this view receipt.

Root retains native observations and final bundle accounting. No source edits or Swift execution by this agent in this receipt update.
