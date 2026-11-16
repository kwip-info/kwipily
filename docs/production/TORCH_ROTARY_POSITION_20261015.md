# Rotary position implementation checkpoint — 2026-10-15

Implements the [approved inline-only design](../design/ROTARY_POSITION_20261015.md).
Exactly Query/Key tensor inputs and outputs; Value bypasses. Sequential offset and
scale are Configuration fields. No position recipe, supplied-position edge, learned
state or persistent cache. Rank3/rank4 float32, explicit sequence axis, adjacent or
split-half pairing, full or even prefix rotation, bounded base/positions/resources.
Shape contracts permit differing Q/K head counts without expanding them.

The isolated worker and standalone model export share the real float32 rotation
kernel. No complex-number dependency is needed on MPS. Architecture signatures
include the settings; changed configurations cannot silently resume old state.
The editable causal token practice uses RMSNorm, Q/K/V projections, rotary Q/K,
causal scaled attention and the existing Train/Generate/tokenizer contracts.
The Layers asset is the separately regenerated revision2 icon.

## Automated evidence

Ignored logs are under `.local/rope-evidence/` in the development checkout.
Initial RED: unsupported rotary operation. Final new Python tests:31 passed, covering
CPU/MPS independent scalar oracle, norm/suffix/position0, gradients, shift invariance,
causal future perturbation, invalid inputs/budgets, graph and standalone execution,
exact CPU resume and configuration refusal. The405-case affected family batch had20
old attention export failures caused by capability metadata entering standalone code;
that boundary was corrected and all20 plus31 rotary cases passed the targeted rerun.
After fixing the practice source-resource identity,184 bundle/negative mutations
passed (`authoring-final-2.log`). These overlapping runs are not additive coverage.

Swift affected family cases were exercised (`swift-family-2.log`). New core practice
and strict-configuration tests passed after correcting the test compiler version;
inspector registration passed. Real managed Train→Generate→export replay passed
(`swift-replay-5.log`,5.56s). Earlier failed integration attempts caught the wrong
practice CSV binding and a test's incorrect export path; neither is claimed as passing.
No full-suite or release acceptance is claimed for this scoped change.

## Native evidence

Development identity only. Opened the actual lesson and its causal practice, checked
planned Q/K, inline positions/options and exactly two data inputs/outputs. Custom
width3 disables Apply and Run; Revert restores defaults. Real Run Flow completed,
Query and Key both displayed captured float32 matrices in the existing viewer.
Generate then completed16 tokens from captured weights. Native source project:
`910F42CD-ABFD-4C18-81CC-35DE6CAA9E38`; no quality claim from this tiny fixture.
Owner trial remains pending. This is not full Llama, a KV cache, heatmap or release.
