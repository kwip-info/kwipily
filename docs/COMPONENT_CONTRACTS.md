# Concrete POC component contracts

> **Archived snapshot — reconciled 2026-10-09.** This records an earlier implementation or plan, not the current catalog, layout, release status or work queue. Retained at this path for existing evidence links. Start with the [documentation guide](README.md), [current tracker](production/TASK_TRACKER.md) and current source before using its historical details.

> Historical seven-component POC contract. This is not the current catalog: the current catalog is described by the generated registry and reviewed coverage report, not the historical counts below. See [current v1 review](production/V1_PROGRESS_AUDIT.md) and [component bundles](../components/README.md) for current scope and authoring boundaries.

Authoring identity is stable node ID; user label is separate from component type. Each primitive has a shared icon and accent in palette, canvas and inspector. These seven components are the current supported catalog, not a claim of an unrestricted plugin catalog.

Every tensor port carries a float32 tensor `[batch, features]`. Batch is preserved. Connections reference stable IDs and named ports. Each input takes exactly one wire; outputs may fan out. Unknown ports, duplicate input connections, cycles, disconnected components and incompatible widths prevent execution. Invalid drafts remain saveable within document resource bounds.

| Component | Purpose | Inputs → output | Editable fields | Shape/behavior |
|---|---|---|---|---|
| Input | Receive the data adapter's mapped measurements | none → out | features | `[B,features]`; current dataset adapter supplies exactly two mapped features |
| Linear | Learn a projection with independent weights and bias | in → out | in_features, out_features | `[B,I] → [B,O]`; incoming width must equal I; parameter count `(I+1)×O` |
| ReLU | Apply elementwise rectification | in → out | none | Preserves shape; negative values become zero |
| Tanh | Apply elementwise hyperbolic tangent | in → out | none | Preserves shape; output values between −1 and 1 |
| Concatenate | Join two feature representations | a,b → out | none | `[B,A] + [B,Bwidth] → [B,A+Bwidth]`; order is a followed by b |
| Add | Combine representations by elementwise sum | a,b → out | none | Requires identical widths; output width unchanged |
| Output | Designate the classification result | in → none | none | Receives exactly two raw class scores per observation; training applies classification loss |

## Parameter controls

Inspector uses one shared flat integer field with decrement/increment controls, visible range and contextual hint. All feature fields accept whole numbers 1…512. Text commits on Return or focus loss. Invalid text remains visible with a correction message; it is never silently clamped. Buttons are disabled at range endpoints. Input-width changes can invalidate downstream shapes; output-width changes can change the following component's required width. The canvas should show unresolved/error state until repaired.

Stable accessibility labels include the field title, `Decrease <field>`, `Increase <field>`, `Block name`, `Connection source`, `Target input <port>`, `Connect selected source`, `Preview to here`, and `Run from here`. Inspector is designed as one scrollable column for a 260-point dock.

User labels edit independently and never change implementation type. Source menus show user labels; port contract rows show upstream label, upstream port and propagated tensor shape. Output rows show shape and downstream connection count. No immutable label implies a specific domain meaning such as token IDs.

## Execution and capture states

`Preview to here` asks the runner to inspect inference through the selected node on a small batch. The button requires a valid graph and an idle runner. `Run from here` requires a valid stored preview cache according to the root store's signature and selected-node rules. Captures must explicitly identify whether weights are seeded initial values or a trained checkpoint. This inspector consumes those statuses rather than inventing them.

Preview caching is inference-only. It must not imply that a cached detached tensor supports training gradients, that training begins at the selected node, or that one completed branch substitutes for the other input to a merge. Full model training executes the complete graph. Root runner/store owns dependency planning, signatures, provenance and resume verification.

Views must distinguish: selected block, compatible target port, disconnected input, shape error, unresolved shape, stage preview in progress, cached valid tensor, and invalidated cache. A component with no editable fields still exposes purpose, ports, shape and preview actions.

## Implementation/verification boundary

Inspector and shared ParameterControl added as UI-first work, with no fabricated unit-test red claim. Their integer domain is already covered by Core bounds tests; UI commit/focus behavior requires root computer-use UAT. Initial integrated build reached these files but was blocked by the intentionally pending root preview APIs. Final build/UAT evidence is recorded by root after integration. Component behavior is implemented by Swift validation plus corresponding Python execution; changing the catalog requires both and parity tests.
