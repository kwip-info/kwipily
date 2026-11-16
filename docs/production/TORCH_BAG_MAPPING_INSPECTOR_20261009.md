# Bag mapping inspector repair — 2026-10-09

Scope: the actual universal-canvas Training data inspector, binary and explicit-objective mappings. This is a bounded authoring repair, not new data operators or packet runtime semantics.

## Observed defect and actual RED

`WorkflowControlInspectorView` selected only the objective or ordinary dense editor. For binary bags the dense Apply action replaced `bagInput` with `featureColumns`; explicit-objective bags tried to resolve an ordinary dense input contract and could not edit grouped bindings. Root's actual `.local/bag-mapping-inspector-red.log` executed six App methods with nine failures (six unexpected), 0.854 seconds, after a successful compile. The log directly observes dense Apply erasing the packet mapping. Rejecting scaffolds were used for the missing editor APIs; no implementation GREEN is claimed here yet.

## Bounded implementation

- `WorkflowControlNodeContext` selects grouped authoring from stored bindings or the actual connected model, including scratch mappings. A broken connected primitive source remains on its repair route. Dense Apply refuses grouped mappings. All commits re-resolve actual source requirements and use the existing selected-node compare-and-swap authority and one owner transaction.
- New `BagInputMappingInspector.swift` contains the ordered draft, binary editor context and reusable square framed source-column controls. Item IDs and optional real item weights have stable paired row identities and move together. Exact column names are preserved. Capacity/layout/weight/final-boundary controls stay on the actual Bag Input source; the mapping does not duplicate them. Packed lengths explain zero/empty groups without fabricated dense shapes or user-entered JSON.
- Existing objective authoring reuses its desired-outcome role and geometry controls. Grouped input columns are derived from the shared packet draft, with separate target roles. It validates the actual prediction contract and retains `bagInput` on Apply; no second editable ordinary input list is shown.
- Existing pending-edit origins, busy/retired context checks, Apply/Discard and undo/redo remain authoritative. A retained draft cannot certify a changed source/model.

## Focused verification

`BagInputMappingInspectorTests` stages six methods covering the published fixed example and connected scratch creation; dense-overwrite refusal; exact names, pending/discard and undo/redo; packed weighted paired reorder; role overlap, duplicate names, empty length and capacity refusal; actual cross-entropy roles with derived packet inputs; busy/retired and changed-source refusal. Undo assertions compare authored nodes/edges, because undo correctly advances document revision.

GREEN, adjacent ordinary/objective regression checks and packaged/native inspector acceptance are pending root execution. The subagent has not run Swift or native UI. This receipt does not grant complete-bundle acceptance.

## Root replay progress (not final GREEN)

- `.local/bag-mapping-inspector-green.log` and `.local/bag-mapping-inspector-final-green.log` stopped at compile prerequisites: Swift's property-wrapper initialization rules. Local initial draft/baseline values now initialize both contexts before any wrapped property is read. The binary presentation uses a child `@ObservedObject` editor to redraw changes.
- `.local/bag-mapping-inspector-replay-green.log` executed 24 App methods, five assertion failures, zero unexpected, 5.296 seconds. All five binary grouped methods and the ordinary/objective/authority adjacent methods passed. The remaining objective test correctly refused its fixture's three-class importance vector against the actual published two-class head. The fixture alone now supplies two class weights; validation was not relaxed. A terminal replay of the corrected objective fixture remains pending.

## Accepted scoped GREEN

Root's terminal `.local/bag-mapping-inspector-accepted-green.log` executed **24 App methods, zero failures, 4.573 seconds**. All six grouped inspector methods pass together with existing ordinary column-draft, objective-mapping and selected-node authority tests. The grouped objective case now edits real cross-entropy target roles, derives ordered input columns from paired packet rows, preserves weights/lengths on Apply and restores the exact authored node/edge state through undo. The earlier class-count fixture failure and initialization compile prerequisites remain recorded above; they are not additional production regressions or semantic RED claims.

Packaged/native inspector verification and final bundle accounting remain root-owned. Source and tests are frozen; this final receipt update changes no runtime bytes.
