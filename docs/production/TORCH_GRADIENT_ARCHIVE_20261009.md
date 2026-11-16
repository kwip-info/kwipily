# Gradient/trainability immutable archive projection — 2026-10-09

Scoped archive projection only; no capability acceptance/native UI/permission or capture-format expansion.

## Actual RED

The root's first joint attempt was compile-only: the new test used plain String literals where JSONValue required explicit .string; these test fixture types were corrected without changing production. Actual semantic RED `.local/gradient-settings-archive-learning-app-semantic-red.log`: GradientArchiveTests five methods, ten failures including two unexpected unsupported-compiler throws. Lower compilers silently omitted selected policies and schema10/compiler9 could not retain them. The combined13-method run also contained other owners' expected failures; its199 assertions are not attributed to this archive slice.

## Implemented boundary

WorkflowArchivePlan now retains optional typed qualified gradientPolicyRecipe and trainabilityRecipe wrappers. Each exact {source,configuration} envelope requires a bounded qualified control address and complete canonical four-key configuration recipe; trainability owner order is canonical, derivative request order is preserved. Source addresses remain distinct from Train, optimizer, scheduler, objective, initialization and one another. New policies require schema10/compiler9 and cannot silently project into old compilers. Selected initialization can coexist in that new pair; old schema9/compiler8 remains accepted and all future pairs refuse.

The new GradientArchiveRecipes helper validates structural membership in captured expanded model nodes and origin inventory, unique identities and genuine input target node kinds, nonempty bounded endpoints, native owner-node selection, and objective-reduction compatibility. It deliberately does not consult today's built-in registry or invent parameter membership/aliases/differentiability. Exact configured port/dtype geometry remains the compiler/History recompile boundary using the captured registry; native parameter membership remains the materialized worker boundary. Unknown port strings beyond those structural checks are not claimed verified here.

Absent new fields remain nil and omitted; existing historical compiler/schema pairs and canonical byte roundtrip controls are retained. No execution permission, captureFormat, source file policy or checkpoint transport expansion occurs in this slice.

## Tests and source freeze

Five methods cover selected qualified recipe retention and changed-policy identity, initialization coexistence, old-compiler refusal, absent legacy canonical byte roundtrips, wrong/colliding sources and wrapper/control/foreign owner/blank endpoint/wrong input kind/version mutations. Root serialized actual GREEN in `.local/gradient-settings-archive-learning-app-green.log`: five archive methods passed with zero failures/skips; the joint13App methods passed. This agent ran no Swift/build/native command. Source was frozen at that boundary.

- `Sources/KwipMLApp/WorkflowArchiveContracts.swift` SHA256 `76038aa66401dfe1373c0bac2c038bf3e398ec5c07effd265a7d8b67f496cddc`
- `Sources/KwipMLApp/GradientArchiveRecipes.swift` SHA256 `ed54a4025df169424a26fe9ef9e20945a45afae39aec5272d585318308979a8b`
- `Tests/KwipMLAppTests/GradientArchiveTests.swift` SHA256 `e0dc3817ef0d233f106f2407dc1b069083c3f48c180c81a687b1cd4e4b3acd0c`

- `gradient-settings-archive-learning-app-semantic-red.log` SHA256 `c1a6fcfd94043220256ea31367d057af07a110fb6b69dca91055dbfc3502e4b6`

- `gradient-settings-archive-learning-app-green.log` SHA256 `161dd57419174f4d6ae1ff44fe5c0e79065b94292ce9dde3c85dcf6cc7352396`
