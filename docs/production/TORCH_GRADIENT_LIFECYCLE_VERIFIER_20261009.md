# Gradient policy native lifecycle verification · 2026-10-09

Scope: strict captured gradient/trainability identity, independently derived native parameter ownership and trainable optimizer subset, update cadence, scheduler and parent lifecycle compatibility. Native code verifies JSON identity and immutable checkpoint pins; bounded Python worker remains responsible for binary optimizer state/counters and numerical continuation. No coverage or packaged/native acceptance follows from these checks.

## TDD state

The initial compiling rejecting baseline and five `GradientLifecycleVerifierTests` methods are ready for root's serialized RED run. No Swift command has been executed by this agent. Results remain pending.

Tests regenerate actual sealed prepared CPU MSE training through the current host worker, with explicit SGD and OneCycle: a two-epoch parent, four-epoch exact continuation, a fresh fine-tune omitting the parent's gradient/trainability/scheduler selections, and an absent-policy baseline. The declared development root supplies `.venv/bin/python`; no cached binary fixture is accepted. Independent native geometry expectations are weight[2,2]/bias[2], weight frozen, bias trainable; twelve training observations and batch size five imply three microbatches and two optimizer updates with accumulation two. Mutation controls cover foreign/missing recipes, native shapes/dtypes/flags/order, optimizer subset names, update cadence and independent derivative permission.

Existing lifecycle/parent production methods remain unchanged pending actual semantic RED. Conditional selected paths must preserve every omitted-policy legacy contract and avoid consulting mutable current workspace state. Parent exact-resume presence/recipe/ownership/cadence parity and fine-tune parent reset semantics will be tested after the additive evidence scaffold is approved.

## Measured RED and implemented candidate

Root's first run compiled but fixture graph decoding refused missing required `name`; no semantic verifier result is attributed to that setup failure. The generator was narrowly corrected to add Core presentation fields before actual engine capture. Root's corrected `.local/gradient-consumer-row-lifecycle-red-editors-green.log` reached all five methods: three valid paths threw the deliberate rejecting baseline; two negative methods passed. This is the semantic RED.

The candidate now validates captured graph/registry native ownership, intrinsic embedding flags and selected overrides, ordered optimizer subset, independent actual-row update cadence, objective reduction compatibility, derivative permission and exact selected checkpoint/settings/provenance/runtime identities. Existing optimizer and scheduler verification accept the verified identity; OneCycle counts optimizer updates rather than microbatches. Parent metadata and lifecycle state lists bind selected/stored policy distinctions and exact-resume identity; fine-tune resets parent-owned state even when the child omits the policy. Tests now assert actual fresh/resume/fine-tune lifecycle, optimizer and scheduler positive paths, changed policy/presence and parent tampering. Serialized GREEN is pending.

Parent metadata does not consult today's registry or unpickle state. When a gradient-only implicit-Adam parent has no serialized parameter inventory, native parent evidence cannot invent that inventory; current-side captured graph ownership and the worker's strict checkpoint graph/native-state restoration retain responsibility. Explicit optimizer inventories and trainability inventories are compared when available. The native verifier does not claim to inspect binary update counters.

Candidate source pins (before serialized GREEN):

- `Sources/KwipMLApp/WorkflowGradientPolicyVerifier.swift`: `57c7fb888abc96c5562ff928a6165c82ff639431e643470c5a6fd9f27e4df811`

- `Sources/KwipMLApp/WorkflowLifecycleVerifier.swift`: `38dee7d6f1eef2a550bb4753a3db88655bf76192f5190c047cc14559833f0c2a`

- `Sources/KwipMLApp/WorkflowLifecycleParentRecord.swift`: `f64e7afd97fece885be5e7ae1a513f14ac13569f5d64e40c35982bdcdbbddf24`

- `Tests/KwipMLAppTests/GradientLifecycleVerifierTests.swift`: `446ccdb8e5ee66f4b6a6643e1d3a7f8389780fb084110c8da7fc271c7c689af9`

## First integrated candidate finding

Root's `.local/gradient-artifact-geometry-export-red-lifecycle-rows-green.log` executed the five lifecycle methods: one assertion failed, zero unexpected errors. Independent ownership, metadata, optimizer subset, OneCycle accumulated cadence, parent decoding, exact resume and fine-tune checks otherwise passed. The failing fresh-list assertion exposed exact source order: the real engine appends scheduler, then gradient policy/trainability, then objective state. The native fresh list was narrowly corrected; new child-only fine-tune additions follow that same order without changing stored-parent ordering or legacy absence.

A sixth actual-worker test now generates a gradient-only implicit-Adam parent and exact continuation, checks both real current-side trainable native owners, admits genuine CPU restoration and rejects a changed accumulation policy. Final six-method serialized GREEN remains pending.

Final candidate pins for the next serialized check:

- `Sources/KwipMLApp/WorkflowGradientPolicyVerifier.swift`: `57c7fb888abc96c5562ff928a6165c82ff639431e643470c5a6fd9f27e4df811`

- `Sources/KwipMLApp/WorkflowLifecycleVerifier.swift`: `6c2e46cda4a46a6bdde5e4b888d6fbe2420b47588b6b7f7565de93d5b5b90a96`

- `Sources/KwipMLApp/WorkflowLifecycleParentRecord.swift`: `f64e7afd97fece885be5e7ae1a513f14ac13569f5d64e40c35982bdcdbbddf24`

- `Tests/KwipMLAppTests/GradientLifecycleVerifierTests.swift`: `b4ea12ba5cbb15d745bb3e575f7e23ed256cea32430c45ac450638f1aa176092`

## Root serialized GREEN

`.local/gradient-consumer-history-export-repair-green.log` completed with32App methods, zero failures/skips, terminal0. The union includes six lifecycle methods, four geometry methods, four export contract methods, four actual prepared-table methods, six derivative decoder methods, one actual CPU training/cold History/export/portable inference/tamper method and seven adjacent lifecycle/initialization methods. The first joint candidate recorded two actual History failures caused by the missing bounded artifact-path admission; the narrow three-path repair is included. This is automated/real-worker evidence, not native acceptance or new catalog credit.
