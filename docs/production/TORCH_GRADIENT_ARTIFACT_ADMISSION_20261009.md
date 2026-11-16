# Omitted gradient metadata admission · 2026-10-09

Finite test-only slice: `GradientArtifactAdmissionTests` exercises the extracted unchanged production `WorkflowPhaseArtifacts.validateUnselectedGradientMetadata` guard. Historical omitted-policy contexts remain allowed. Three reserved fields (`settings.microbatchesPerEpoch`, `settings.optimizerUpdatesPerEpoch`, `runtime.gradientSummaryContext`) must refuse without a selected gradient policy; foreign recipe/provenance/checkpoint identities, cadence, derivative artifacts and explicit permission must also refuse. Existing older gradient-summary capture remains distinct.

Three methods are ready for root's serialized RED. No Swift commands or production edits were performed by this agent. Root owns the three-check production repair and subsequent GREEN. No new coverage/native acceptance is claimed.

Root actual verification: `.local/gradient-results-green-admission-publication-red.log` executes three admission methods with three failed no-throw assertions against the absent reserved-field checks. Root adds only the three missing checks. Final `.local/gradient-candidate-results-bridge-green.log` executes all three admission methods successfully inside5Core +47App methods, zero failures/skips. Production and test source remain separate from native acceptance.
