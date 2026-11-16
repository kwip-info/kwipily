# Checkpoint save and restore: finite closure

Implementation contract frozen by root, 2026-10-09, under the owner-authorized all88 scope. This document grants no implementation acceptance or coverage. Scope is the single adopted `pytorch.torch.save` unit, consolidating `torch.load`, with all five original modes: weights-save, continuation-save, weights-restore, continuation-restore, incompatible-artifact-refusal.

## Position in the remaining inventory

After the accepted eight-loss closure, the ten reviewed units still requiring closure are Module composition; EmbeddingBag; LBFGS; Dataset; DataLoader; checkpoint save/restore; exported executable program; requested/reset/accumulated gradients; no-grad/trainability; and initialization (Xavier uniform, Kaiming uniform, normal, zeros). Current accepted count is78/88; this single lifecycle unit remains planned. Save/restore is the smallest next slice because real persistence, History selection, portable export and lifecycle execution already exist. It must not claim `torch.export.export` or close the independent data/gradient/initialization units.

## Consumer boundary: reuse existing controls

Recommend an **artifact-backed lifecycle bundle**, not a new tensor-forward node or second checkpoint engine. Existing Train configuration chooses Fresh, Exact resume or Fine-tune weights; its Parent checkpoint selector chooses a verified completed History entry. History already creates a checkpoint and offers a portable trained package. Those remain the canonical authoring surfaces. Qualified source/address, parent run/request, current document revision and compatibility stay captured by existing workflow compilation and private bindings.

The bundle must explicitly describe these existing configurable actions rather than advertise an inert palette icon:

| Action | Consumer controls and result |
|---|---|
| Save learned weights | Completed History entry → portable trained package → owner-selected destination and folder name. Saved checkpoint may contain additional state; weight-only restoration deliberately selects modelState and never restores optimizer/RNG. Do not mislabel the full checkpoint bytes as a minimal weights-only file. |
| Save continuation | Successful completed run writes its private pinned checkpoint/manifest/metrics. This is an epoch-boundary snapshot, not periodic recovery or mid-batch save. |
| Restore weights | Train → Fine-tune weights → verified parent. New run explicitly resets applicable optimizer, RNG, epoch, metrics and scheduler/objective training state; exact learned weights/buffers are validated and transferred. Frozen-model inference is another existing weight-only consumer, with its own trainability policy. |
| Restore continuation | Train → Exact resume → CPU parent and CPU target, target total epochs greater than parent. Matching graph/data/runtime/optimizer/scheduler/objective/stochastic policy are mandatory. No implicit device change. |
| Refuse incompatible artifact | Existing verification must explain corruption, wrong identity, incompatible keys/shapes/dtypes, changed data/config/runtime, unsafe files and unsupported continuation before training state changes. |

A future separate minimal weight-only file format or arbitrary external checkpoint importer is **not required** for this bounded application-owned save/load unit. If the reviewed scope is interpreted to require either, obtain explicit owner approval before adding it; do not credit unsupported arbitrary pickle import. Original mode names still all require independent save/read evidence.

## Existing foundation versus missing bundle work

Existing behavior to rebind, not reimplement:

- `runner/engine.py` saves a real Torch checkpoint; `checkpoint_lifecycle.py` reads bounded regular bytes and invokes `torch.load(..., map_location='cpu', weights_only=True)`, validates stored contract, model state, optimizer/counters, RNG and objective/scheduler state. CPU epoch-boundary exactness and MPS weights-only are explicit.
- `WorkflowCheckpointSelectionView.swift`, `WorkflowPrivateCheckpointBindings`, `WorkflowCheckpointSnapshot.swift` and `WorkflowLifecycleParentRecord.swift` provide private parent selection, hash pins and snapshot capture. Snapshot bounds are manifest20MB/checkpoint100MB/metrics4MB; parent is immutable. Scheduler dependencies/rate-trace remain required where the current lifecycle contract demands them; verify actual snapshot transport rather than assume the three-file helper alone carries every family.
- `WorkflowHistoryReader.swift`, `WorkflowTrainedExport.swift`, `TrainedModelInputContract.swift` and the portable inference resource already stage/reverify/publish model source, checkpoint, exact input/output contract and optional training helpers. Objective numeric outputs and classification outputs remain distinct. Saving never copies private rows or source bindings.
- Prior numerical tests cover uninterrupted versus resumed CPU runs, fresh fine-tuning, corrupt/symlink/incompatible/dtype refusal, preserved parent bytes, frozen restoration, portable inference, objective/optimizer/scheduler continuation and stateful buffers. These are reusable fixtures, not automatic new acceptance evidence.

Missing: an explicit versioned lifecycle-bundle declaration and capability admission contract; discoverable shared documentation/configuration grouping; generated themed save/restore artwork; approachable lessons with independently measured restoration/continuation outcomes; one coherent current-source mode matrix and actual native workflow binding. Do not fabricate a forward adapter, duplicate the existing selectors, or invent a new checkpoint schema merely for coverage accounting.

Proposed metadata describes supported operations, application-owned provenance only, checkpoint schema versions, file/tensor/expanded-allocation limits, trusted source policy, CPU exact boundary and MPS weights-only policy. Exact descriptor keys/registry location require root approval after reviewing the current bundle generator's non-forward integration seam. Preserve all old registry definitions and checkpoint bytes when the bundle is omitted.

## Lessons and original artwork

Two small lessons can cover the five modes without pretending serialization changes predictions:

1. **Keep and reuse learned settings:** train a tiny affine model, save its actual result, reopen the saved package, restore weights and compare predictions on different held-out rows; then fine-tune on a changed tiny dataset and verify optimizer memory restarted. Explain weights as learned settings and checkpoints as saved training state. Grade actual saved identities, restored settings and predictions, not a learner-entered success flag.
2. **Continue from where training stopped:** compare one uninterrupted CPU run with a save/reopen/resume construction. Independently check learned settings, optimizer memory, RNG, complete epoch history and parent immutability. A changed batch size or corrupt fixture must visibly refuse; MPS offers explicit weights-only continuation. Guided, practice, private varied fixtures and cold reopening use the same strict worker boundary.

Use new image-generated paired save/restore checkpoint artwork in the existing Training family palette, with an accessible label and documentation placement beside the existing Train/History actions. No duplicate static icon-only tool counts as implementation. Lesson controls reuse the existing lifecycle configuration templates; instructions define new terms before mentioning exactness or randomness.

## Bounded implementation ownership

- **Runtime/tests:** narrow `checkpoint_lifecycle.py`/engine changes only if a specific new failing test proves a gap; new lifecycle bundle helper/discovery if the approved descriptor needs it; independent new save/load matrix tests. Existing frozen loader and export code remain authoritative.
- **Core/bundle authoring:** versioned artifact-backed declaration and exact typed configuration/readiness bridge, generator support only at its existing non-forward capability seam, metadata/example/resource registration. No new tensor ports or workflow-schema bump without a demonstrated need.
- **App:** shared existing Train checkpoint inspector/History action grouping and capability admission, parent/capture/export verification only where a measured gap exists. Root serializes Swift and native testing.
- **Learning/art:** two bounded tasks/references/worker assessments, original generated assets and actual bundled examples; independent scalar affine/reference-state expectations. No arbitrary imported model code execution in assessment.
- **Accounting:** root alone binds all five original modes and ten standard coverage dimensions to actual final evidence after closure. Existing receipts remain historical and unchanged.

## Finite exits and required negatives

1. Approve the artifact-backed operation boundary, exact metadata/descriptor and UI locations before coding.
2. Capture meaningful RED for missing bundle admission/lesson behavior and each concrete discovered lifecycle gap; do not manufacture RED by weakening old behavior.
3. Save application-owned full continuation state and restore only weights separately on CPU/MPS; verify all learned parameters, integer/floating buffers, predictions, device policy and parent byte identity. Inspect actual payload member set and bounded loader behavior.
4. Compare CPU uninterrupted versus save/restart/resume using nontrivial SGD or Adam, stochastic model state and one scheduler/objective case. Verify optimizer memory/counters, RNG, metrics and identity independently. Test changed graph, dtype, recipe, mapping/data membership, batch/seed/runtime, nonadvancing epochs and missing required state. MPS exact resume refuses; MPS weight restoration truly runs on MPS.
5. Test wrong hashes, rehashed incompatible payload, unexpected state keys, unsafe/symlink files, malicious/nonallowlisted pickle input, compressed expansion and oversized metadata **before deserialization/allocation**, using tiny injected limits where possible. Preserve current resource ceilings and no unsafe `weights_only=False` fallback.
6. Exercise atomic destination publication, overwrite refusal, cancellation/failure cleanup, privacy exclusions and source-import-independent saved inference. The portable package includes user-authored architecture code; loading it executes that owner-reviewed model source, so do not describe the entire package as an untrusted-code sandbox.
7. Actual four lesson modes and two examples pass; native root confirms History save → close/reopen → select parent → restore/fine-tune/resume, visible actionable incompatibility and portable package save. Final source-bound prior regression is finite and deduplicated.
8. Root freezes one acceptance receipt with all original modes, current supported backends/limits and original themed artwork, then refreshes bindings. App Store distribution/UAT and clean-machine proof remain separate gates.

Stop when these exits pass. Arbitrary checkpoint import, periodic autosave, distributed/sharded checkpoints, cross-version exact continuation, GPU exactness, mid-batch cursors and executable-program export are explicitly separate work, not this closure.

## Frozen implementation boundary · 2026-10-09

Use the existing bundle generator with one `kind: artifactLifecycle`, ID `kwip.lifecycle`, one `checkpoint_lifecycle` entry and fixed action bindings to existing Train/History surfaces. Generate separate lifecycle metadata/discovery data; preserve all83 ordered component definitions and existing generated component output bytes. No canvas node, tensor ports, workflow schema change, second persistence engine or arbitrary executable metadata. The authored sources/presentation/learning references are validated before publication.

Worker discovery adds only `lifecycleCapabilities.checkpoint_lifecycle`. The exact descriptor is:

```json
{"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.checkpoint-lifecycle","checkpointSchemaVersions":[1],"modes":["weights-save","continuation-save","weights-restore","continuation-restore","incompatible-artifact-refusal"],"sourceTrust":"ownedCompletedHistory","loadPolicy":"weightsOnlyCPUMap","artifactLimits":{"manifest.json":20000000,"checkpoint.pt":100000000,"metrics.json":4000000,"rate-trace.json":4000000},"requiredArtifacts":["manifest.json","checkpoint.pt","metrics.json"],"schedulerTracePolicy":"requiredExactlyWhenParentHasScheduler","saveTiming":"successfulRunCompletionAtEpochBoundary","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"cpuResume":"epochBoundaryExactCPU","mpsResume":"weightsOnly","fineTune":"restoreModelStateResetApplicableTrainingState","parentImmutable":true,"externalImport":false,"periodicSave":false}
```

The strict Core decoder requires this exact reviewed literal. Missing/contradictory lifecycle discovery revokes restore readiness before parent staging; fresh training and historical inspection retain existing behavior. Typed generated metadata supplies documentation/art alongside existing configuration and export actions, with long text in Information. Do not add another chooser.

The conditional scheduler trace transport has already passed actual App exact-resume/fine-tune verification in the loss batch; retain it and bind its receipt rather than repeating implementation. The remaining concrete loader review is expanded ZIP storage validation before deserialization. Reuse the existing frozen-checkpoint preflight where semantically compatible; independently test refusal before `torch.load`, preserve current resource ceilings, and do not advertise an expanded-allocation guarantee until proven.

Two assessment identities are reserved: `checkpoint-restore-weights` and `checkpoint-continue-training`; matching lesson IDs are `assessment.` plus each identity. Their examples must perform actual save/read/restoration or continuation, with independent state/prediction expectations and guided/practice/private/cold receipts. Learning agent proposes the smallest existing lesson integration seam before touching shared dispatch.

File ownership: runtime agent owns checkpoint_lifecycle.py, additive engine discovery sibling and new runtime tests; authoring agent owns new manifest/generated files, additive generator validator/emitter and Core typed lifecycle metadata/decoder/tests; learning agent initially owns a new scoped lesson contract only; root owns App integration, shared learning dispatch leases, serialized Swift/build/native and final accounting. No agent edits accepted catalog/receipts.
