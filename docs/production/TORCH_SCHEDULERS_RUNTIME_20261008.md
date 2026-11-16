# Five schedulers: finite runtime proof — 2026-10-08

## Result and scope

**Runtime slice passed:184 new scheduler checks plus198 adjacent optimizer/checkpoint/portable-inference checks,382 total, zero failures/skips in10.03s.** Actual host Torch2.14.1, CPU and MPS. Final terminal log `.local/scheduler-final-runtime-and-prior-green.log`. This is scoped runtime proof, not native App/package, transport/watchdog, learning, broader prior-bound regression or catalog acceptance.

The frozen family comprises StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR and LinearLR. New runtime helper/tests and narrowly scheduler-aware engine, optimizer validation and checkpoint lifecycle are implemented. Root owns Swift/UI/transport/native/artwork/catalog; authoring supplied actual registry63 and generated scheduler tables. Actual authored/generated/discovered capability equality passed for all five; no fixture registry bypass or model-forward scheduler operation was added.

## Frozen APIs and data shape

`runner/scheduler_configuration.py` exports `canonical_configuration(component_id,supplied)`, `validate_recipe(value)`, `constructor_options(recipe)`, `capability(component_id)`, `validate_compatibility(recipe,optimizer_recipe)`, `preflight(recipe,optimizer_recipe,epochs,batches,parameter_count)`, `create_scheduler(optimizer,recipe,optimizer_recipe)` and `SchedulerController(optimizer,recipe,optimizer_recipe,parameter_count)`.

The portable recipe has exactly schemaVersion1/componentID/configurationVersion1/config. Defaults: Step interval2/gamma.1; Cosine T_max10/minimum0; Linear1/3→1 across5 ticks; Plateau all reviewed native defaults with validation_loss/min explicitly selected; OneCycle peak.1/planned10 batches/pct.3/cosine/cyclingtrue/minMomentum.85/maxMomentum.95/divisor25/finalDivisor10000/two phases. Bounds, choice/type checks, actual phase spans, derived rates, compatibility and resource limits are independently admitted before execution.

Engine `run(...,optimizer_recipe=None,scheduler_recipe=None)` accepts only an explicit reviewed optimizer with a scheduler. No-scheduler optimizer recipe/default behavior and generated training-recipe bytes remain unchanged. Discovery advertises exact recipe-only descriptors. Generated tables expose literal CAPABILITIES and partial NORMALIZERS, without forward resolvers.

Optional checkpoint contract adds `scheduler: {schemaVersion:1,recipe}`. New checkpoint payload adds:
```text
scheduler: {recipe,state,ticks,timing,rateTrace}
```
`state` is exact upstream native state whitelisted and recomputed before load. `ticks` counts completed ticks, not constructor initialization.

One optional `rate-trace.json` artifact has exact header:
```text
{schemaVersion:1,recipe,timing,initialization:{tick:0,rate,momentum?},records:[...]}
```
Initialization contains actual constructor settings separately from completed tick records. Record keys: tick/epoch/usedRate/nextRate/nextRateUnused. OneCycle adds local batch and sorted unique updatedParameterIndices; momentum cycling adds usedMomentum/nextMomentum. Plateau adds metric/metricValue/changed. Timing is afterOptimizerBatch or afterCompletedEpochValidation. Trace is bounded to10000 records and4000000 UTF-8 bytes, with count/field/ownership-based conservative preflight before record allocation; no silent truncation. Actual persisted pretty-printed JSON is also bounded.

Manifest adds settings.schedulerRecipe, provenance.scheduler `{recipe,implementation,torchVersion}` and rateTrace `{schemaVersion,path,bytes,sha256,records,timing}`. Real-time schedulerTick events contain schemaVersion/runID/scheduler recipe/record. Epoch events gain optional scheduler recipe/ticks/usedRate/nextRate/nextRateUnused metadata. All optional scheduler fields/artifacts are omitted when unconfigured. Effective schedule rates live in the initialization and tick trace; settings.learningRate retains the linked optimizer's base rate.

Root explicitly reviewed **binary-only Plateau native best/mode_worse state**. Exact +/-infinity sentinels are admitted only through whitelisted native state replay; mode_worse remains its native sentinel. No JSON state projection was added. JSON recipe/initialization/records contain no nonfinite values, so no native sentinel is serialized to JSON.

## Independent numeric, timing and state checks

All25 scheduler/optimizer pairings ran on CPU and MPS (50 tests), using independently calculated weighted learned updates under the actual scheduled rate/momentum/beta1. Test expectations use separate scalar rate/update functions, never configured native scheduler outputs as gold. Both OneCycle curves and two/three phases were exercised through the full horizon, including actual negative unused terminal next rate for linear annealing. Adagrad cycling-off works; incompatible cycling, beta1 bounds and horizon/resource overruns refuse.

Plateau tests cover min/max, both metric choices, relative/absolute improvement, patience0, cooldown, floor and ignored epsilon reductions on CPU/MPS. Independent retained best/bad/cooldown state and native rate are compared, and missing/bool/nonfinite metrics refuse. Epoch schedules include Step gamma0/gamma1/nondefault intervals, Cosine endpoint/nonzero minimum, Linear decreasing/equal factors and constructor initialization effects.

Five actual CPU graph runs resumed from one epoch to three and matched uninterrupted model parameters, actual optimizer state, scheduler state/trace, parameter update counts, RNG state and metrics **exactly**. Five actual MPS graph runs refused exact continuation and completed fresh weights-only fine-tuning with scheduler reset/initial epoch0. Rehashed foreign native scheduler state, altered trace rates/initialization/typed ticks, foreign groups, missing ownership, changed counters and corrupted rate artifact refuse. Checkpoint loading verifies the rate artifact bytes/hash, payload equality and completed epoch/batch geometry before output allocation; native load_state_dict is never the admission check.

OneCycle records exact per-tick gradient presence as updatedParameterIndices. This proves SGD momentum-buffer creation/persistence when current scheduled momentum is zero and parameters receive gradients on different ticks. A parameter updated only at zero momentum legitimately has no buffer; another updated earlier at positive momentum retains its buffer. Strict optimizer context permits only declared scheduled rate/momentum/beta1 and exact scheduler group metadata. Fixed optimizer options and parameter IDs/state remain strict; no broad bypass or empty-or-buffer allowance was added. Epoch schedulers preserve the existing fixed-momentum per-parameter state rule.

Standalone fresh export includes SCHEDULER_RECIPE/create_scheduler/training_step/completed_epoch_step. All five standalone sequences reproduced independently calculated rates and learned updates; wrong OneCycle epoch stepping and missing Plateau metric refuse. Default emitted code measured3905–4300 UTF-8 bytes, below the existing8192-byte training_recipe.py bound. Inference remains separate and never executes the recipe. This proves fresh training sequences; exact restoration remains the versioned production checkpoint path.

## Meaningful RED/GREEN and corrections

- Helper absent:39 RED checks (`scheduler-config-red.log`), then39 GREEN controls. Root froze Step2/OneCycle10 defaults after the initial proposal; one three-phase negative fixture was corrected from a valid geometry to an actually negative final span. Missing-helper RED was not claimed as native geometric evidence.
- Native construction API absent:13 RED/39 deselected (`scheduler-native-red.log`), then52 GREEN including independent CPU/MPS rates/updates (`scheduler-helper-green.log`).
- Engine optional keyword/state/trace absent:5 RED/52 deselected (`scheduler-integration-red.log`), then57 GREEN (`scheduler-integration-green-attempt.log`).
- First continuation replay:4 RED/6 passed/57 deselected (`scheduler-resume-first.log`). Epoch schedulers incorrectly applied OneCycle's gradient-index buffer rule without per-batch lists. Repair restricted that override to OneCycle; fixed epoch momentum retained its prior strict state policy.67 GREEN (`scheduler-resume-green.log`).
- Export optional scheduler API absent:5 RED/67 deselected (`scheduler-export-red.log`), then72 GREEN (`scheduler-export-green.log`).
- Real metadata typed-integer gap:4 RED/180 deselected (`scheduler-manifest-types-red.log`); bool/float schemaVersion/records were admitted through Python equality. Exact integer types/ranges now refuse before output allocation.
- Intermediate metadata test used an incorrect registry storage assumption: five failures/175 passes. Runtime contracts live in authored bundle/generated table/discovery, not ComponentDefinition rows; fixture corrected to actual product sources without changing numeric assertions.
- Final184 new +198 prior **382 GREEN** (`scheduler-final-runtime-and-prior-green.log`). No numerical tolerance or malformed-state expectation was weakened.

## Reproduction and evidence identities

```sh
.venv/bin/python -m pytest -q \
  runner/tests/test_scheduler_runtime.py \
  runner/tests/test_optimizer_runtime.py \
  runner/tests/test_checkpoint_lifecycle.py \
  runner/tests/test_portable_inference.py
```

| Source/evidence | SHA-256 |
|---|---|
| runner/scheduler_configuration.py | `136bb85ff4dd6891a5037aba80b819ab3d870069759bc89c8314d92a85cfea78` |
| runner/tests/test_scheduler_runtime.py | `450d3ff0bd652c619794e541ec4596582f1474d5a79b030e8578cb1d0797cd4a` |
| runner/optimizer_configuration.py | `bb91d8c63a5b44896378a8bd22635481bf49f9377ff5aa19c0492f2adf5aba30` |
| runner/engine.py | `7cbf1b1578e6e94434887ed660f3a6c33f36f99f554500cdef5132b627b98b72` |
| runner/checkpoint_lifecycle.py | `1309f381f8b3b526ac31e732c8fb0dd403624f0593595bbc26cd712f3db365a9` |
| Helper RED log | `efb24852be165ff1a872891cba7cd259818621db93810879a6ba3f464279c9f3` |
| Native API RED log | `f763a124c905bcbd5dff4a6978433df360f096bd7a102af3c75481ef672a7b2e` |
| Engine RED log | `794002184c04e8893f20e32139868439eb8b141eda288979eb02957b6889db66` |
| Continuation RED log | `fd07fcd0c9d98362aff362b9db4e37dc085566bd27fbcfee9e6e58b2c59115f9` |
| Export RED log | `9993cca9f79fe7e85b97409c2f0cb830c65c094aa17480776338c581dceafe06` |
| Metadata-type RED log | `b137b4a63ce72e5e2f57d328b5b7d357c0084779592331869f06ff0a897ce8f5` |
| Final382 GREEN terminal | `e7d65d36a39c15b0c1f597c11de9e9feb909ad519945c0fde940030357aff73b` |

Prior frozen receipts/catalog semantics were not edited. Remaining root-owned integration gates include actual canonical five-workflow host/CLI supervision evidence, App source-qualified capture and cold History, native OneCycle construction/run/trace/export, learning receipts, bound-prior replay, packaging and acceptance. Runtime sources are ready to freeze for those integrations; any concrete integration failure should be handled as a new finite repair.

## Native artifact verifier integration (root-serialized Swift evidence)

A separately leased App slice extends `WorkflowPhaseArtifacts.trainingArtifacts` with default-absent scheduler recipe/configuration arguments, forwards the universal capture plan, and reads `rate-trace.json` through the existing descriptor-relative bounded reader. The immutable inventory includes its exact size/hash. No scheduler selection rejects foreign optional scheduler metadata and any rate-trace file. Selected runs require exact canonical settings/provenance/checkpoint identity and independently recomputed constructor/record rates, saved Plateau metric observations, epoch/batch geometry, and declared OneCycle update ownership. A shared `validateSchedulerRecord` performs bounded structural checks for both live events and immutable traces; full trace validation additionally proves numerical curves. Native code does not interpret Torch checkpoint pickle/state: actual typed state/counters remain independently checked by the worker.

The meaningful root-run RED `.local/scheduler-artifact-red-app-second.log` contained one method with five assertion failures, zero unexpected failures: all five foreign scheduler artifacts were previously admitted. SHA-256: `f5d19ebf333d2cc13fd08cd493c65b6867959656e76e8530b11ef849bfaa8158`.

Root's subsequent `.local/scheduler-artifact-green-history-export-red.log` passed all **10** `SchedulerWorkflowArtifactTests` methods, zero failures/unexpected failures/skips, 0.405 seconds. SHA-256: `e8872645a2867514b0a81ddf22e928554c0c3a7c9d349823eca6d81a04cc8c55`. The combined 17-method command remained RED on two separately root-owned History/export methods; this is scoped artifact proof, not a whole-App GREEN claim. Fixtures use explicitly synthetic immutable checkpoint bytes to test native metadata/inventory admission and never claim actual Torch state evidence.

The additive nonpersisted `WorkflowLifecycleParentEvidence.hasScheduler` defaults false. Selected fresh/exact policies include scheduler/rateTrace, while fine-tuning resets them if either the trusted parent or the selected target has a scheduler. Root owns population from the hash-verified parent. A newly written exact-resume scheduler-presence parity test is awaiting root's serialized RED; the additional guard is not yet implemented/tested in this paragraph's evidence.

Two further bounded native-verifier gaps were isolated before repair. Root's `.local/scheduler-parent-endpoint-red.log` captured one method/one assertion failure/zero unexpected failures: otherwise valid exact-resume metadata could claim scheduler restoration from a trusted unscheduled parent. SHA-256 `950cc1078517ad7285520b8b25674abbc1a5e659714e21900d40530f3a427a73`. Root's `.local/scheduler-momentum-endpoint-red.log` captured one method/one assertion failure/zero unexpected failures: native linear OneCycle's representable endpoint `0.29999999999999993` was incorrectly rejected by strict declared `0.3...0.9` range checks. SHA-256 `2eb29754758ca18d78ff6dd33c04c6d4b0f94b810c3236c0c2ebaf78432bfd19`.

The CPU host probe used installed Torch's actual SGD/OneCycle, total8/pct.5/linear/base.3/max.9: ticks1–3 produced momentum `0.7`, `0.5`, `0.29999999999999993` and rates `0.036000000000000004`, `0.068`, `0.1`. The new endpoint test retains a materially wrong `0.299` refusal. After these semantic REDs, exact resume now requires parent/target scheduler-presence parity. Structural momentum bounds retain strict physical0–1 checks and allow only `max(1e-15, abs(maxMomentum)*1e-12)` endpoint rounding around declared bounds; no clamping occurs, and full immutable curve equality is unchanged. Root's final twelve-method artifact replay is pending at this entry.

Final root-serialized artifact result: `.local/scheduler-context-experiment-red-artifact-green.log` passed **all12** `SchedulerWorkflowArtifactTests`, zero failures/unexpected failures/skips, 0.407seconds (0.408wall). This includes both exact-parent presence parity and native linear representable endpoint repairs. Log SHA-256 `1359b3fbc7f960d0b9eb48cdf1652132782b42609d372d717a7bead578ddf4a7`. A separately root-owned Core experiment-context RED in the combined invocation prevents claiming a whole-command GREEN; the App artifact suite itself was terminal GREEN.

Artifact slice source hashes at this finite closure: `WorkflowLifecycleVerifier.swift` `7f1c2f66a7a4dd4b00b291d291dac9a859c6a879618fd464a07a8ccf15bb06a2`; `WorkflowPhaseArtifacts.swift` `f4e3c1a69e7329061977b68c51e96749eb02be878a328b56f00d69e87537d3b5`; `SchedulerWorkflowArtifactTests.swift` `b7e92b85d9a91dd75f563b167a2b9d5901db3793b767a611de21bc065c6118f1`. No native application/artwork/distribution or catalog acceptance is claimed by this slice.
