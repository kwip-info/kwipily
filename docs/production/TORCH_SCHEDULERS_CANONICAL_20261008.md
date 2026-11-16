# Canonical scheduler host/export proof — 2026-10-08

All five genuine Core-emitted prepared scheduler examples passed on actual CPU and MPS: **10 canonical cases, no failures or skips**. Both host commands terminated with exit 0. This is host-only proof, not native UI, signed packaging, final accounting or release acceptance.

## Inputs and independence

Root emitted actual schema-7/compiler-6 factory fixtures at `.local/scheduler-canonical-examples`, with six JSON files per scheduler: workflow, model, dataflow, optimizer-recipe, scheduler-recipe and settings. Each file is individually hashed in the detailed receipt. Source-qualified control nodes and their unique scheduler edge to Train were independently checked against the emitted canonical recipe. No handwritten workflow was substituted.

The existing sealed BN1 native archive supplies the identical preparation graph: 48 training rows and 16 validation rows, two float32 features. Its manifest pin and dataflow digest were checked before use. Torch 2.14.1; seed42, batch32, three epochs; actual explicit CPU and MPS exercise both devices even though these factory fixtures request CPU.

Independent scalar arithmetic implements the exported parallel model forward pass, five schedule curves, Plateau improvement/cooldown/floor logic and momentum-weighted learned updates. Native scheduler outputs are measured, never used as expected curves. CPU resumes compare model/optimizer/scheduler/trace/update-count/RNG state and metrics exactly; all five MPS exact resumes refuse and actual weights-only fine-tunes declare fresh scheduler/trace reset.

## Measured matrix

| Scheduler | Training device | Saved trace records | Export helper bytes | Scalar inference maximum error | Fresh helper update maximum error |
|---|---|---:|---:|---:|---:|
| scheduler_step_lr | cpu | 3 | 3805 | 7.24e-08 | 5.23e-08 |
| scheduler_step_lr | mps | 3 | 3805 | 5.89e-08 | 5.23e-08 |
| scheduler_cosine_annealing_lr | cpu | 3 | 3826 | 4e-08 | 6.62e-08 |
| scheduler_cosine_annealing_lr | mps | 3 | 3826 | 5.45e-08 | 6.62e-08 |
| scheduler_reduce_on_plateau | cpu | 3 | 4067 | 6.63e-08 | 1.02e-07 |
| scheduler_reduce_on_plateau | mps | 3 | 4067 | 5.48e-08 | 1.02e-07 |
| scheduler_one_cycle_lr | cpu | 6 | 4200 | 1.34e-07 | 1.33e-07 |
| scheduler_one_cycle_lr | mps | 6 | 4200 | 1.54e-07 | 1.33e-07 |
| scheduler_linear_lr | cpu | 3 | 3895 | 8.85e-08 | 5.02e-08 |
| scheduler_linear_lr | mps | 3 | 3895 | 1.03e-07 | 5.02e-08 |

Every saved model was inferred on both CPU and MPS. Each exported `training_recipe.py` ran independently on both devices: five steps for Step/Cosine/Linear, 12 Plateau steps (exercising the default patience-triggered reduction), and the full ten-batch OneCycle horizon including its terminal unused next rate. All helpers fit the existing8192-byte bound.

All ten saved traces matched checkpoint trace content, canonical settings, manifest hashes/byte counts/record counts and independent scalar expectations. Maximum persisted rate error: 3.469446951953614e-18. OneCycle canonical training completed six batch ticks; the separate fresh helper sequence exercised all ten planned ticks. Constructor initialization was verified separately from actual ticks.

Portable inference ran with batch size1 and the full16-row held-out set for every case. The optional trace is hash-checked on each package reload: deliberately changed bytes refused in all ten cases. The host-built inference inventory uses the actual exported model/checkpoint/recipe/trace and current application inference resource; it is not a claim that the native exporter created these packages. Parent separately owns actual History/export UI verification.

## Reproduction

```sh
.venv/bin/python .local/verify-scheduler-canonical-workflows.py \
  --fixtures .local/scheduler-canonical-examples \
  --archive '/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/056FF2E4-2C6A-4E0E-81B5-4EC1A1B6A7B0/F3F3695C-71E6-453A-ADB4-58A09EB6F928' \
  --output .local/scheduler-canonical-host-20261008
.venv/bin/python .local/verify-scheduler-saved-traces.py
```

Both proof scripts refuse replacing existing proof outputs. Receipt stores per-case exact model/trace/checkpoint/factory inputs and current source hashes.

## Evidence identities

| Evidence | SHA-256 |
|---|---|
| `.local/verify-scheduler-canonical-workflows.py` | `8cd27b761af6d6aecf9dba34704241f47c536d89554a8328a2de5c79735d1964` |
| `.local/verify-scheduler-saved-traces.py` | `82fcd8a4b42820e84ba9ebf6b260ffa32eb807e65d746ef719f2adb73bdbb365` |
| `.local/scheduler-canonical-host-20261008.log` | `48e7d48c1d6e9f9045ce256ffaf9121a696c89545b00d9eb9af1da6f47b57a44` |
| `.local/scheduler-saved-trace-scalar.log` | `e115e19a39e8090933920ef5f7dceb33a079b9a94e18c44d29048ae1cd4e3367` |
| `.local/scheduler-canonical-host-20261008/receipt.json` | `38595a0707be5b66431c2831063e2e7d175a4502071d43b88c733528c9da8174` |
| `.local/scheduler-canonical-host-20261008/saved-trace-receipt.json` | `2cdd13791dd38fb05815cd6e19a647e96ed5315c2293d2bce65bb95541a0d9a8` |

| `.local/scheduler-canonical-host-20261008/artifact-hashes.json` | `330682af0d161d99a25bee16e6d53ecc84e7b03240b7d7beb434de086e551b86` |

## Finite gaps

None in this host matrix. Native representative run/live trace, verified History, actual staged portable publication, signed candidate testing and frozen acceptance remain root gates. Other scheduler variations/optimizer pairings are covered by the separately recorded runtime suite; this matrix makes no claim to repeat every variation. No Swift, packaging, native UI or catalog mutation was performed by this worker.
