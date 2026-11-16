# Standard optimizers: genuine canonical host verification — 2026-10-08

## Scope and result

**Passed: all five genuine Core-emitted optimizer workflows trained on CPU and MPS (10 cases), with 30 completed training artifacts across full, first-epoch, exact-continuation and fine-tuning runs.** Five attempted exact MPS continuations correctly refused. This is host verification on Torch 2.14.1 using `.venv/bin/python`, not an App UI/native submission or acceptance claim. No signed bundled Python, Swift compiler or UI was used by this verifier.

Fixtures came from root's actual serialized Core example factory under `.local/optimizer-canonical-examples/{optimizer_sgd,optimizer_adam,optimizer_adamw,optimizer_rmsprop,optimizer_adagrad}`. Each exact `model.json`, `recipe.json` and training settings drove the run; the emitted workflow/dataflow files are also individually hashed in the result. Settings: three epochs, seed42, batch32, explicit recipe rate.03; requested `auto` was deliberately exercised as explicit CPU and explicit MPS for this matrix. Root reported the integrated Core/App 26-test GREEN that emitted these fixtures; this receipt does not independently claim that Swift test run.

Sealed preparation was read from the existing actual BN1 native archive:
`/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/056FF2E4-2C6A-4E0E-81B5-4EC1A1B6A7B0/F3F3695C-71E6-453A-ADB4-58A09EB6F928`.
The current emitted canonical dataflow digest exactly matched that retained preparation: 48 training observations and16 validation observations. Preparation manifest SHA-256 `f400d534dfb6d6c959ce9b3fd54fa4c330ca29987b7ed4f40da297c001f4d49b`; dataflow digest `dbe30871e37c5d19246469161a4e4c1c88e755c22ca245ed2a48da88188d89b3`. Prepared data is reused honestly; these are new host optimizer training runs, not new native preparation captures.

## Independent checks

The scalar forward oracle reads actual learned weights/biases and implements each graph stage using Python loops: two learned affine branches, ReLU, `math.tanh`, concatenation and the learned classifier. It never uses the production model's forward operation to construct expected values. All16 held-out observations/classes per case, captured stage values and prediction-input identities matched. Maximum scalar-logit error across all10 cases: `4.969540485788571e-7`; tolerance `2e-5` absolute/relative. Exported learned models ran on both CPU and MPS with full-batch and batch-one inference; largest full/microbatch difference `7.152557373046875e-7`.

Each case verified checkpoint/manifest recipe identity, effective learning rate, artifact bytes/hashes, exact ordered trainable-parameter ownership and strict native optimizer state. Each of six owned parameters had six actual updates after three epochs. Optional exported `training_recipe.py` created a fresh optimizer on the learned exported model; a trusted mixed-sign gradient produced the independently calculated scalar first update on both CPU and MPS. Maximum first-update error `1.0231620928280449e-7`, within `3e-5` absolute/relative. The fresh-step expectation uses the independent scalar-update implementation from runtime tests, not the exported constructor or native optimizer update.

Actual saved portable inference helper subprocesses, batch1 and batch16, passed for all10 cases (20 CLI executions). Each inference package inventories the optional training recipe without executing it. Expected classes matched exactly and largest CLI scalar error was `4.48423888244065e-7`.

All five CPU one-epoch→three-epoch continuations matched uninterrupted training **exactly** for learned tensors, actual optimizer state, RNG state, per-parameter update counts and metrics. All five MPS exact continuations refused with the existing CPU-only guarantee. All five MPS weights-only fine-tuning runs completed with `weightsOnly` lineage and initial epoch0; no exact stochastic MPS continuation guarantee is claimed.

## Real integration RED and bounded repair

The first canonical attempt preserved `.local/optimizer-canonical-host.log` and `.local/optimizer-canonical-host-20261008`. SGD CPU numerical/export/fresh-step checks passed, but inference refused `training_recipe.py` in the export inventory. This was a real integration gap: the native export builder already includes the optional file while the inference helper's allowlist omitted it.

Under root's explicit lease, new tests first captured meaningful RED: **2 positive failures/2 negative passes/31 deselected**, `.local/optimizer-inference-recipe-red.log`. The only production repair was `Sources/KwipMLApp/Resources/inference.py`: add optional `training_recipe.py`, bounded to8192 bytes and valid UTF-8, verify its inventory hash, and never execute/import it during inference. The four required artifacts and unknown-file refusal remain intact. Tests use executable poisoned recipe text, exact8192-byte content, changed-byte hash rejection, oversized and invalid UTF-8 refusal. Combined actual CLI/portable inference regression: **59 passed in31.99s, zero failures/skips**, `.local/optimizer-inference-recipe-green.log`.

The entire canonical matrix restarted into a fresh directory after GREEN; first failure artifacts were retained. The corrected verifier exited0, and `.local/optimizer-canonical-host-corrected-20261008/receipt.json` has terminal status `passed`, ten case results, hashes of all25 fixture files, all30 completed artifact inventories/run IDs and per-case numerical/continuation results.

## Reproduction and hashes

Command:
```sh
.venv/bin/python .local/verify-optimizer-canonical-workflows.py \
  --fixtures .local/optimizer-canonical-examples \
  --archive '/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/056FF2E4-2C6A-4E0E-81B5-4EC1A1B6A7B0/F3F3695C-71E6-453A-ADB4-58A09EB6F928' \
  --output .local/optimizer-canonical-host-corrected-20261008
```
The helper refuses overwriting an evidence directory; reproducing requires a fresh output path.

| Evidence/source | SHA-256 |
|---|---|
| `.local/verify-optimizer-canonical-workflows.py` | `9e51ae357db8a9208e533d4a1c05ac97d7667e75223e0bbdbc4eef3bf38bb954` |
| Corrected `receipt.json` including30-artifact inventory | `8ee7aecffae0dc3a408275e62ceef2353c1a9633d9d9f2455e83d872736c8ac3` |
| First failing canonical log | `d878a34b1038809e0ab6d3950114c5f7344f695f486e1581eb305d9408b46345` |
| Corrected canonical log | `7d168d8263133d5a9b7e0e1fc6de1a17cd036ce40ffb0950f364dac399c8bac2` |
| Optional recipe RED log | `fe684f795c0c406118d57a4b51418a6b3fd7f2606dbdda339ad19fe32659d67b` |
| Optional recipe GREEN log | `9d7a74efeedaf0fc00825647e37893f1ee3be5932d6d087cbc0472f027c8099e` |
| `runner/tests/test_optimizer_cli_delivery.py` | `2c6093cb2c5d3ec74fe576e73b4ced2ba83de9bd928476373459c3510e4eeb65` |
| `Sources/KwipMLApp/Resources/inference.py` | `678819e40fc07e58ccc4db3b1d870de4c7c9dde208d13f63286fdfac407853b6` |
| `runner/optimizer_configuration.py` | `6ae9578e3e3d1f94c3ac57f399d02d7c53d7f261316f23bf56b0e2c40791602a` |
| `runner/engine.py` | `35d9b9b6f2be62b62caf2f55414cd92ee338045d5b1b37161144be454ecef4d6` |
| `runner/checkpoint_lifecycle.py` | `9710bc00a52e792722548a6e241ba2f302c2eb8b1f43a75408db351d38319277` |
| `Sources/KwipMLCore/OptimizerWorkflowExample.swift` | `9aecc65afed849a4460edc21df935c3a1c92b2d641c4ec9ee3610a5e77e582be` |

This finite receipt does not assert native optimizer UI execution, packaging after the inference repair, cold learning acceptance, clean-machine distribution, full prior-source regression or coverage credit. Those gates remain root-owned.
