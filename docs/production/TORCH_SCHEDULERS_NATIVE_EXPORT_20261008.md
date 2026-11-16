# Actual native OneCycle export verification — 2026-10-08

**Independent host verification passed**, terminal exit0. This worker verified the actual package saved by root through the native application; it did not interact with the native UI or create a substitute export. Root owns the observed run, Activity, saved publication and cold History checks.

## Exact native identities

| Identity | Value |
|---|---|
| projectID | `2805A0A3-41A6-4B8C-8CAD-B35F30EE3004` |
| runID | `BB7E23F3-C6AB-4DFB-89B2-3D85C766D4BB` |
| requestIdentity | `2e52398281bb84e08b9e05fa7f6f1eed5e5350aa7538d785e26031222939925f` |
| sourceRevision | `0` |
| archiveDirectoryID | `F89F626A-895F-43F3-BB22-2D0A851C28CA` |
| workerRunID | `dc1a9214-d3c4-43d9-bcd0-cd10f6cafff9` |
| trainingDevice | `mps` |
| torchVersion | `2.14.1` |

Owner-saved package: `.local/KwipML-model-BB7E23F3`. Container archive: project 2805A0A3-41A6-4B8C-8CAD-B35F30EE3004, directory F89F626A-895F-43F3-BB22-2D0A851C28CA. Logical application run and training worker run remain distinct identities.

The complete native outcome inventory has33 artifacts. Each file was reopened with directory-relative `open`/`O_NOFOLLOW`, required to be a bounded regular file, and compared to its saved size/SHA pin. The capture, actual workflow, registries, runtime evidence and all captured source bytes were also pinned independently. No owner package byte was modified.

The portable manifest matches the captured revision0/request/run/training manifest and canonical recipe from `control:scheduler`. Its actual workflow edge leads from that source to Train.scheduler. All eight portable inventory files passed exact hash/size checks; model.py/checkpoint.pt/rate-trace.json/training_recipe.py also match the native archive byte for byte.

## Model, source and scheduler evidence

The held-out input came from the actual pinned preparation manifest:48 training and16 validation rows, two prepared float32 features. Preparation graph/manifest/source pins and captured model graph match native training provenance. The scalar parallel-model oracle uses checkpoint weights and independent affine, ReLU, tanh and concatenation arithmetic.

The MPS training checkpoint has six completed OneCycle ticks across three epochs and two batches per epoch. All six ordered learned parameters have update count6. Actual checkpoint state, canonical recipe, trace contents, tick/epoch/batch geometry and gradient-presence indices were checked. Each used/next learning rate and momentum matches a separate scalar interpolation formula exactly (maximum observed error0). Strict production scheduler-state replay and optimizer-state ownership validation passed independently on CPU and MPS; no unchecked `load_state_dict` was treated as validation.

## Measured exported behavior

| Device | Scalar inference maximum error | Full vs one-row maximum error | Fresh helper learned-update maximum error |
|---|---:|---:|---:|
| cpu | 1.54e-07 | 1.19e-07 | 3.87e-08 |
| mps | 1.39e-07 | 1.79e-07 | 1.33e-07 |

Actual package inference CLI passed with batch1 and batch16. Maximum scalar errors were 9.104797304981815e-08 and 1.5426294819320674e-07. Predicted classes matched the independent scalar oracle for every held-out row.

The actual exported4200-byte training helper created fresh optimizer/scheduler state on CPU and MPS. Ten independent weighted-gradient updates exercised the complete planned horizon, comparing rates, momentum and learned values at every step. This fresh-training check does not claim restored native training state or saved data. The native run has six completed ticks; the helper independently exercised all ten planned ticks.

## Reproduction and results

```sh
.venv/bin/python .local/verify-scheduler-native-export.py \
  > .local/scheduler-native-export-20261008.log 2>&1
```

The first proof-script attempt terminated with a TypeError caused by the script using `context` instead of the actual `scheduler_context` keyword in optimizer-state validation. Its log is preserved; the correction changes no product behavior or evidence expectation. The corrected full proof completed exit0.

| Evidence | SHA-256 |
|---|---|
| `.local/verify-scheduler-native-export.py` | `a55fc908a3b43e45b8ae8289177fb229f8bd0316bf3930286cf76e130087a739` |
| `.local/scheduler-native-export-first-attempt-20261008.log` | `c251d0f107ad9bbe11019d75a39b71f3a3f0980da9e47bc23078061e05d2fc46` |
| `.local/scheduler-native-export-20261008.log` | `22584bc8d3e8b6f16f42f10a0e8069bd8f949d3225455c2372960218599336ce` |
| `.local/scheduler-native-export-20261008/verification.json` | `8269109fe8337d3b0b721208bc6d2c6f3391c51b0d90ef73bcdbd0e8b46ac06b` |
| exportManifestSHA256 | `7ba7b2f42b4835504e5c4751938ee706f49c255e2bcff47b51d8f61d1bf643f5` |
| captureSHA256 | `800f1e2b798d8d2eb23527945ddedf115038fd906764ac5f0e7c8f67ee1bc93e` |
| preparationManifestSHA256 | `f400d534dfb6d6c959ce9b3fd54fa4c330ca29987b7ed4f40da297c001f4d49b` |
| modelManifestSHA256 | `38ac7a6a5e064cea330b97c19c82f69069a98a99590835c7abd6cf6f6a245e02` |

Detailed JSON records exact source pins and every actual portable file hash. The proof script refuses overwriting existing verification outputs. Native ownership observations are supplied by root rather than inferred from this host run. No Swift, packaging, UI, catalog or tracker mutation was performed here. This receipt contributes finite export evidence; it does not independently grant family acceptance or release status.
