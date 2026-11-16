# Standard optimizers: actual native AdamW export verification — 2026-10-08

## Result and boundary

**Passed**: host verification of the actual native-trained, History-verified and owner-saved AdamW package `.local/KwipML-model-B7B8325F`. Root performed the App UI run/History/Choose destination/Save package actions; this agent independently read the resulting archive and package using host `.venv/bin/python` (Torch2.14.1). No bundled/signed Python, Swift/build/UI operations, source mutations or catalog acceptance changes were performed by this verifier.

Exact capture identity matched root's requested artifact:

| Identity | Actual value |
|---|---|
| Project | `0E1ADABB-CFCB-4C81-9784-F4255B24182E` |
| Logical run | `B7B8325F-071A-40B7-ADE2-E807D1412C38` |
| Archive directory | `FF5B7E25-E86F-4146-BF9E-C3C73C6108C7` |
| Request identity | `495e39d8c786c5e66877178e92f37d9cf846fc87eec4a4e07977323ff4215dd2` |
| Revision | `1` |
| Worker run | `743d1c3c-bd97-48b7-8be3-f723658da4a1` |
| Actual training device | `mps` |
| Completed epoch | `3` |

Archive path:
`/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/0E1ADABB-CFCB-4C81-9784-F4255B24182E/FF5B7E25-E86F-4146-BF9E-C3C73C6108C7`.

The original supplied path used the logical run ID as the directory name and did not exist. That initial lookup failure is preserved in `.local/optimizer-native-adamw-verification.log`. Read-only inspection located the exact capture by logical ID/request/revision; root independently confirmed the FF5 directory. No alternate run was substituted.

## Captured recipe, ownership and artifacts

Compiler5's captured optimizer wrapper contains the exact canonical four-key recipe and qualified source `{domain: control, localID: optimizer}`. The verifier checked the immutable workflow node payload, unique `Train.optimizer` connection, capture/workflow/registry/runtime evidence inventory hashes, actual compiled graph, manifest/checkpoint recipe equality and exported recipe code identity. Native recipe: AdamW lr.02, beta1.9, beta2.999, eps1e-8, weight_decay.01, amsgradfalse, maximizefalse.

The connected authored Run setup retains learningRate.03. The captured compiled plan and actual manifest use effective learningRate.02, matching the explicit optimizer. The verifier initially mislabeled compiled effective settings as inactive settings; this ignored-helper label was corrected to resolve the actual qualified Run setup node and connection. Initial numerically passing result/log was preserved under `.local/optimizer-native-adamw-initial-label-20261008` and `.local/optimizer-native-adamw-verification-correct-path.log`. Final rerun exited0 into the requested output directory.

Saved export inventory hashes and sizes matched all seven files. Model, checkpoint and training recipe were byte-identical to the immutable training archive. Checkpoint epoch matched the manifest, ordered ownership matched actual exported named trainable parameters and provenance, and strict runtime optimizer-state validation passed for all six parameters with six actual updates each. Binary optimizer state is verified directly by the host runtime; no native JSON moment/counter projection or exact MPS continuation claim is made.

## Independent numerical evidence

All16 held-out inputs/labels matched the sealed preparation's validation tensor and captured prediction rows. The scalar oracle implements learned affine branches, ReLU, Python `math.tanh`, concatenation and the learned classifier using actual learned weights/biases; expected values do not call the production/exported forward operations.

| Check | Maximum absolute difference |
|---|---:|
| CPU full-batch vs independent scalar logits | `1.0319589127671946e-7` |
| MPS full-batch vs independent scalar logits | `1.0863330757171497e-7` |
| CPU/MPS batch-one vs full-batch logits | `1.1920928955078125e-7` |
| Captured eight-stage values vs scalar oracle | `5.591976270924448e-8` |
| Saved inference CLI batch1 vs scalar oracle | `9.30946137955857e-8` |
| Saved inference CLI batch16 vs scalar oracle | `1.0319589127671946e-7` |
| Fresh exported AdamW first update, CPU/MPS | `6.210871883460811e-8` |

Every predicted class matched exactly. Inference preserved the model state. Forward checks use `2e-5` absolute/relative tolerance; fresh update uses `3e-5`. The fresh-update expected values come from an independent scalar optimizer calculation using a trusted mixed-sign gradient. This verifies fresh optimizer construction from the captured recipe, not restoration/continuation of training state. The saved inference helper hashes the optional recipe without executing it; recipe execution occurs only in the explicitly requested fresh-training check.

## Reproduction and exact hashes

```sh
.venv/bin/python .local/verify-optimizer-native-export.py \
  --archive '/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/0E1ADABB-CFCB-4C81-9784-F4255B24182E/FF5B7E25-E86F-4146-BF9E-C3C73C6108C7' \
  --export .local/KwipML-model-B7B8325F \
  --output .local/optimizer-native-adamw-20261008
```
Use a fresh output path for replay; the verifier refuses overwriting existing evidence. Actual terminal result is `.local/optimizer-native-adamw-20261008/verification.json` (status `passed`); CLI logs live beside it. Package/archive remained read-only.

| Artifact | SHA-256 |
|---|---|
| Verifier | `25da33bb4bf716cef93be03b3c58fdabe048a915f4698d98033271dce6fc9fed` |
| Final verification JSON | `1e75f1e5ab23a2d9ec3a747861f9a73f8439a0520e921a63e17de90b927fa367` |
| Final terminal log | `06181a35765f3d874fa117b51588ee258f217f1ab06631ddba7afe27cd47520d` |
| capture.json | `ff7a60a29dc9bcda8442a581025e139a1e9c2b7b1c7dbbd34ae5460b5e062442` |
| Training manifest | `a5c5aa4998b1a0c9cacce765f6ae0a5692c83a525b66822a55fbae12f9bf10d1` |
| Preparation manifest | `f400d534dfb6d6c959ce9b3fd54fa4c330ca29987b7ed4f40da297c001f4d49b` |
| export-manifest.json | `a6841db8c583da602a616f459b78da3f8442a9dd233dc7600a5f9e7a79387c4b` |
| checkpoint.pt | `354392077cc8131261f2839ad3f60aa51e2ed1df73e8a955602338f59a89d3ab` |
| model.py | `2c313b7c408fc050f2dd55b48ea2d0ea14a2002705a47a4a3cb5d673c4caf7b7` |
| training_recipe.py | `83727f919218015e20f98629f419ea3c8bee862c8024cc402f6dd0738edc452b` |
| inference.py | `678819e40fc07e58ccc4db3b1d870de4c7c9dde208d13f63286fdfac407853b6` |
| input-contract.json | `5a7bdf1ab69bf1dcc0d8a79c4203b44b36854d79637be1f8b1e7932f0e2d2a61` |

This receipt covers this actual saved AdamW native run only. All-five CPU/MPS host canonical evidence, exact CPU continuation and MPS weights-only checks are separately recorded in `TORCH_STANDARD_OPTIMIZERS_CANONICAL_20261008.md`. Root owns cold History replay, final prior regression/accounting, packaging/distribution and acceptance decisions. All prior49 frozen receipts remain untouched.
