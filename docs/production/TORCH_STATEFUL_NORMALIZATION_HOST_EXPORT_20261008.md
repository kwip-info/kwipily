# Remaining normalization canonical host exports · 2026-10-08

Finite host-only closure of the five genuine prepared Core example classifiers, independently serialized by root to `.local/stateful-normalization-canonical-examples`. No substitute Python graph factory, native App observation, signed bundled interpreter or acceptance/coverage grant is involved. Host `.venv` Python3.13.6/Torch2.14.1 CPU/MPS executed all ten model-training combinations.

The preparation comes from the actual previously saved native archive `651FC9FF-1B18-4355-A466-793E8A82AEE7/C0D9D4A2-9688-4B71-8BAC-981A0DFFA931` under the App container workflow-runs directory, reused with root authorization because the preparedClassifier source/partition contract is unchanged. Manifest digest: `f400d534dfb6d6c959ce9b3fd54fa4c330ca29987b7ed4f40da297c001f4d49b`. Exact prepared rows: 48 training and 16 validation. Its source preparation digest/feature/target selection are admitted by the real prepared-data loader rather than inferred from a similar dataset.

Command: `.venv/bin/python .local/run-stateful-normalization-canonical-exports.py --graphs .local/stateful-normalization-canonical-examples --archive <the exact archive above> --output .local/stateful-normalization-host-canonical-20261008 --devices cpu mps`. Terminal exit0. Each graph trained three epochs with seed42, batch32 and learning rate0.03, then loaded actual saved model/checkpoint for both CPU and MPS inference. Actual runtime manifests confirm selected training device and completed epoch3; checkpoint/prediction/activation hashes and byte counts match the saved manifests. The current helper/source/Core factory hashes and each genuine serialized graph/checkpoint/manifest hash are bound in receipt.json.

Expected forward results use plain Python arithmetic: saved learned 2→channel-grid affine projection, channel/group scalar statistics, saved learned affine scale/offset where present, then saved classifier. BatchNorm uses its actual saved running means/variances; InstanceNorm computes each observation/channel's spatial mean and correction-zero variance; GroupNorm computes each observation's two channel groups. Expected values never call a runtime adapter, engine forward or native normalization. Exact authored controls, channels, reshape sizes, layer widths and connectivity are verified against the independent reference architecture.

All 16 held-out rows per run matched independent class decisions, actual saved prediction classes and saved standalone inference CLI classes. Saved CPU/MPS models matched scalar results with full and one-observation batches. Every captured stage's available values matched the scalar reference. The saved actual inference helper was copied into each host package with a verified four-file inventory/input contract and executed in fresh host subprocesses with batch1 and batch16. These are honest host-created packages, not native UI exports.

| Component | Training device | Maximum CPU/MPS scalar error | Maximum standalone CLI scalar error |
|---|---|---:|---:|
| batch_norm1d | cpu | 1.92965545e-07 | 1.92965545e-07 |
| batch_norm1d | mps | 2.24968228e-07 | 1.50307466e-07 |
| batch_norm2d | cpu | 2.12212468e-07 | 2.17942265e-07 |
| batch_norm2d | mps | 2.2111035e-07 | 2.2111035e-07 |
| instance_norm1d | cpu | 5.77328148e-07 | 5.77328148e-07 |
| instance_norm1d | mps | 3.20539098e-07 | 3.20539098e-07 |
| instance_norm2d | cpu | 2.56183327e-07 | 2.56183327e-07 |
| instance_norm2d | mps | 2.9323802e-07 | 2.3998857e-07 |
| group_norm | cpu | 2.76374947e-07 | 3.37974412e-07 |
| group_norm | mps | 4.09073766e-07 | 2.48520932e-07 |

Tolerance was absolute2e-5/relative2e-5; maximum actual errors across all runs: scalar 5.77328148066e-07, microbatch 3.57627868652e-07, captured-stage 2.54007993306e-07. This does not claim untracked BatchNorm microbatch independence: the canonical BatchNorm examples track running statistics and are measured in evaluation mode. Remaining general runtime/state/resume/refusal evidence is in `TORCH_STATEFUL_NORMALIZATION_RUNTIME_20261008.md`; native, App and broader prior-replay gates remain separate.

Evidence hashes:

- `.local/run-stateful-normalization-canonical-exports.py` SHA-256 `837c4cbc61b3f9590310f9482f4dc7194e5ca61d7303f986e9a7cd3e3fa0fe14`.
- `.local/stateful-normalization-host-canonical.log` SHA-256 `a7827bdd20179467562f4b670473769a52527d63bc33b28682d39020d75b5537`.
- `.local/stateful-normalization-host-canonical-20261008/receipt.json` SHA-256 `7bafdacebc34de3955815cd10deaf43f2d53cf02dc380cda9fde5ab83f7eb764`.
