# Recurrent native export independent proof — 2026-10-08

Status: **terminal host GREEN** for the actual root-observed native saved LSTM package. This worker performed no native interaction, Swift build, signed bundled Python invocation or coverage mutation. Root reported the native guided LSTM lesson pass, actual run completion, UI export and History verification of 30 artifacts. The host verification below independently checked those saved bytes and numerical behavior; broader native/cold UAT remains root-owned.

## Actual identities

| Identity | Value |
| --- | --- |
| Project | `43E0D7C3-27E5-4E14-ABA5-C41337101520` |
| Logical run | `B1BF7D99-5525-4411-B3B1-B228EFF956F8` |
| Request identity | `c51c4f7e9843989f07c647d74ccd41a22d817c424fcceb7a7cc07e5aad867428` |
| Source revision | `0` |
| Archive directory | `40C93F9B-750A-4572-85E2-6DF942007441` |
| Worker run | `3ef4342a-2dc1-4967-a7b4-2e0bdc1ec296` |
| Actual training device | `mps` |
| Component | `lstm` |
| Capture / compiler contract | `1 / 1`, implicit Adam, no scheduler or explicit optimizer recipe |

Actual package: `.local/KwipML-model-B1BF7D99`.

Actual archive: `/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/43E0D7C3-27E5-4E14-ABA5-C41337101520/40C93F9B-750A-4572-85E2-6DF942007441`.

## Integrity and independent measurement

The verifier opens artifact directories and files relative to anchored descriptors with `O_NOFOLLOW`, requires bounded regular files, verifies byte counts and SHA-256, and rechecks the actual export/archive bytes at completion. It verifies every one of the 30 saved outcome artifacts, the capture, workflow, model/data registries, runtime evidence and captured sources. The actual package has exactly six pinned files: README, checkpoint, inference helper, input contract, model source and requirements. Exported model source and checkpoint must exactly match the immutable archived worker outputs. Logical run, request identity, project and revision must match the capture and portable package.

The model manifest, captured graph/hash, checkpoint bytes/contract/epoch, metric-history hash and recurrent registration order all agree. The copied preparation is loaded using its pinned manifest, dataflow digest, qualified source ports, feature/target labels and float32 two-feature contract. Raw source identity must occur in the capture's source pins. The actual split contains **48 training / 16 held-out observations**.

Expected values come from the frozen plain-Python scalar recurrence/gate reference and scalar learned linear head from the canonical proof. No native recurrent operation, production adapter or exported helper computes expected values. The verifier independently derives native LSTM gate/linear weight names and shapes, rejects extra/missing or pretend hidden-state parameters, and requires finite dense float32 saved weights. Strict implicit Adam saved state is validated with the actual completed batch count on CPU and MPS.

The actual exported model is imported from its pinned bytes without product-runtime imports. CPU and MPS full and one-row inference match scalar logits and class choices. Native portable-layer `out`, `hidden` and `cell` are measured separately against scalar memory-state results. The actual exported inference CLI, invoked with host `.venv`, passes both batch sizes 1 and 16 with the held-out rows and matching classes.

| Measurement | CPU maximum absolute error | MPS maximum absolute error |
| --- | ---: | ---: |
| Full logits vs scalar | `2.9299553491735075e-08` | `2.312720999778084e-08` |
| One-row logits vs scalar | `3.9956437557914226e-08` | `2.312720999778084e-08` |
| `out` vs scalar | `5.354746906993668e-08` | `4.6915719614792906e-08` |
| `hidden` vs scalar | `5.354746906993668e-08` | `3.864462549962866e-08` |
| `cell` vs scalar | `6.875445568432781e-08` | `8.396257394904794e-08` |

CLI maximum errors: batch 1 `3.9956437557914226e-08`; batch 16 `2.9299553491735075e-08`. All class choices agree. No export or archive bytes changed during verification.

## Actual command and evidence

```sh
.venv/bin/python .local/verify-recurrent-native-export.py \
  --archive '/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/43E0D7C3-27E5-4E14-ABA5-C41337101520/40C93F9B-750A-4572-85E2-6DF942007441' \
  --package .local/KwipML-model-B1BF7D99 \
  --output .local/recurrent-native-export-20261008-final \
  --project-id 43E0D7C3-27E5-4E14-ABA5-C41337101520 \
  --run-id B1BF7D99-5525-4411-B3B1-B228EFF956F8 \
  --request-identity c51c4f7e9843989f07c647d74ccd41a22d817c424fcceb7a7cc07e5aad867428 \
  --component lstm
```

Terminal **exit 0**, no skips or warnings.

| Evidence | SHA-256 |
| --- | --- |
| Final verification JSON `.local/recurrent-native-export-20261008-final/verification.json` | `06dee9c60987b0382196b635fdd400a407d8710cf56e7dbb50468ee71e1b8d6e` |
| Final log `.local/recurrent-native-export-final.log` | `01df506d21aedbe573418b9c25fc7cc24ea4438534ae9299449a797b8af3e93e` |
| Native checkpoint | `1c0d8e2a47a820a196d83c31bc27607c66fdb1be067848c8de539da91c1912bc` |
| Captured request | `981e324469409c4b938cacdabbbfdf401a0ab4407c39213de204b00bea0ab5b5` |
| Native model manifest | `605865e85ccc3eceb664c2b997c9b44a16dec0e0b1b3a39b1f16874f00e2849d` |
| Actual export manifest | `6b28f90a1a3f25fcd1b09e4502652dba6ab1dab2116c38b1b0f8a026cf35a83c` |
| Independent verifier source | `c0aca304d039acfa835aac8f5120a9ac85bbe2abf786b05b8f823152bdd1b3b7` |
| Frozen canonical oracle proof | `0de7b0d0e818e76439c8ea635e4708a3d084d56aa5b7624220137a747ddd515c` |

The first verifier attempt inherited a scheduler-template schema 7/compiler 6 assertion and refused the genuine implicit-recipe capture before any numerical claim. This was a verifier-fixture assumption error, not a product failure. The correct native pair 1/1 was explicitly pinned; the original failed log is preserved at `.local/recurrent-native-export-capture-assumption-red.log` and the final passed evidence occupies a distinct directory. No source hashes or recorded native identities were silently refreshed.

This measured native package supplements the genuine six-factory CPU/MPS canonical host matrix and the runtime state/projection/lifecycle tests. It does not independently grant acceptance or assert external mutable state/packed sequence support.
