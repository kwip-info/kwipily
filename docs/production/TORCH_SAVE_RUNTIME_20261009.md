# Checkpoint lifecycle runtime discovery and storage preflight — 2026-10-09

Finite TORCH-LIFE.1 runtime slice under the frozen TORCH_SAVE_CLOSURE.md descriptor. This is not save/load capability acceptance, native UI evidence, or a new checkpoint format.

## Implementation

- checkpoint_lifecycle.capability() returns a fresh exact reviewed descriptor. engine.capabilities() adds only lifecycleCapabilities.checkpoint_lifecycle; no training behavior or component discovery entries change.
- Lifecycle loading now invokes the existing frozen_checkpoint._check_storage_archive before torch.load. A lazy local import avoids the existing module dependency cycle. The shared preflight retains the current uncompressed Torch ZIP storage format, at most10,000 unique entries, at most100MB total expanded entry bytes, and refusal of encrypted/compressed/invalid ZIP archives. No unsafe loader retry or resource ceiling increase is introduced.
- Existing checkpoint identity/semantic compatibility checks still precede deserialization, and Torch load retains weights_only=True with CPU mapping. Existing scheduler trace transport remains unchanged. No additional expanded-allocation guarantee field is advertised.

## Actual TDD and bounded verification

Host .venv/bin/python, product repository. No Swift/build/native/signed bundled Python execution by this agent.

RED command: `.venv/bin/python -m pytest -q runner/tests/test_checkpoint_lifecycle_capability.py`. Actual7failed/1passed/zero skips in2.56s: exact descriptor/sibling absent, and five malformed archive controls reached a forbidden Torch loader. Valid owned parent loading already passed.

GREEN command: `.venv/bin/python -m pytest -q runner/tests/test_checkpoint_lifecycle_capability.py runner/tests/test_checkpoint_lifecycle.py runner/tests/test_frozen_checkpoint.py runner/tests/test_scheduler_parent_transport.py`. First45passed/zero skips in3.00s; a final explicit nonallowlisted-pickle refusal test was then added. Final **46passed/zero failures/skips in2.92s**, terminal exit0. One intentional zipfile duplicate-name warning belongs to the duplicate-entry refusal fixture.

The nine new checks independently pin the descriptor, mutable-container isolation, compressed storage, expanded-byte overflow using a tiny reviewed bound, duplicate entries, too many entries, invalid ZIP, real safe CPU mapping/parent immutability, and nonallowlisted pickle refusal with exactly one weights-only load attempt. The final command also replays existing ordinary lifecycle, frozen checkpoint and scheduler-parent contract tests. No broad release/backend parity claim follows from these bounded checks.

## Preserved discovery and source identity

Before/after canonical componentCapabilities projections are byte-identical:77 existing descriptors, SHA-256 `948eb7b62183991bdd14687b4f4f95006ee9bad2d8a94bf279a28578758233e2`. Generated registry and component authoring files were not edited. Old frozen receipts/catalog were not changed.

| Path | SHA-256 |
|---|---|---|
| `runner/checkpoint_lifecycle.py` | `66c5cd27da00fe4e3a794c6daa798284fd26f734af051dca414d3aee9c988ab5` |
| `runner/engine.py` | `f62d9be7898c03b3242104e12092f2cce1864c82156635b52ab6b5c491f16f91` |
| `runner/tests/test_checkpoint_lifecycle_capability.py` | `e1c36669e7f199d61d318c19b4bc915d8287a7a74e209039f1e3be924aac457c` |
| `.local/checkpoint-lifecycle-capability-red.log` | `ec682edb4f90f9bc0ad5ca62642713edbd90e7fa48f90c65c64ebe1885476e7a` |
| `.local/checkpoint-lifecycle-capability-green.log` | `6afe5da8bc6c4e5909c524c34fb4d5c3ae3b99ca14ea39d360b8b8c73cb4fe38` |
| `.local/checkpoint-lifecycle-capability-final-green.log` | `4755b48b2f7aeee149550c10486d40130273c26ed735eb2cf3bd83a907b51d25` |

The reused unchanged frozen preflight source hash is `b4bc5f685de6358bc1d0b3cfa48d2f5575d8e48abdadad2f941bde9f39909b76`. Machine-readable pins: `.local/checkpoint-lifecycle-capability-pins.json`. Sources are frozen for parent integration. Bundle/strict native admission, lessons/art, final independent mode matrix and native closure remain separately owned.
