# Packaged worker integrity

The canonical packager now emits `workerIntegrityVersion: 1` and `workerSourceHashes` in `runtime-manifest.json`. Coverage is `launch.py`, `components.json`, and every regular file below `runner/`, including operator registry, examples and lessons. The bootstrap checks the exact inventory and hashes before importing runner code. Missing, changed, extra or symlinked files fail with one JSON error and exit 1. The bootstrap accepts only its verified bundled registry; an external `--registry` override is rejected.

Bounds are 64 KiB manifest, 512 traversed worker-directory entries, 10 MB per file and 32 MB total worker resources. Directory traversal does not follow symlinks. File reads use descriptor-based regular-file checks with `O_NOFOLLOW` and `O_NONBLOCK`; FIFO inputs cannot block the integrity read. Inventory paths are relative and restricted to the documented coverage. Successful launch never updates the manifest or writes bytecode (`-I -B` remains required).

This check is packaged-only. Explicit development execution uses `runner/main.py` directly and needs no packaged manifest. An old/missing packaged integrity version is rejected without checkout fallback.

## Build and migration

Fresh `scripts/runtime_package.py --output NEW_PAYLOAD` creates the inventory automatically after copying worker resources. Packaging continues to exclude tests, bytecode and cache directories.

For an existing **build-owned, unsigned payload**, first copy the current runner snapshot using the same exclusions, the canonical `scripts/runtime_launch.py` as `PAYLOAD/launch.py`, and the current model registry as `PAYLOAD/components.json`. Then run:

```sh
.venv/bin/python scripts/runtime_package.py --refresh-worker-integrity PAYLOAD
```

Refresh preserves other manifest metadata and updates integrity version, complete worker inventory, entry hash and registry hash. It refuses a launcher that differs from the canonical current bootstrap, preventing accidental attestation of an old bootstrap that does not enforce checks. This is a build command; the installed application never invokes it. Sign/re-sign the candidate only after snapshot and refresh. No existing payload or candidate was modified during implementation of this change.

## Trust and remaining limits

A checksum manifest is not self-authenticating. An attacker able to replace both files and manifest can recompute checksums; the signed containing application and its read-only deployment are the authenticity boundary. This check does not hash all Python standard-library or third-party native/wheel files, provide a reproducible-byte build, or claim protection against concurrent malicious mutation after checking. Operating-system code-signature verification, dependency hash locking and distribution review remain separate layers.

The current installed-package and standalone license inventories remain evidence rather than legal acceptance. The exact standalone archive inventory still references absent `licenses/LICENSE.zlib-ng.txt`. Its `LICENSE.zlib.txt` is present and zlib metadata names the system library `z`; those facts do not establish that the missing reference can be ignored. The full metadata and available notices are present in the inspected payload5 `standalone-license-evidence` directory, with `redistributionReviewComplete: false`. Canonical fresh packaging still generates the wheel inventory but does not automatically obtain/reconcile the full standalone archive notices; distribution preparation must incorporate reconciled standalone evidence before signing. No downloads or legal acceptance occurred in this task.

## Test evidence

Initial integrity tests: **10 failed, 1 passed**. The changed-worker test actually executed its replacement code before the guard existed; missing/legacy manifest and other invalid-inventory variants also ran incorrectly. A separate external-registry test failed before override rejection. A stale-bootstrap refresh test failed before canonical-bootstrap matching was added. All are regression tests now.

The relocated learning test now supplies a build-generated manifest, and explicitly refreshes it only when deliberately building a modified registry fixture. This preserves the test of actual registry forwarding while independently testing checksum rejection. Combined verification results are recorded below.

Final combined runner and runtime-packaging verification: **188 tests passed in 31.29s**. No payload5 or application candidate files were changed.

## Current-source build verification

An internally consistent manifest can describe an old worker. `scripts/build-packaged-app.sh` verifies source currency before native build and rechecks the copied payload before signing. It uses the build-host `python3`, or the explicit `KWIPML_BUILD_PYTHON` interpreter path. Neither check refreshes hashes or runs the supplied worker. Other packaging entrypoints must run this check before signing any reused or refreshed payload:

```sh
.venv/bin/python scripts/runtime_package.py --verify-current-worker /absolute/path/to/KwipMLRuntime
```

This read-only build check first validates the payload's integrity manifest, then compares its exact worker/bootstrap/registry inventory with the current repository's shipping inputs. The same copy policy excludes tests and caches for both fresh assembly and verification. Changed, missing or extra resources fail with their relative names. Fresh runtime assembly invokes this check automatically before returning. Refreshing integrity alone does not establish source currency. The verifier neither repairs nor rewrites a payload, and is not invoked at app launch (installed apps do not need a checkout).

This check covers worker sources, resources and component registry. It does not establish that the native app executable or third-party binaries were built from current sources, validate dependency licenses, or replace signing, runtime probes and native UAT.
