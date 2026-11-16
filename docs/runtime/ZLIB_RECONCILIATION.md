# Exact standalone archive: system-z metadata reconciliation

2026-10-05. Technical packaging evidence, not redistribution approval. No live payload was refreshed for this slice.

## Observed facts and bounded inference

The retained Astral Python 3.13.6 build 20250814, target `aarch64-apple-darwin`, full archive SHA-256 is `f169b076da928b34ceff3578438fe68b65ef8009cae64fbc7150515a0e61d007`. Its original `PYTHON.json` references both `licenses/LICENSE.zlib.txt` and `licenses/LICENSE.zlib-ng.txt`; the archive contains the former but not the latter. Metadata declares the zlib extension link exactly `{"name":"z","system":true}`. Observed payload libpython links `/usr/lib/libz.1.dylib`, and its interpreter reports zlib runtime 1.2.12. No local/static libz appears in the archive listing.

Upstream's exact release [download definitions](https://raw.githubusercontent.com/astral-sh/python-build-standalone/20250814/pythonbuild/downloads.py) declare library name `z` for both zlib 1.3.1 and zlib-ng 2.2.4. Its [license annotation function](https://raw.githubusercontent.com/astral-sh/python-build-standalone/20250814/pythonbuild/utils.py), `add_licenses_to_extension_entry`, adds matching notices for every matching download, including system links. Therefore the missing zlib-ng reference is explained as an inference from metadata plus binary linkage, not a blanket license conclusion or proof that all third-party redistribution obligations are satisfied.

The referenced candidate zlib-ng 2.2.4 source archive digest is `00bbd88709bc416cb96160ab61d3e1c8f76e106799af7328d0fe434dc7dd5004`, declared Zlib license. Its [license text](https://raw.githubusercontent.com/python/cpython-source-deps/zlib-ng-2.2.4/LICENSE.md) was inspected; no supplemental notice is inserted or misrepresented as extracted from this Python archive.

## Canonical build behavior

Fresh builds require an explicit local archive; the packager neither downloads it nor depends on a hidden `.local` default:

```sh
.venv/bin/python scripts/runtime_package.py \
  --output .local/runtime-build/KwipMLRuntime \
  --standalone-archive .local/runtime-feasibility/cpython-full.tar.zst
```

The supplied archive is opened without following the final symlink, required to be regular, bounded at 128 MiB and copied from that descriptor into a private temporary snapshot. SHA-256 must match the exact known archive before `_build` mutates its destination. All subsequent inventory extraction uses that verified snapshot, preventing external pathname replacement from changing the inspected bytes. The installed managed interpreter must report the supported Python request, arm64 host and build 20250814; unknown distributions fail with unresolved evidence rather than borrowing these notices.

The existing standalone inventory extractor reproduces `standalone-license-evidence/PYTHON.json`, original `inventory.json` and the actually present referenced notices. The new separate `reconciliation.json` records the inspected libpython zlib dependency paths/runtime version, archive/build/target, authoritative sources and `explainedMetadataOvermatch` only for the exact case above. Other combinations remain `unresolved`. Other missing references stay in `remainingMissingReferences`; the original inventory always retains every missing reference. Both metadata and runtime results remain `redistributionReviewComplete:false`.

`runtime-manifest.json` references the standalone inventory/reconciliation and their hashes alongside existing wheel/CPython notice inventory. Existing `--refresh-worker-integrity` behavior is unchanged; it does not retrofit standalone evidence into a live payload. Root must review and authorize any future payload attachment/rebuild separately from this source change. A complete packaging run was not executed in this slice.

## Evidence and verification

Read-only investigation commands included:

```sh
shasum -a 256 .local/runtime-feasibility/cpython-full.tar.zst
/usr/bin/tar -tf .local/runtime-feasibility/cpython-full.tar.zst
/usr/bin/otool -L .local/runtime-feasibility/payload5/KwipMLRuntime/python/lib/libpython3.13.dylib
.local/runtime-feasibility/payload5/KwipMLRuntime/python/bin/python3.13 -I -B \
  -c 'import zlib;print(zlib.ZLIB_VERSION,zlib.ZLIB_RUNTIME_VERSION)'
```

Ten initial tests failed before the new API/precondition existed. After implementation, exact-case, unknown archive/target/Python/build/runtime version, static/local/extra-library linkage tests passed. Additional tests preserve unrelated missing notices and reject changed/symlink archives before destination creation. An integration test inventories the retained exact archive into an isolated temporary directory, probes the existing payload read-only and confirms actual zlib notice copying, original missing reference preservation and separate technical reconciliation. It skips explicitly when retained local evidence is unavailable; it executed on this host. No binary, original inventory, live payload or signing state was changed.

Final runtime-focused suite: `.venv/bin/python -B -m pytest scripts/runtime_tests -q` — **32 passed in 3.18 seconds**, including real isolated archive attachment. No skips were reported.
