# Bundled runtime feasibility — production step 04

> **Historical runtime feasibility receipt (2026-10-05).** Later production signing/source-disclosure closure is recorded in [redistribution evidence](../production/DIST_REDISTRIBUTION_CLOSURE_20261009.md); Apple validation/upload and owner-confirmed review submission are in the [Apple receipt](../production/DIST_APPLE_VALIDATION_20261009.md). The POC identity and unfinished-distribution statements below belong to this original probe, not the current production package. Clean-machine/UAT acceptance remains separate.

Recorded 2026-10-05, America/Chicago. This is a verified worker portability slice, **not completion of the production distribution gate**. Root-owned native sandbox/UI and bookmark evidence must be reconciled separately. Clean-account installation has not been performed; no new account, distribution signing identity, App Store submission or system preference was created/changed.

## Concrete result

A self-contained arm64 runtime payload was built from Astral-managed CPython 3.13.6 (standalone build 20250814) and the existing pinned package versions, including PyTorch 2.14.1 and NumPy 2.5.3. It was copied outside the checkout into an app-resource layout and executed with:

- checkout, uv interpreter directory, system Python framework and Homebrew reads explicitly denied;
- network denied by a Seatbelt command profile;
- working directory outside the checkout and hostile `PYTHONHOME`/`PYTHONPATH` values;
- no virtual environment, uv invocation, installer or developer tool used by the running worker.

Both CPU and MPS completed real 30-epoch training on a controlled local CSV, reporting 11/12 held-out observations correct (accuracy 0.9166667). Standalone exported-model output matched the compiled architecture exactly on CPU after loading each checkpoint. Checkpoint-backed inference previews completed on both devices. SIGTERM cancellation exited 130, emitted no completed event and wrote no success manifest. The probe verified that reading a checkout file actually raised PermissionError, rather than assuming isolation from command arguments alone. Machine: macOS 26.6, build 25G72, arm64.

See [machine-readable relocation evidence](relocation-evidence.json). This command-level Seatbelt probe does **not** establish App Sandbox entitlements, user-selected access, native quit/reopen, or a clean account. It relocated the worker payload in an app-resource directory layout; root's actual packaged application is a separate artifact and UAT target.

## Runtime layout and native contract

```text
Contents/Resources/KwipMLRuntime/
  runtime-manifest.json
  launch.py
  components.json
  python/bin/python3.13
  python/lib/...
  runner/...
  licenses/...
  license-inventory.json
```

Core `RuntimeLocation.resolve(resources:developmentRoot:allowDevelopment:)` returns typed `executableURL`, `entryURL`, `registryURL`, `workingDirectoryURL`, `kind`, `packageVersions`, and `argumentPrefix`. Bundled launch arguments begin `-I -B launch.py`: Python ignores host Python environment/site injection and does not write bytecode. Bootstrap adds its packaged registry when the caller omitted an explicit registry argument, and imports only the packaged runner. It removes inherited MPS-fallback enabling, so unsupported execution cannot silently switch to CPU through that shell environment variable.

A present malformed bundle fails closed; it never falls back to a checkout. Development fallback requires explicit opt-in and existing checkout files. Bundle paths must be relative, bounded and contained even after resolving symbolic links. Runtime manifest is limited to 64 KB. The resolver validates layout/manifest; operating-system code-signature enforcement and package verification are separate layers.

Runtime code/resources are read-only application content. User projects, imported source snapshots and run artifacts must live outside the bundle. Root owns `workspaceDirectory`/Application Support integration, source bookmarks and their UI verification. The current bundle identifier remains `info.kwip.KwipML.POC`.

## Reproduction

Build-time prerequisites are uv, Python and Apple developer command-line tools. They are not runtime requirements. These commands create local ignored payloads and use ad-hoc signing only:

```sh
uv python install 3.13.6
.venv/bin/python -m pytest scripts/runtime_tests -q
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter RuntimeLocationTests
.venv/bin/python scripts/runtime_package.py --output .local/runtime-build/KwipMLRuntime \
  --standalone-archive .local/runtime-feasibility/cpython-full.tar.zst
.venv/bin/python scripts/runtime_audit.py .local/runtime-build/KwipMLRuntime --sign-ad-hoc --report .local/runtime-build/native-audit.json
.venv/bin/python scripts/runtime_probe.py .local/runtime-build/KwipMLRuntime --report .local/runtime-build/relocation-probe.json
scripts/build-packaged-app.sh .local/runtime-build/KwipMLRuntime
```

Use a new empty payload destination; the packager refuses to overwrite existing content. It copies the uv-managed interpreter and modifies only that copy. The copied EXTERNALLY-MANAGED marker is removed for package assembly; the original managed interpreter is unchanged. Console wrappers with build-directory shebangs are omitted; Python aliases remain internal relative links. Bytecode caches and runner tests are excluded.

uv's installed libpython dylib identity pointed at its user installation. The packager normalizes that copied identity to `@rpath/libpython3.13.dylib` and re-signs it. The native audit also normalizes the copied OpenMP dylib's absolute build-machine identity, then signs/verifies each Mach-O file. It rejects non-system absolute native dependencies. These are build-copy modifications, not system-library changes.

## Tests and native dependency evidence

- Core red: tests initially failed compilation because `RuntimeLocation` did not exist. Four Core tests then passed: relocation, explicit development opt-in, malformed-bundle refusal, and escaping path/symlink rejection.
- Packager red: tests initially failed because its module did not exist. Three tests then passed: existing output preservation, external-link rejection and internal-link acceptance.
- First real package install exposed uv's managed-interpreter marker. The correction changed only the copied interpreter, followed by a successful pinned install.
- Native audit exposed absolute libpython/OpenMP install identities. Corrected copies passed inspection and ad-hoc verification.
- All **39 Mach-O files** verified after ad-hoc signing. No non-system absolute dependency remained. Native load commands report minimum OS versions 11.0 or 14.0; **14.0 is the candidate floor**, not a tested compatibility claim. Intel hardware and older macOS versions were not tested.
- Payload size is **637,493,588 bytes (~608 MiB)**, excluding the native app shell. There are 17 installed distributions: the 16 pinned development-environment packages plus pip supplied with standalone Python. Production trimming of pytest/pip/tooling is not yet claimed.

See [native dependency inventory](native-dependencies.json). Package versions are pinned, but wheel/source archives are not yet hash-locked end to end; this is not a supply-chain attestation or reproducible-byte build claim.

## Licensing evidence and remaining review

The payload includes wheel license/notice files and CPython's license. [Wheel inventory](wheel-license-inventory.json) records installed versions, upstream declarations, classifiers and copied notice paths. Its `reviewRequired` field is only a mechanical missing-metadata/notice flag; false does **not** mean redistribution approval. All 17 distributions had a declaration/classifier and at least one notice file. Larger bundles include 100 PyTorch notices, 17 NumPy notices and 15 setuptools notices.

Notable declarations from installed metadata: PyTorch combines Apache-2.0 (including LLVM exception), BSD-2/3-Clause, BSL-1.0 and MIT; NumPy combines BSD-3-Clause, 0BSD, MIT, Zlib and CC0-1.0; typing-extensions declares PSF-2.0; packaging declares Apache-2.0 OR BSD-2-Clause. Other declarations are chiefly MIT/BSD families. These are inventory facts, not a legal interpretation of combined redistribution obligations.

The uv interpreter comes from **Astral python-build-standalone**, not a python.org macOS framework installer. Astral documents the distribution's licensing metadata and license texts in its full archives. The uv install-only directory does not include a full PYTHON.json dependency-license manifest. The exact full 20250814 arm64/Python 3.13.6 archive was fetched from the official release. Its SHA-256 matches the GitHub release asset digest: `f169b076da928b34ceff3578438fe68b65ef8009cae64fbc7150515a0e61d007`. [Standalone inventory](standalone-license-inventory.json) records per-extension declarations, link metadata and SHA-256 values for the extracted referenced notices. The archive declares Python-2.0/CNRI-Python plus Bzip2, MIT, X11, BSD, Apache/OpenSSL, TCL and Zlib families, and labels SQLite/liblzma dependencies public domain. These are upstream declarations, not redistribution clearance.

A concrete discrepancy remains: the zlib extension metadata references `licenses/LICENSE.zlib-ng.txt`, absent from this exact archive; `LICENSE.zlib.txt` is present and zlib's link metadata says system library `z`. This needs upstream clarification/redistribution review; it is not silently treated as satisfied. The extracted notices and full PYTHON.json are retained locally under `.local/runtime-feasibility/standalone-license-evidence`. Later payload5 inspection confirms the extracted metadata and available notices are included under `standalone-license-evidence`, while `redistributionReviewComplete` remains false. The missing zlib-ng notice still requires reconciliation before release. The reproducible extractor `scripts/runtime_standalone_licenses.py` verifies a supplied archive digest, rejects escaping notice paths and lists missing references. Its two tests were first observed failing for the absent implementation, then all five packaging tests passed.

[Exact upstream archive](https://github.com/astral-sh/python-build-standalone/releases/download/20250814/cpython-3.13.6%2B20250814-aarch64-apple-darwin-pgo%2Blto-full.tar.zst). [uv Python distribution policy](https://docs.astral.sh/uv/concepts/python-versions/), [standalone licensing documentation](https://github.com/astral-sh/python-build-standalone/blob/main/docs/running.rst), [full distribution metadata](https://github.com/astral-sh/python-build-standalone/blob/main/docs/distributions.rst).

## Distribution and sandbox constraints

Apple documents that embedded command-line tools inherit the containing app's sandbox; the child uses app-sandbox and inherit entitlements. User-selected dynamic file permissions are not automatically equivalent to inherited static entitlements. Root's private, content-hashed source snapshot approach is therefore tested separately from bookmark restoration. [App Sandbox documentation](https://developer.apple.com/documentation/security/protecting-user-data-with-app-sandbox), [sandbox inheritance](https://developer.apple.com/library/archive/documentation/Miscellaneous/Reference/EntitlementKeyReference/Chapters/EnablingAppSandbox.html).

Ad-hoc signature verification establishes local code integrity for this probe; it does not establish Developer ID/App Store signing, notarization, hardened-runtime compatibility or App Review acceptance. A curated bundled runtime avoids runtime package installation, but executable plugin policy and downloaded-code behavior still require a concrete distribution decision under Apple's current review requirements. [App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/).

Outstanding step-04 acceptance: root-owned sandboxed native worker/source/save/cancel/quit/reopen UAT; clean-account/no-developer-tools installation; persistent source-access denial/recovery; exact interpreter native-license reconciliation; actual distribution signing/review path; hardened runtime and supported-hardware matrix. None is silently marked complete by the successful relocated worker probe.

## Packaged worker checksums

See [worker integrity and migration](WORKER_INTEGRITY.md) for the canonical inventory generation, fail-closed bootstrap checks, migration command and remaining trust boundaries. Fresh payloads include the complete runner resource inventory; direct development execution remains explicit and separate.

### Fixed database and API broker entry points

The bundled launcher is `launch.py` at the runtime root; the fixed database capture,
database output and API service programs live in its `runner/` directory. The helper
resolves them through `RuntimeLocation.brokerEntry`, which also supports the explicitly
selected development runner directory. Missing scripts, symbolic links and escaping
paths refuse; a packaged helper never falls back to a developer checkout or an adjacent
lookalike file. The signed bundle and sealed worker inventory remain authoritative.
