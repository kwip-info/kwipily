# Source-built Sparkle boundary

Upstream: https://github.com/sparkle-project/Sparkle, version2.10.0, commit `eef1a539a373c1f1a320624b1130fc5de7b2e100`. `peer-boundary.patch` is the local KWIP modification; retained upstream `LICENSE` includes external-component notices. Include these notices with any future binary redistribution. No framework binary is tracked or embedded in the application yet.

Apply to a clean checkout of that exact commit (`git apply --check` before `git apply`). The recorded local build succeeded with:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
  -project Sparkle.xcodeproj -scheme Sparkle -configuration Release \
  -derivedDataPath ../Sparkle-KWIP-build ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
  MACOSX_DEPLOYMENT_TARGET=14.0 \
  DOWNLOADER_SANDBOXED_ENTITLEMENTS=Downloader/Downloader.entitlements \
  XPC_SERVICE_BUNDLE_ID_PREFIX=info.kwip.KwipML.Sparkle \
  SPARKLE_EMBED_INSTALLER_CONNECTION_XPC_SERVICE=1 \
  SPARKLE_EMBED_INSTALLER_STATUS_XPC_SERVICE=1 CODE_SIGNING_ALLOWED=NO build
```

The resulting downloader bundle ID ends in `DownloaderService`, not `Downloader`. Sign nested services using their actual entitlements and the selected Developer ID; sign the Autoupdate executable explicitly as `org.sparkle-project.Sparkle.Autoupdate` before the enclosing framework/app. No ad-hoc customer release is accepted by the patch. Do not grant network.client to the parent or inherited ML workers.

Patch purpose: exact Apple/team/parent checks for both installer bridge listeners; exact necessary peers for installer/status endpoints; validation throughout installation stages. Team `MT68X9Z6NQ` and production identity `info.kwip.KwipML` are intentional. Team-wide trust would let other same-team software approach privileged helpers. The inherited-worker boundary also depends on the OS sandbox, and patch causality must not be inferred solely from a denied worker connection.

Recorded local signed service probes accepted the legitimate parent and denied inherited workers and a distinct same-team parent across all three XPC services. Those probes do not establish actual update installation, helper-replacement resistance, rollback, or app integration. Repeatable release automation and update acceptance remain later work under the enterprise ML-DIRECT notes.

## Repeatable dependency build

Use `scripts/sparkle_build.py` with the explicit cached upstream checkout and a new `.local` output directory. It clones the exact commit, applies this patch, runs the source recipe above and creates an arm64 XCFramework with source/patch/notices/tree-digest correspondence. No Keychain use, dependency signing or app embedding occurs at this step. See `docs/production/DIST_SPARKLE_DEPENDENCY_20261011.md` for actual local evidence and remaining integration gates.

The patch also refuses downloader redirects and uses ephemeral cookie/credential/cache-free sessions. The signed actual XPC redirect probe first failed with the prior source, then passed after the patch. Run `scripts/sparkle_downloader_probe.py` only as explicit local signed verification; it uses a synthetic loopback origin and never customer credentials. Current source/linkage/evidence and remaining production integration: `docs/production/DIST_SPARKLE_INTEGRATION_20261011.md`.
