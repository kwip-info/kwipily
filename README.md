# Kwipily

A native macOS visual workflow and machine-learning app, formerly Kwip ML.
Build data workflows, train supported models, inspect results and keep projects locally.
Free to use, with no KWIP account, activation or license key.

## Release status

The source is available now. The signed free Mac download is still undergoing
release validation; this repository does not announce a completed binary release.
The verified download link will be added here when the release is ready.

## Build on macOS

Use an Apple Silicon Mac and Xcode 27.2 / Swift 6.4.2 (the release toolchain),
and uv with Python 3.13. The packaged app targets macOS 14 or later; the package
manifest language floor is not a tested compiler compatibility matrix. Select your Xcode developer directory with DEVELOPER_DIR
if it is not the default. From this checkout, run:

```sh
./scripts/setup.sh
```

This installs Python dependencies and creates `.local/KwipML POC.app`, an ad-hoc
signed development app with a separate identity. Keep the checkout and `.venv`
in place while using it. Optional model downloads and provider integrations may
require their own credentials or licenses. No KWIP credentials are required.

## Tests

```sh
KWIPML_TEST_DEVELOPMENT_ROOT="$PWD" \
KWIPML_SKLEARN_TEST_PYTHON="$PWD/.venv/bin/python" swift test
.venv/bin/python -m pytest runner/tests
.venv/bin/python scripts/component_bundles.py --check
```

Some optional integration tests require separately acquired model fixtures; report any skips.

Frozen historical regression inputs are included in `Tests/historical`; no earlier
Git history is required. Production packaging scripts are included for transparency;
official signing, notarization and hosted release operations require publisher
credentials and private acceptance records. They are not prerequisites to build locally.

## License

Apache 2.0; see LICENSE and NOTICE. Third-party components retain their licenses
and notices. The license does not grant use of KWIP trademarks beyond its terms.

## Support

Email contact@kwip.info. Support is provided as time permits. Never include
credentials or private datasets in issues. The standalone product site will
launch alongside the signed free Mac release.

This is a source snapshot with fresh Git history. Private release receipts and
upstream website captures are excluded. Some publisher verification scripts refer
to those private inputs; the build and runtime tests above do not require them.
Historical technical notes included for component provenance describe their original
checks, not validation of the current release. Developer home paths are anonymized.
