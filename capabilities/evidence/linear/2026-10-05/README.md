# Partial Linear numerical evidence

This folder contains one real automated observation covering 32 numerical forward/backward cases, not a completed capability or native UAT acceptance.

- `numerical-report.json` is a byte-for-byte copy of the retained local report, including its original dirty-checkout annotation and subsequent production-content resolution. No numerical tests were rerun to assemble this folder.
- `execution-contract.json` was captured from the actual current worker's `--capabilities` response after checking its runtime/device identity and exact relevant production source hashes against the retained report and commit `2d05d89ff0fe81070624f567f471c072ac7cf472`.
- `manifest.json` pins production sources and acceptance scope to that commit; it separately pins the report's test code to `946a54d935e3e3806e78b9f8d8319f7ee827a545`, verifying the test-source hash retained in the original report.

The report's original `sourceRevision` is checkout HEAD at observation, not a claim that this clean commit executed. `observationProvenance` identifies the dirty state, exact production hashes and resolved content commit. The final environment matches the retained report. All paths inside the artifacts are repository-relative; no datasets, provider state, private filesystem paths or binaries are included.

The observation's training dimension means numerical backward differentiation, not optimizer/state lifecycle acceptance. The separate 40-test pytest result is intentionally not claimed as an observation here because no exact retained log was bound. Native candidate acceptance, curriculum, diagnostics, lifecycle, broad domain coverage and 2.14 documentation versus 2.14.1 patch alignment remain open requirements.

Verification performed:

```sh
python3 -B scripts/verify_partial_evidence.py capabilities/evidence/linear/2026-10-05/manifest.json
```

Exit 0: `integrityValid:true`, `accepted:false`, `coverageCredit:0`; one passed observation, zero failed/skipped observations. This validates reference integrity, not the truth or completeness of the observation. It does not update the capability catalog or establish a coverage percentage.
