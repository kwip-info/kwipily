# PyTorch consumer scope contract — 2026-10-08

Catalog version `0.2.0` records the owner's approved **88 consumer units**, covering all 99 canonical source identities from the unchanged `selection-v1.json`. The minimum is 80 evidence-complete units, target 84, with at least one accepted unit in every required family. This is not coverage of 90% of the entire PyTorch API.

`approved-consumer-scope` means the owner approved the selected consumer denominator. It does not approve per-operation runtime modes, backend boundaries, implementation evidence, native acceptance, or final deferred units. `boundary_review` and `deferrals` remain pending; both block the release gate. The numerator remains unaccepted until the owning acceptance work attaches current verified evidence. Source presence and POC references earn no complete-unit credit.

The backward-compatible schema-1 extension `source_members` declares each grouped canonical API and its compulsory `required_modes`. Primary capability IDs remain stable. Composition, datasets, checkpoint save/restore, requested/accumulated gradients and initialization consolidate eleven catalog entries, without deleting any source API. These are grouped consumer semantics, not aliases. Every grouped mode still requires every evidence dimension. Member symbols must map reciprocally in the retained inventory; duplicate source/mode ownership and missing member provenance are rejected. `consolidation.retired_catalog_ids` records the old catalog IDs; persisted component IDs are unchanged.

The raw official snapshots and all 9,066 inventoried records remain intact. Only reciprocal capability mappings and selected-source rationale change. Existing unsupported, excluded, infrastructure and alias accounting remains visible. Public APIs outside the selected list are out of the selected v1 scope, not covered.

`deferrals.units` contains named selected capability IDs and rationale. To pass release, its status must be `owner-approved`, with an approval reference, no more than eight deferred units, and an exact match to the incomplete-unit list. An empty required family still blocks release. This makes a high aggregate incapable of concealing an unaccepted family or unnamed deferral.

The offline migration accepts only the original proposed 0.1.0 catalog, preserves existing evidence without upgrading its status, and is idempotent on 0.2.0. The proposal generator now refuses an approved catalog before writing any file. Do not use inventory recapture or the proposal generator to overwrite this reviewed catalog; future changes require an explicit migration.

```sh
python3 capabilities/migrate_pytorch_consumer_scope.py
python3 -m unittest discover -s capabilities/tests -v
python3 scripts/coverage_report.py --output capabilities/reports/v1.json
python3 scripts/coverage_report.py --format markdown --output capabilities/reports/v1.md
python3 scripts/coverage_report.py --require-release
```

Verification: actual initial contract RED is `.local/consumer-scope-red.log`; final GREEN is `.local/consumer-scope-green.log`. The release command correctly exits 1. These are accounting tests, not framework numerical or native acceptance. A read-only comparison against the original Git catalog verifies migration reproduces the checked-in 88-unit catalog and preserves every inventory identity, disposition, documentation field and non-selected exclusion rationale. No framework imports, installs or network calls are involved.
