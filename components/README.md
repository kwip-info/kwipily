# Component authoring bundles

This directory owns reviewed declarative component families, presentation, configuration,
conditional ports, runtime contracts and linked learning/examples. Current accepted PyTorch
scope is 87/88 capabilities with LBFGS deferred; see [coverage](../docs/CAPABILITY_COVERAGE.md)
and the [tracker](../docs/production/TASK_TRACKER.md). A family folder is not a capability count.
`registry-base.json` retains definitions outside migrated bundle ownership.

## Change a bundle

1. Find the owning family `bundle.json` and its referenced native, worker, tests and evidence files.
2. Define bounded acceptance. Extend shared templates and adapters; manifests are declarative
   and do not authorize arbitrary dynamic code imports.
3. Add meaningful failing controls, implement actual behavior and update the complete bundle:
   configuration/ports, persistence, runtime/export, artwork, example and assessed lesson.
4. Generate derived resources, check drift and run affected authoring/runtime/Swift tests.

```sh
.venv/bin/python scripts/component_bundles.py
.venv/bin/python scripts/component_bundles.py --check
.venv/bin/python -m pytest scripts/tests -q
```

Do not edit `Sources/KwipMLCore/Resources/components.json` or `Bundled*.generated.swift` catalogs
directly. Swift and Python consume shared generated metadata; actual adapters and discovery must
agree with it. Descriptor publication alone cannot establish runtime availability.
`published: false` withholds unfinished entries from the relevant discovery/palette routes.

## Contracts and boundaries

Use [contributing](../CONTRIBUTING.md), the [product contract](../docs/CONTRACT.md), relevant
[architecture decisions](../docs/architecture/INDEX.md) and the installed development skill when
available. The [planned component inventory](../docs/components/README.md) includes future scope;
it is not the implemented coverage denominator. Central shared seams remain reviewed code, not
a general executable plugin system. Earlier pooling migration receipts describe their original
slice and do not limit the current validator to that historical family.
