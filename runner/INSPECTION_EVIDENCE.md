# Completed dataflow inspection — 2026-10-05

`--inspect-dataflow MANIFEST --dataflow-manifest-sha256 SHA` is read-only. The only additional accepted option is the packaged launcher's `--registry`. This command never imports archived programs, executes graph operators, reads original CSV sources, allocates model tensors, or rewrites artifacts.

Events are `dataflowInspectionStarted`, the existing `dataflowPreview` shape, then `dataflowInspectionCompleted`. Completion contains the pinned manifest SHA, graph digest, artifact names, original cache report, and `provenance` with recorded registry digest / NumPy version plus `historical:true`, `sourceRevalidated:false`, `cacheReuseVerified:false`. Historical off-policy previews have a null cache key because off-policy manifests do not retain those keys. Their status/reason are executed/disabled. Their output previews match original execution.

`completed_inventory` in dataflow_storage is shared with prepared-data loading. It verifies the complete inventory before previews: regular-file/no terminal symlink reads, 2 MB manifest, 300 files, 50 MB per file and 200 MB aggregate, exact SHA and byte counts, no traversal/duplicate names. Existing validated table/state readers interpret data only. The current supported registry must match the archived registry exactly; unsupported historical registries fail explicitly. Recorded NumPy and implementation digests are preserved without requiring equality to the current runtime. This permits historical viewing without claiming current cache eligibility. Changes to the shared storage implementation intentionally invalidate prior execution cache reuse identities.

Every expected graph output must be present exactly once and have its declared table/state type; fitted states must match the graph and producing operator. Table data version, row identities, membership, column types and bounds are checked by the existing cache table reader. All previews are staged in memory before any are emitted. Individual JSON events stay under 900 KB; cumulative preview JSON is capped at 32 MB. Inspection rejects unsupported schema versions, malformed cache reports and excessive retained cells. An integrity pin authenticates caller-selected bytes, not independent proof that an untrusted producer computed them honestly.

Cancellation propagates through the normal worker signal handling. It cannot emit completion and has no cleanup ownership over the historical output directory. Native callers must discard partial previews when the process fails or is cancelled.

Evidence: the initial focused suite produced five expected failures (missing API/CLI), then implementation passed. Final targeted regression command:

```
.venv/bin/python -m pytest runner/tests/test_dataflow_inspection.py runner/tests/test_prepared_data.py runner/tests/test_prepared_preview.py runner/tests/test_dataflow_cli.py runner/tests/test_dataflow_cache.py -q
```

Result: **57 passed in 11.75 seconds**. Coverage includes off/materialize/reuse output equality; real CLI success and pin/options failure; all-file tampering including exported code; symlink rejection; repinned malformed manifests/tables/states; missing ports; historical runtime preservation; cumulative preview rejection before delivery; archived Python treated only as bytes; cancellation preserving all original file hashes; and existing prepared training/inference, cache and standalone export replay regressions. Native restart/UAT remains a separate integration gate.
