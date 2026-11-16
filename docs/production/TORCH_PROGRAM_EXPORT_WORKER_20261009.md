# Executable inference-program worker — scoped evidence

Date: 2026-10-09. Scope: new inference-program worker/configuration/retained-source adapter and one narrow existing tensor-composition capture seam. This is not native UI acceptance, publication, or admission of the three remaining capability modes.

## Contract

`runner/program_export_configuration.py` admits exactly seven controls at configuration version 1. Defaults: `strict=true`, `batch_mode=static`, `example_batch_size=2`, `minimum_batch_size=2`, `maximum_batch_size=2`, `device=cpu`, `prefer_deferred_runtime_asserts=false`. Missing/extra controls, bool-as-integer, unsupported values and inconsistent bounds refuse; defaults are explicitly selected rather than merged. Static requires identical bounds/example (including batch 1). Bounded permits minimum 1, requires minimum < maximum, an example at least 2 within the interval, example <=128 and maximum <=4096.

`export_program(model, example, input_contract, configuration, *, source_identity, check_cancelled)` returns actual `torch.export.save` bytes and manifest. `reload_program(bytes, manifest, *, expected_manifest_sha256)` verifies the exact manifest pin and program inventory before actual `torch.export.load`, independently verifies native tensor signature/state/ranges, and returns a callable with `.predict`. `build_package(..., destination, ...)` writes a new five-file package: `program.pt2`, `manifest.json`, `inference.py`, `README.md`, `requirements.txt`.

The manifest schema 1/kind `kwipmlExportedProgram` has exactly: schemaVersion, kind, runtime, configuration, input, output, sourceIdentity, shapeConstraints, rangeConstraints, state, files. Runtime records exact Torch version and selected CPU/MPS device. Input/output retain real dtype and fixed sample axes; output must be float32 and preserve the observation axis. A static leading dimension or one named bounded `torch.export.Dim` is supported. Native range constraints must equal the requested interval; unsupported shape specialization refuses rather than narrowing it. No automatic device fallback or version portability is claimed.

The source identity contains exact canonical runID plus requestIdentity/modelManifestSHA256/exportManifestSHA256. It preserves the actual retained trained source identity rather than substituting the new operation identity.

Limits: parameters/buffers/constants each at most 2M elements; each input/output at most 2M values at maximum batch; tensor rank 2..8 and sample axes 1..4096; graph <=4096 nodes; program <=100MB; manifest <=2MB; ZIP <=512 entries and <=100MB declared uncompressed bytes. Traversal, duplicate ZIP entries, encrypted entries and symlinks refuse. This is a trusted app-generated program contract, not a safe sandbox for hostile serialized models or Python code.

The cloned model is captured in evaluation mode. Original model state, mixed training modes, and CPU RNG remain unchanged; MPS RNG is restored when selected. Mutating state/inputs, unsupported output geometry, nonfinite state and untraceable data-dependent Python control flow refuse. Graph capture/save/load boundaries call cancellation; a callback cannot interrupt a native call already in progress. Root supervision retains hard cancellation, deadline, sampled RSS and CPU thread authority.

## Retained-source transport

`execute_export(request_path, expected_sha256, emit, *, check_cancelled)` accepts request schema 1, or schema 2 with the existing executionBudget. Exact common fields: schemaVersion, kind=`programExport`, projectID, exportID, requestIdentity, sourceDirectory, sourceManifestSHA256, configuration.

The request must live at `workflow-program-exports/<projectID>/<exportID>/request.json`; sourceDirectory is exactly the sibling `source`; new output is `package`. It reads private bounded regular files through anchored descriptors, checks manifest/inventory pins and rechecks retained directory identities and original bytes between capture boundaries. Only the verified source bytes are copied to a private temporary snapshot for the original pinned inference helper to load. A directory rename/replacement cannot continue against a stale open source descriptor.

Actual native trained input-contract schemas 1 and 2 are admitted. Schema 3 bag packets explicitly refuse with direction to the existing packet inference export. The CLI builds synthetic zero-valued tracing inputs, preserving int64 inputs where declared. Direct library callers supply their trace example and must account for its inclusion in the serialized program.

Standalone command:

```
python -m runner.program_export_cli --request <absolute/request.json> --request-sha256 <sha256>
```

Events are JSON objects with `type=programExportStarted` or `programExportCompleted`, projectID/exportID/requestIdentity. Completion additionally contains manifestSHA256. Standalone failure emits `programExportFailed` with a bounded error and exit 1. Success exits 0. This entry point does not grant one-shot managed-operation ownership; root integration must supply the native retained owner/claim and supervision.

## Necessary existing-source correction

A genuine generated branched trained model initially failed actual export: `_execute_tensor_composition` converted `torch.isfinite(...).all()` to a Python `.item()` condition, producing an untraceable data-dependent guard. Only this function in `runner/engine.py` changed. Eager execution preserves its existing finite-input/output ValueErrors. During capture, the same conditions become real `torch._assert_async` graph assertions. A zero-multiplied nonfinite source still refuses, and finite-source arithmetic overflow still refuses. MPS negative assertion checks synchronize before observing the asynchronous device error. Assertions are not removed to make capture pass. Other generated adapters with untraceable control flow may still produce an actionable export refusal; this receipt does not claim all application architectures are exportable.

## Actual TDD and verification

All commands used repository `.venv/bin/python`, Torch 2.14.1 on the available CPU and MPS devices. No Swift builds or native UI operations were performed by this worker.

| Evidence | Actual result |
| --- | --- |
| `.local/program-export-semantic-red.log` | 17 failed, 1.11s; buildable helper scaffolds reached intended NotImplementedError |
| `.local/program-export-first-green.log` | 17 passed, 3.23s |
| `.local/program-export-cli-semantic-red.log` | 9 failed, 0.90s; retained adapter scaffold |
| `.local/program-export-signature-red.log` | 2 failed / 18 passed, 2.86s; native signature forgery and portable inventory redirection exposed |
| `.local/program-export-signature-green.log` | 20 passed, 2.75s |
| `.local/program-export-composition-red.log` | 4 failed, 2.06s; actual CPU/MPS capture of existing finite guard exposed |
| `.local/program-export-integrated-green.log` | 2 failed / 33 passed, 3.65s; MPS negative test initially omitted device synchronization |
| `.local/program-export-integrated-green-2.log` | 35 passed, 3.52s; explicit native MPS error synchronization |
| `.local/program-export-directory-red.log` | 1 failed, 1.13s; stale source directory descriptor exposed |
| `.local/program-export-worker-final-green.log` | **36 passed, 3.91s** across three new suites |
| `.local/program-export-adjacent-green.log` | **227 passed**, four existing upstream KL warnings, 12.36s |

Final worker command:

```
.venv/bin/python -m pytest runner/tests/test_program_export.py runner/tests/test_program_export_cli.py runner/tests/test_program_export_composition.py -q
```

Adjacent command:

```
.venv/bin/python -m pytest runner/tests/test_tensor_composition_engine.py runner/tests/test_tensor_composition_configuration.py runner/tests/test_module_composition_export.py runner/tests/test_portable_inference.py runner/tests/test_objective_portable_inference.py -q
```

Independent expectations use literal affine coefficients, branched tanh arithmetic, literal embedding lookup and manual BatchNorm evaluation math. Tests execute real static and bounded native captures under strict true/false, actual minimum/interior/maximum calls and out-of-range native refusals, int64 input with true output axes, original state/modes/RNG preservation, CPU/MPS explicit devices, mutation/untraceable/cancellation/NaN/budget refusals, freshly re-pinned metadata forgery, archive path and byte tampering, and source pin/helper/model/symlink/directory substitution failures. A standalone Python `-I` process loads the saved five-file package outside the repository without the original model class. Actual retained app-trained binary branched and selected objective packages capture and reload learned results, with original source bytes unchanged.

## Remaining root-owned gates

The managed worker operation/authority route and existing supervisor integration are now verified below. Remaining native gates: stage source from independently verified native History, bind native output inventory and manifest pins, provide lifecycle action and lessons through the existing authoring/generation system, and run packaged native acceptance. Dataset and DataLoader remain separate modes with separate closure evidence. No catalog, coverage report, parent-step tracker or accepted prior receipt was mutated by this slice.


## Managed operation and independent discovery follow-through

The worker now admits `toolMode: programExport` under the existing schema-2 `workflowOperation` authority. It checks the complete staged trained-source inventory, inner request identity, configuration and budget against the retained outer operation before invoking native export. Source remains private under `workflow-operations/<project>/<operation>/source`; output is `package`. No alternate lock, process supervisor or storage namespace was added. An unmanaged main invocation is refused. The isolated helper namespace remains available for private helper tests. Program events use the shared managed `event` envelope after `operationReady`.

| Actual log | Result |
| --- | --- |
| `.local/program-export-operation-red.log` | 2 failed, 9 passed, 1.43s |
| `.local/program-export-operation-green.log` | 47 passed across the initial worker/operation union, 4.12s |
| `.local/program-export-managed-native-green.log` | 13 passed, 4.45s; genuine owned claim/bootstrap/supervisor capture and hard-deadline terminal cleanup |
| `.local/program-export-managed-final-green.log` | 189 passed, 6 existing gated skips, 27.51s |
| `.local/program-export-capability-red.log` | Independent sibling advertisement test failed before producer implementation |
| `.local/program-export-capability-green.log` | 1 passed |
| `.local/program-export-final-current-green.log` | 50 passed, 7.48s across all five new program suites |

`program_export_configuration.capability()` is an independent literal producer advertised as `programExportCapabilities.program_export`; it does not import the authored bundle or expected descriptor. These results establish worker capture/reload, operation supervision and discovery. They do not establish the new native History action, native transport cancellation/publication, lesson integration or final three-mode admission.
