# AvgPool1d scoped implementation

Status: source integration and scoped automated acceptance complete; packaged native execution and History verification pass, with remaining acceptance below. No production gate closure claimed.

Contract: batched [N,C,L], C1–512/L1–4096, kernel and explicit stride1–4096, nullable stride follows kernel, padding0–floor(kernel/2), Boolean ceil_mode/count_include_pad. No dilation, indices, learned state or train/eval difference. Reuses reviewed dilation-one window geometry; averaging semantics and divisor tests are independent.

Reference: [PyTorch AvgPool1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AvgPool1d.html). Runtime verification uses the installed local PyTorch, not a documentation-version compatibility claim.

## Implemented source integration

The existing pooling bundle now also owns AvgPool1d's five controls, teal pooling icon identity, purpose, ports, strict runtime descriptor, worked learning content and prepared example factory. Generator branches explicitly allow the reviewed average-pooling contract; no arbitrary adapter imports were introduced. Public registry contains 29 primitive components after this addition (the larger canvas also contains block, data and control families).

Native configuration reuses MaxPool1d window geometry with fixed dilation one, removes index output, and adds the independent include-padding divisor switch. Graph validation and migration understand the type. Worker registration, strict semantic normalization, generated discovery, actual functional execution, single-output capture and standalone source export are connected. Existing bundle-driven app discovery, Learn entry, controls and availability routing are reused.

Example: Linear2→14, reshape[2,7], AvgPool1d(window3,stride2,padding1,ceiling), ReLU, flatten8, Linear8→2. Total60 parameters. Synthetic feature positions are disclosed; this does not treat arbitrary tabular columns as a natural temporal axis. Learning is an explanatory worked example, not scored mastery.

## Verification record · 2026-10-07

- `.local/avgpool1d-red.log`: collection failed for missing new configuration module before implementation.
- `.local/avgpool1d-graph-red.log`: five graph tests failed because the public registry did not yet contain the operator, before bundle generation.
- The first independent window oracle over-admitted ceiling windows when kernel exceeded input plus allowed rounding. It was corrected to enumerate only windows whose end fits the configured padded domain plus at most stride−1 extra positions; this was a test-oracle repair, not a claimed product defect.
- `.local/avgpool1d-integrated.log`: 37 Python tests passed, no skips. Includes exhaustive small geometry/value/gradient windows, strict invalid settings, noncontiguous values, standalone source forward and backward parity, empty-batch export rejection, actual three-epoch CPU checkpoint continuation with exact model/optimizer/RNG equality, activation-cache reuse, both divisor modes on real MPS, exact worked learning arithmetic and bundle generation/contract checks.
- `.local/avgpool1d-pooling-regressions.log`: 32 existing MaxPool1d/2d/3d graph tests passed, no skips after the shared engine changes.
- `.local/avgpool1d-swift.log`: 11 Core tests passed (2 new AvgPool1d, 9 existing max-pooling graph regressions), zero App tests selected. Native tests were added after implementation and are not claimed red-first.
- Root-coordinated `.local/flow-revisions-avgpool-green.log`: 22 Core tests including FLOW and 2 AvgPool1d; 4 App tests including actual worker discovery and real prepared three-epoch code/training/verified History. All passed without skips.
- `.local/avgpool1d-discovery-regressions.log`: 6 Core and 10 App tests passed without skips, including exact capability rejection, stale support reset, actual max1d/max2d/average worker probes and repeated real AvgPool1d prepared execution/History. This verifies source integration, not packaged visual UAT.

## Remaining acceptance

Packaged native architecture-source export, reversible divisor editing and reopen pass. Portable trained-weight package replay remains root-owned. Broad TORCH-01 closure and production coverage percentages are not claimed here. AvgPool2d/3d, adaptive pooling, unpooling and later inventory families remain undelivered by this slice.

## Packaged native evidence · candidate 4a1096b

Opened Average windows and gradients from Learn and inspected window 3, explicit stride 2, padding 1, ceil mode and include-padding controls, plus the worked divisor/gradient explanation and synthetic-axis caveat. Universal Run completed operation `BEF630C0-5EE5-4142-A988-C486525E6DDC`. Bottom Data Results showed captured revision 0 and Local averages output `[8,2,4] float32`; the first values were −0.477, −0.845, 0.157, 0.380, 0.448, 0.366, −0.199, −0.084. History verification accepted 30 artifacts, six computed preparation nodes, three epochs and validation loss 0.49009. The displayed 100% accuracy is a small synthetic example result, not general model quality. Current sources were explicitly not revalidated by historical verification. A full app quit/relaunch returned to the project launcher; Resume restored the AvgPool canvas, its selected Local averages component and configured stride/ceil settings. History listed and reverified the same 30-artifact receipt after restart. Changed controls and exported package replay remain separate checks.

Native Code → Save Python source exported `.local/avgpool1d-native-20261007.py` after reopening. Executing that exact file with fixed seed 17 and float64 inputs produced `[8,2]`; independent explicit three-position sums (including zero-padding divisors) matched forward and input/parameter gradients with maximum absolute error 0.0. Empty batches were rejected. Evidence: `.local/avgpool-native-export-verification.json`. This verifies the native source export, not a portable trained-weight package. The native include-padding checkbox changed generated code to `count_include_pad=False`; Undo restored True and Save retained the restored sample.

## Exact native trained-package replay · 2026-10-07

The native run `BEF630C0-5EE5-4142-A988-C486525E6DDC` is retained under project `A1D15F7B-46C9-421D-A046-B22659B80FD5`, History directory `FD2DBAC9-AF21-463F-9630-32E87BCED2CB`. Its run archive initially contains generated architecture source and a trained checkpoint, rather than an inference-package folder. The subsequent native History → Saved model source → Export trained model flow created the actual package `.local/KwipML-model-BEF630C0`, including `inference.py`, `model.py`, `checkpoint.pt`, input contract, README, requirements and export inventory.

That exact package was copied to `.local/native-avgpool-portable-20261007/native-export-package`. All inventory sizes/hashes passed; copied source/checkpoint hashes match the native run. The supplied `inference.py` ran unmodified with the real standalone Torch 2.14.1 interpreter using `-I -B` and the retained prepared feature rows. Sandbox rules denied the original workspace, original export folder, checkout runner/Sources, development environment, runtime runner modules and network. A read probe confirmed the original export was inaccessible. No original files were changed.

The process exited zero and reproduced all **16 of 16 saved class predictions**. Its eight retained output-logit values matched with maximum absolute difference **0.0**. CPU validation loss was **0.4900863766670227**, exactly matching the retained native MPS run; accuracy was **1.0** for this synthetic example. The input contract remains already-prepared `[N,2] float32`, ordered `signal`, `context`; raw-source preprocessing is excluded. A separate preliminary test-owned harness also restored the unmodified architecture/checkpoint on CPU and MPS with the same saved classes, loss and eight retained logits. The supplied package's inference contract is CPU.

Evidence: `.local/native-avgpool-portable-20261007/report.json`, `package-run.log`, `package-stdout.json`, sandbox profile and `verify-package.py`. Export-manifest SHA256: `e2713640a7aaa2360dfb1203e48d852af1301834bbc154257356ea53056064c1`; inference SHA256: `ceec96cc996318d43ca950e39f0237a3f235e8ed5e59458cbada0120a3ef3b25`; checkpoint SHA256: `b0714d5b3c4a211d5f1821a78b426b69f9e27de5f7ea8e219878f9a8b40bf98e`. This closes the specific trained-package replay check previously listed above; it does not establish broader TORCH-01 coverage, resume acceptance, or production distribution readiness.
