# Transposed convolution assessed learning · 2026-10-08

## Finite objective and identities

The shared convolution lesson path now supports three additional tasks:

| Task suffix | Component | Public sample shape | Kernel shape | Output sample shape |
| --- | --- | --- | --- | --- |
| `conv-transpose1d-shared-kernel` | ConvTranspose1d | `[2,3]` | `[2,2,3]` | `[2,8]` |
| `conv-transpose2d-shared-kernel` | ConvTranspose2d | `[2,3,3]` | `[2,2,3,3]` | `[2,8,8]` |
| `conv-transpose3d-shared-kernel` | ConvTranspose3d | `[2,3,3,3]` | `[2,2,3,3,3]` | `[2,8,8,8]` |

Assessment IDs are `assessment.` plus each task suffix, version 1.0.0; matching component requirements are version 1.0.0. Configure two input/output channels, kernel size 3, stride 2, dilation 1, groups 1, padding 0, output padding 1 and enabled bias in every spatial direction. Input→ConvTranspose*d→Output is the required measured operation path. Kernel and bias are supplied by the task; learners configure the graph without typing parameters or derivatives.

The objective is overlapping scatter windows, shared channel weights and output extent. Every input position spreads a weighted window into output positions; overlapping contributions add. Output padding chooses additional extent, not appended source values. In this bounded geometry, windows cover positions 0–6 and the extra position 7 has no window contribution but still contains channel bias. A transposed convolution does not recover the original input. Required Build/Configure/Check sections define channel, kernel/window, stride, dilation, groups and padding; optional reasoning retains the mathematical relationship.

Four seeded private fixtures vary signed values and observation counts 1/2 while retaining the declared public geometry. The largest individual output tensor has 2048 values. The total assessment inventory budget is 16384 elements with a 5000ms time budget; this does not enlarge the shared 4096-value limit for an individual teaching tensor.

## Independent measurement and reuse

`TransposedConvolutionLessonReference.swift` computes expected values through explicit scalar scatter-add indexing; it does not invoke a convolution library. It independently computes output and input/kernel/bias influence, including bias contributions on the extra extent. Actual execution uses `torch.nn.functional.conv_transpose*d` and autograd, supplied task state and strict runtime policy in the existing bounded executor. Exact shapes plus absolute/relative tolerances 1e-6 are retained. Float64 assessed arithmetic is separate from the reviewed float32 production contract.

The existing `ConvolutionLearningTask`, `convolutionGraph` envelope, graph validation/shape resolution, tensor lesson shell, runtime readiness, receipt verification and measured-table selector are reused. Output/kernel/input/bias criteria remain separate. Old standard Conv1d/2d/3d task IDs, raw instructions, versions, fixture generation and criterion feedback strings remain unchanged; their shape-specific tests retain their original three-task domain. One existing lifecycle test now iterates all six tasks instead of duplicating the entire store test body.

Receipts bind task version, project, canonical submission, runtime and regenerated fixtures. Guided/Practice measured arrays remain available only for matching current submissions and non-assessment-origin receipts. Switching mode clears the current result; private attempts remain private after switching to Guided or reopening in another mode. No new plugin engine, parameter editor or Results redesign was needed.

## Actual RED/GREEN and corrections

- Root missing-API Swift RED: `.local/transposed-convolution-learning-red.log` recorded absent reference and new enum identities.
- Python actual missing `execute_transposed` RED: `.local/transposed-convolution-learning-python-red.log`, 10 failures.
- First integrated Swift `.local/transposed-convolution-integration-first.log`: Core 6 tests passed. All three actual transposed guided/private workers and same-mode saved replay passed. App discovery remained intentionally missing. Three audience assertions incorrectly banned the phrase “reverse the original” even in the truthful negation; the assertion was corrected to verify the explicit limitation instead of rejecting its wording.
- Root broader `.local/transposed-convolution-final-swift.log` passed actual all-six lifecycle, metadata/discovery, audience and selected/queued admission checks, but the log is **not an all-green run**: obsolete task-title and activation-catalog inventory expectations failed after the new IDs were introduced.
- Root corrected those explicit inventories and reran narrowly: `.local/transposed-convolution-inventory-green.log` and `.local/transposed-convolution-sigmoid-inventory-green.log` passed. Root reported 105 distinct applicable Swift checks after reconciling the broader and narrow runs; final unique aggregation belongs to root, not to a sum of repeated runs in this document.
- Agent Python `.local/transposed-convolution-learning-python-green.log`: 57 passed, comprising 17 new transposed lesson tests, 18 standard-convolution lesson tests and 22 existing learning-evidence/LazyLinear regressions. Standard convolution numerical behavior remained green after executor reuse.
- Packaged candidate built successfully in root’s `.local/transposed-convolution-package.log`; native acceptance and final bundle/accounting receipts are root-owned and separate from this implementation evidence.

Independent Python checks include signed and all-ones scalar output/influence in all dimensions, repeated shared use, overlapping windows, extra boundary bias, and a distinct second observation that leaves the first observation unchanged while shared gradients sum correctly. Wrong operator, missing/disconnected operation, incompatible output padding, nonfinite supplied state and authored-result envelope fields refuse. Strict policy dispatch and low budget tests execute before tensor allocation; malicious enlarged output extent is rejected by allocation-free production shape resolution. Cancellation before work produces no successful fixture evidence.

This slice establishes three bounded shape/value/influence objectives, not general transposed convolution expertise, arbitrary graph equivalence, inverse reconstruction or full release readiness. Root separately owns native acceptance, production authored metadata, example discovery and final accounting.
