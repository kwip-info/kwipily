# Shared-kernel convolution lessons · 2026-10-08

## Finite delivered objective

Three independently assessed no-code tasks use the same existing tensor lesson shell:

| Task / assessment suffix | Component | Public sample shape | Shared kernel shape | Output sample shape |
| --- | --- | --- | --- | --- |
| `conv1d-shared-kernel` | Conv1d | `[2,4]` | `[2,2,2]` | `[2,3]` |
| `conv2d-shared-kernel` | Conv2d | `[2,4,4]` | `[2,2,2,2]` | `[2,3,3]` |
| `conv3d-shared-kernel` | Conv3d | `[2,4,4,4]` | `[2,2,2,2,2]` | `[2,3,3,3]` |

Assessment IDs prefix these suffixes with `assessment.`; content and assessment versions are 1.0.0. Each requests the matching component capability version 1.0.0. The leading observation direction is separate from the public sample shapes. Four seeded fixtures vary values and observation counts 1/2 while retaining the public channel/spatial/kernel contract.

Learners connect Input → the requested convolution → Output and configure two input/output channels, kernel size 2 in every spatial direction, stride/dilation 1, groups 1, enabled bias and zero padding amount 0. Explicit zero padding and Valid with amount 0 are equivalent. The task supplies input/kernel/bias/upstream measurement data; learners do not type weights or derivatives. The public objective explains channels, spatial directions, local windows, sharing, stride, dilation, padding and change sensitivity at a high-school level. Deeper interpretation remains in the existing framed optional reasoning section.

The graded criteria are output, inputGradient, kernelGradient and biasGradient, each with absolute/relative tolerance 1e-6. The independent Core oracle uses scalar multiply/add indexing for cross-correlation and sums influence across overlapping/shared uses; it does not call PyTorch. The executor measures real `torch.nn.functional.conv*d` and autograd using supplied state. Assessment arithmetic is float64; it does not extend the production float32 device contract. No claim of general convolution mastery, arbitrary graph equivalence or broader release acceptance follows from these tasks.

## Integration and privacy

`ConvolutionLearningTask.swift` owns task and assessment generation; `ConvolutionLessonReference.swift` owns the independent oracle; `ConvolutionLearningSubmission.swift` owns the graph-only schema-1 `convolutionGraph` envelope. Shared task/canvas/draft/submission/receipt dispatch handles the new tasks without changing earlier identities. Production configurations and shape resolvers remain authoritative. The existing observation-preserving lesson path permits one requested convolution plus reviewed layout transforms; disconnected graphs, wrong operations and missing state refuse. Valid names/layouts and preserving identity transforms do not determine grades.

App readiness checks the matching optional Conv1d/2d/3d runtime descriptor. Guided and Practice can show measured arrays, supplied state and influence tables. Current Assessment mode and assessment-origin receipts never expose fixture arrays through formative projections. Saved receipts bind project, canonical submission, runtime/version, seed and fixture digest and recompute the grade. Changing mode intentionally clears the current displayed result; reopening in another mode reports that an attempt exists in another mode instead of treating it as that mode’s result.

`TensorMeasuredOutput.kernelGradient` is explicitly a shared parameter tensor, so it appears in the generic Results selector without observation slicing. Kernel/bias influences always execute; no hidden checkbox or new parameter editor is needed for this fixture-supplied lesson.

## Actual verification and corrections

- Root captured actual missing task/reference/submission API Swift RED in `.local/convolution-learning-integration-red.log`.
- Python missing-module RED is `.local/convolution-learning-python-red.log`.
- A focused additional missing `kernelGradient` case RED was captured in `.local/convolution-integration-red.log` before the results enum addition.
- Integrated discovery RED `.local/convolution-discovery-red.log` recorded missing assessed/example discovery. It also exposed six incorrect test expectations after a mode switch: the existing store intentionally clears its current result and only restores receipts in the same mode. The test was corrected to verify actual private-mode reopen first, then Guided-mode non-disclosure and the “another learning mode” status. No lifecycle or privacy behavior was relaxed.
- A subsequent existing audience regression found that the direct task-text projection omitted the established **Configure** heading and the word “automatically.” The required view now uses **Build**, **Configure** and **Check** sections and explicitly states automatic supplied weights and measured influence checks. It also tells learners to select Conv1d/2d/3d before editing and spaces Kernel size 2, Stride and Dilation 1, and Groups 1 clearly. An initial partial log diagnosis incorrectly identified the missing Select action as the failed assertion; Select remains a useful copy improvement, while the actual heading/automatic-check failures were fixed without weakening tests. Grading identities, fixtures and rules did not change.
- Final scoped Swift GREEN `.local/convolution-integration-green.log`: Core 4 and App 11 tests pass, including all three actual guided/private worker runs and saved replay, public/private boundaries, discovery and six shared admission cases. Root owns Swift builds and native verification.
- `.local/convolution-learning-python-green.log`: 40 tests pass in the agent batch, comprising 18 new convolution tests plus existing learning-evidence and LazyLinear lesson regression tests. Root independently reran the 18 new tests successfully.

Independent Python expectations cover all-ones output sums, kernel/bias influence, overlapping input multiplicity, signed/distinct channel values, all three dimensions and equivalent valid padding/identity reshape. Refusal controls include wrong operator, disconnected graph, absent supplied kernel, nonfinite data and incompatible stride/output shape. Actual policy dispatch and budget refusal execute before tensor allocation; a malicious huge padding configuration is rejected by allocation-free production shape validation. Cancellation before execution leaves fixtures empty and does not draw random values.

Source files and logs are scoped evidence, not shipped private expected tables. No environment downloads, arbitrary submitted code execution or production classifier-boundary changes were needed. Root separately owns bundle authority, native acceptance, artwork provenance and final accounting.
