# SiLU, LeakyReLU and ELU acceptance · 2026-10-08

Frozen finite acceptance for `pytorch.torch.nn.SiLU`, `pytorch.torch.nn.LeakyReLU` and `pytorch.torch.nn.ELU`. Each owns **forward**, **input-gradients** and **shape-contract** modes. These operations have no learned parameters; surrounding trainable layers supply checkpoint, continuation and exported-classifier evidence. No broader framework or release acceptance is inferred.

## Reviewed configuration and execution boundaries

| Capability | Component | Configuration | Assessed lesson | Prepared example |
|---|---|---|---|---|
| SiLU | `silu` | No mathematical coefficient | `assessment.silu-values` | `prepared-silu-classifier` |
| LeakyReLU | `leaky_relu` | `negative_slope`, default 0.01 | `assessment.leaky-relu-slope` | `prepared-leaky_relu-classifier` |
| ELU | `elu` | `alpha`, default 1 | `assessment.elu-alpha` | `prepared-elu-classifier` |

Production models use schema 2, dense float32 CPU/MPS tensors, ranks 2–8, with a separate batch axis and one to seven sample axes. Each value is transformed independently; shape, order and batch membership remain intact. Parameters are signed finite float32 coefficients, including zero, bounded by ±3.4028234663852886e38. Missing coefficients normalize to declared defaults. Booleans, null, strings, nonfinite/overflow values, unknown fields, unsupported schema/dtype/layout and invalid shapes refuse. A valid extreme coefficient can still produce a nonfinite output; capture validation refuses that result rather than fabricating stability.

V1 deliberately uses **out-of-place** execution. Explicit `inplace` configuration refuses; no mutation optimization or memory-saving equivalence is claimed. Branch tests prove upstream tensors remain unchanged. The declarative activation bundle owns exact definitions, field templates, ports, runtime descriptors, generated bridges, icon identity and example factories. Optional worker descriptors are verified separately from static authoring availability; absence or malformed support blocks a selected flow before source capture and again at queued dispatch. Disconnected unused components do not block the selected model. Runtime refresh clears stale optional support and the verified component count excludes unsupported members.

Independent scalar expectations use a stable logistic factor times the input for SiLU, a positive/negative piecewise slope for LeakyReLU, and `alpha * expm1(x)` on ELU’s nonpositive branch. Input influence at exact zero is 0.5 for SiLU, the configured negative slope for LeakyReLU and the configured Alpha for ELU. Negative and zero coefficients are tested rather than restricted to conventional positive settings. Learned weights are not invented for these parameterless operations.

## Numerical, lifecycle and export evidence

Actual missing-helper RED preceded production implementation. The final actual-registry activation replay passed **397 checks with no skips or warnings**: 263 new cases and 134 existing activation controls. It covers real CPU/MPS, ranks 2–8, batch sizes 1/3, noncontiguous inputs, signed/default/extreme coefficients, independently calculated outputs and input derivatives, surrounding affine parameter gradients, generated export, stable saturation, branch preservation, allocation/capture refusal, exact CPU continuation, cache reuse and fine-tuning. MPS classifier training and preview execute for every member; MPS checkpoint guarantees remain weights-only. Source and actual identities are in `TORCH_CONFIGURABLE_ACTIVATION_NUMERICAL_20261008.md`.

The broader replay of **53 bound Python files passed 3,001 checks with zero skips**. Three existing warnings describe convolution same-padding copies, permitted embedding-renormalization CPU fallback on MPS and construction of an intentionally refused prototype nested tensor. They do not establish new activation fallback or unsupported activation acceptance. Log: `.local/configurable-activation-prior-python-replay.log`.

## Approachable independently assessed lessons

The three matching no-code lessons define arrays, observations, activations, slope, Alpha and measured influence before using those terms. Learners connect Input → requested activation → Output, configure the coefficient when one exists, then inspect measured output and influence. No code or handwritten derivatives are required. Optional deeper reasoning stays separate.

Guided and private fixtures include signed values, exact zeros and independent observations. Targets are empty SiLU configuration, LeakyReLU slope 0.2 and ELU Alpha 2. A wrong valid coefficient measures but fails independent grading; wrong component identity, unsupported topology/layout, in-place configuration, nonfinite evidence and resource violations refuse. Private evidence cannot be presented as guided feedback. Versioned task, submission, draft and receipt identities preserve prior lessons and cold replay.

Lesson arithmetic is float64 for independent assessment; production model execution/export remains float32. This is explicitly not a claim that production supports float64. Python lesson replay passed **67 checks with no skips**, including 15 new and 52 prior pooling/convolution learning checks. All three actual guided/private worker, saved replay and cross-mode lifecycle checks passed through the App store.

## App integration and actual native observations

Final Swift regression executed **160 tests: 159 passed, one existing opt-in validation-timing benchmark skipped, zero failures** (79 Core, 81 App). An earlier broad run failed two obsolete registry-count expectations; updated metadata checks and the final regression passed without weakening per-component assertions. Logs distinguish that failure from the successful final run. The shared admission fixture covers **156 scenarios**: 13 operations × training/experiments × six admission/cancellation/revocation boundaries. A real missing queue-admission gate and missing searchable lab produced semantic RED before their fixes. Failed-gradient inventory validation also refused the three parameterless types before the reviewed inventory was extended; an unknown-type refusal remains tested.

Each member has one searchable assessed lab entry and one authored example/factory. Registry grouping supplies the accepted amber circular activation family. All three icons were genuinely generated with image_gen, preserved without pixel edits, and shared by palette/canvas. Original generation prompts, sources and hashes are in `docs/design/component-icons-v2/CONFIGURABLE_ACTIVATION_RECEIPT_20261008.md`. Native Light palette inspection showed clean transparency and distinct curves beside existing activation artwork. Final release theme/accessibility matrices remain separate.

Representative native lesson project **467293EB-33A1-43E7-9F38-C04BA0B63B39** used real Add actions and explicit port connections. LeakyReLU’s default slope 0.01 failed the requested independent criteria. Editing/applying slope 0.2 passed both criteria; private Assessment also passed. A full quit/relaunch restored slope 0.2 and verified the saved assessment against its task, submission and measured evidence. Native Settings verified Python/PyTorch support with CPU/MPS and 33 available components after discovery. Initial unavailability during discovery was transient. Native ELU’s Alpha field accepted signed −0.5 and generated `functional.elu(..., alpha=-0.5, inplace=False)`.

The canonical packaged worker includes both new activation modules and passed integrity verification, local signing and strict package verification. Executable SHA256: **b1b04cf2394fd88be9b9701e2239d04828162d4d2986d61645ac8c7f6dfc6091**. No distribution signing or App Store submission is claimed.

## Real prepared-flow and portable replay

Native project **BCCB2CE1-0C14-4F7C-AF0C-CE488C8342F7** trained its 15-component prepared flow for three epochs with **LeakyReLU negative slope 0.2** on actual **MPS**. Captured revision **1**, run **0E046FBC-B0FC-484D-8CCC-1DF55F28E932**, request **0d74499941e9828113aeee7d23e89efd6e40778f2b16c8f171b6062858f159a6**. History verified the archive: 30 artifacts, six preparation stages computed, zero reused. Validation loss 0.24322 and 100% on 16 bundled teaching observations do not establish generalization or model health.

Native Saved model source retained `negative_slope=0.2, inplace=False`. The reviewed export was saved to `.local/KwipML-model-0E046FBC` with code, checkpoint, strict inference helper, prepared-input contract and hashes. Prepared float32 inputs have shape [N, 2] and column order signal → context. Source rows and fitted preprocessing are excluded; callers must reproduce the captured preparation before inference.

Independent verification loaded the actual saved MPS checkpoint on CPU and used ordinary scalar loops over its learned affine weights and the slope 0.2 branch. Across all 16 retained observations, maximum absolute error was **9.978827675283242e-8** against scalar predictions and **1.1920928955078125e-7** against captured output samples. Captured feature-to-activation error was **2.9802322554228766e-9**. Strict checkpoint loading into exported Model matched the inference helper exactly (**0** maximum error). Batch size 1 versus 16 differed by at most **1.1920928955078125e-7**, and all class indices matched captured predictions. The portable `--example --batch-size 1` command completed successfully.

Exported model SHA256: **62c064747676f3d8eddafbba697b9cf0dda664dea42bd6986225d4b46f3065ad**. Checkpoint SHA256: **d54f786cfd8bb63c7bf6682d6534df759c71839017c65e905c639524f09a5012**. Export manifest SHA256: **16d8b6b60da954e7e19427967e31b3140cbe09192d76ad0721934f5d08ea63b0**. Ignored binary evidence and native screenshots remain local; committed contracts, test sources and this frozen receipt bind the accepted scope.

A separate read-only agent reviewed production configuration/execution/export, independent tests and lesson privacy/lifecycle seams and reported no remaining blocker in these reviewed boundaries. The review does not replace the actual tests or native observations. Each member earns credit only when all ten evidence dimensions and all three modes are independently bound in the catalog. Future component families, DATA-02, remaining curriculum, consolidated UAT/DIST and final owner release acceptance remain open.
