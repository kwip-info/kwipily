# ConvTranspose3d delivery

Status: scoped ConvTranspose3d delivery accepted locally, including packaged native execution/export/restart. Broader production gates remain open.

## Contract under test

Nine explicit settings: input/output channels, kernel size, stride, padding, output padding, dilation, groups and bias. Each spatial setting is an explicit `[depth, height, width]` triple. Groups divide both channel counts. Only zero boundaries are supported. Dynamic forward `output_size` is outside this static graph contract.

For each spatial axis, output length is `(input - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1`. Weights have shape `[input channels, output channels / groups, kernel depth, kernel height, kernel width]`. Output padding selects the resulting extent; it does not append zero-valued output data. Transposed convolution is not a general inverse of convolution. Reference: [PyTorch 2.14 ConvTranspose3d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose3d.html).

Portable validation requires output padding smaller than stride **or** dilation, independently on every axis. Valid cropping greater than the effective kernel remains permitted when the final shape is positive and within bounds.

Admission limits match the existing convolution family: channels/groups 1–512, spatial lengths/kernel/stride/dilation 1–4096, padding/output padding 0–4096, two million parameters, 100 million input/output/uncropped sample elements, and 128 million input-position × weight multiply-adds per observation. Uncropped extent is a conservative geometry limit, not a claim about actual peak allocator usage. Final-output admission precedes uncropped multiplication to bound intermediate arithmetic; Swift work admission uses division before multiplication; Python uses arbitrary-precision integers.

## Device feasibility — 2026-10-07

Pinned PyTorch 2.14.1, actual CPU and available Apple MPS: three valid asymmetric/grouped configurations completed forward and backward with finite outputs and input gradients. They include mixed strides, output padding admitted via dilation and large valid cropping. Observed shapes were `[2,4,7,6,18]`, `[2,4,5,6,17]` and `[2,4,4,7,8]`. Log: `.local/conv-transpose3d-probe.log`.

The invalid case with output padding `[2,0,0]`, stride `[2,2,2]` and dilation `[1,1,1]` failed on CPU but executed on MPS. Therefore backend acceptance cannot replace portable preflight. Nonzero boundary modes reflect/replicate/circular were refused by the constructor. These probes establish feasibility, not numerical parity, lifecycle or native acceptance.

## Configuration verification

Python RED failed collection because the new configuration module did not exist. Final GREEN: 441 tests passed, no skips, including configuration regressions for ordinary 1D/2D/3D and transposed 1D/2D; one expected ordinary-convolution same-padding warning. Logs: `.local/conv-transpose3d-config-red.log` and `.local/conv-transpose3d-config-green.log`.

The fixtures compare noncubic output shapes with actual CPU PyTorch and check strict triples, per-axis output-padding alternatives, exact two-million weights with bias crossing, exact 128-million work, exact 100-million input/output/uncropped volume and next-over refusals. Oversized inputs are already refused by shared layout construction; output and uncropped limits overlap, while cropped fixtures isolate the uncropped check.

Swift RED failed compilation on the missing configuration API. GREEN: 31 Core tests passed, no failures or skips, including eight new 3D transposed configuration tests and previous convolution families. Logs: `.local/conv-transpose3d-config-swift-red.log` and `.local/conv-transpose3d-config-swift-green.log`. Swift admits parameter count and bounded final output axes before larger products, and checks work through division before multiplication.

## Remaining acceptance

Broader accessibility/device/interaction matrices, graded curriculum and production/distribution acceptance remain open. The scoped native journey below is complete.

## Graph execution and export

Private nine-field component and schema-2/float32 shape validation now integrate the 3D configuration. Aggregate graph parameters are checked before module allocation; a transposed convolution alone can supply trainable state. Existing transposed-family dispatch maps now construct, execute and export `nn.ConvTranspose3d` with its exact tuple attributes and stable checkpoint keys. Runtime and generated code both require a nonempty batch with the authored sample shape. Public discovery and palette registration are now implemented; packaged native acceptance remains pending.

Swift graph RED: missing component API; GREEN: 30 Core tests passed, no failures/skips, including configuration and prior convolution graph regressions. Logs: `.local/conv-transpose3d-graph-swift-red.log`, `.local/conv-transpose3d-graph-swift-green.log`.

Python graph RED: 27 failed/6 passed before integration. Final focused GREEN: 217 passed, no skips, expected ordinary same-padding warning only. An independent nested 3D scatter-add reference checks float64 values and gradients for all inputs/parameters, generated code/state reload, grouped and asymmetric settings, dilation/output-padding alternatives, bias modes and noncontiguous inputs. Twelve real MPS cases compare outputs/all gradients against CPU. Logs: `.local/conv-transpose3d-graph-red.log`, `.local/conv-transpose3d-graph-green-final.log`.

The first GREEN attempt exposed a test-oracle issue: cropping removed every depth contribution in one valid geometry, leaving the reference weight unused by autograd. The oracle now retains an explicit zero weight dependency and asserts zero/bias-only output for that case. A separate cropped fixture retains nonzero contributions. No production change was needed for this finding; the original failure remains in `.local/conv-transpose3d-graph-green.log`.

## Checkpoint lifecycle

RED: four failures/nine passes identified the missing transposed-3D parameter ordering in checkpoint schedules. The fix adds this kind to the existing schedule activation and membership sets, preserving previous family signatures. Final combined verification: 324 Python tests passed with no skips; expected ordinary same-padding warning only. This includes prior convolution graphs/lifecycles, general checkpoint/frozen-state/preview regressions, exact CPU four-epoch versus two-plus-resume equality despite RNG-consuming callbacks, gradient parameter shapes, frozen/export parity, fine-tune reset, initialized/trained cache reuse and stale-configuration refusal, and actual MPS training/preview. Log: `.local/conv-transpose3d-runtime-lifecycle-green.log`. Log: `.local/conv-transpose3d-lifecycle-red.log`.

## Discovery and prepared example — verified locally

The worker advertises the exact versioned transposed-3D contract: four sample axes, five spatial triples, zero boundaries, portable output-padding rule, bounded resources and static output geometry. Native optional support must be cleared on stale/malformed discovery and must gate execution. The public component uses the reviewed nine fields and C/D/H/W connectors. No graded lesson is claimed.

Python discovery RED: three failures for missing capability API/event. GREEN: 48 passed, no skips, including prior discovery and current 3D graph checks. Logs: `.local/conv-transpose3d-discovery-red.log`, `.local/conv-transpose3d-discovery-green.log`.

Swift discovery RED failed on missing runtime descriptor API. After discovery implementation, a separate example RED failed on the missing example factory. Logs: `.local/conv-transpose3d-discovery-swift-red.log`, `.local/conv-transpose3d-example-swift-red.log`. Final Swift verification passed 22 tests (12 Core, 10 App), no failures/skips. The actual 3D example generated code, trained and verified History in 3.363 seconds. Project reopening, exact registry/descriptor fields, stale support and previous transposed discovery/example regressions passed. Log: `.local/conv-transpose3d-discovery-swift-green.log`. Final Python verification after public registry integration passed 61 tests, no skips, covering discovery, graph and lifecycle: `.local/conv-transpose3d-discovery-green-final.log`.

The selected example geometry is two measurements → 48 learned features → `[2,2,3,4]` → grouped 2-to-4-channel kernel `[3,3,3]`, stride `[2,2,2]`, padding `[1,1,1]`, output padding `[0,0,0]` → `[4,3,5,7]` → ReLU → 420 flattened responses → two class scores. With bias it has 1,098 parameters. The modular Learn entry explains scatter sums, geometric overlap counts of 1/2/4/8, output geometry and the qualified adjoint relationship, with the constructed-volume limitation stated explicitly. A rose Layers icon and compact `ConvT 3D` palette caption retain the full accessible name. The width output-padding exercise changes 420 responses to 480 and requires the corresponding classifier input edit.

Fresh runtime `.local/conv-transpose3d-runtime-20261007` copies the pinned dependencies and canonical worker/bootstrap/registry. Integrity verification reports 76 files matching current sources. This is build preparation, not signed-candidate or native acceptance.

## Packaged native acceptance — 2026-10-07

Candidate `af76f16`, fresh 76-file canonical runtime `.local/conv-transpose3d-runtime-20261007`. Packaging verified worker sources before/after local signing; `.local/conv-transpose3d-package.log`. Native observations used the 1440×900 KwipML window through computer use.

- Learn → Explore → expanded volume explanation rendered equations, overlap/adjoint discussion and complete constructed-volume caveat after scrolling. Example opened its real prepared workflow.
- Layers ribbon displayed distinct untruncated ConvT 1D/2D/3D captions and rose icons. All nine configuration controls fit the inspector; bottom Results remained independent. Declared input `[N,2,2,3,4]` and output `[N,4,3,5,7]` were exposed on connectors.
- Output padding `[0,0,0]` → `[0,0,1]` committed with Return, then toolbar Undo restored `[0,0,0]`. Bias disabled and code inspected before training: real `nn.ConvTranspose3d`, exact triples/groups and `bias=False`.
- Run `2282F3B7-85FC-428C-AE4D-5400046811DC`, request `6bcaee63ddf8fef5d5b4f0920daa02fd36c272211946bc71655e079c86636b4f`, started `2026-10-07T11:43:09.991Z`, finished `2026-10-07T11:43:15.109Z`. Three epochs completed; native History verified 29 artifacts, six computed preparation nodes and zero reused. Validation loss 0.0003254 / 100% on the tiny teaching fixture is execution evidence, not a generalization claim.
- Native reviewed export saved `.local/KwipML-model-2282F3B7`. Review explicitly required prepared float32 `[N,2]` signal/context inputs and disclosed excluded source rows/fitted preprocessing. Independent `inference.py --example --batch-size 2` exited zero, producing logits `[[0.13218945264816284,-0.2450476586818695],[-4.207324028015137,3.9552135467529297]]`, classes `[0,1]`. Log `.local/conv-transpose3d-export-inference.log`; synthetic example verifies executable portability, not accuracy.
- Quit/process absence, restart, Resume: output padding `[0,0,0]` and bias disabled persisted. History reverified the identical archived run and 29 artifacts. Candidate quit again and process absence confirmed.

Scoped delivery is closed; no broad production gate or distribution acceptance is implied.
