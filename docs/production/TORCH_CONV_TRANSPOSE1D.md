# TORCH-01 · ConvTranspose1d contract

Status: configuration, graph/runtime, portable source and checkpoint lifecycle verified. Public discovery, prepared example and scoped packaged native acceptance are verified. Broader PyTorch and production gates remain open.

## Authored operation

Actual nn.ConvTranspose1d, batched float32 [N,C,L]. Nine exact configuration keys: in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, groups, bias. Integer scalar settings; channels/groups1…512, kernel/stride/dilation1…4096, padding/output_padding0…4096. Boolean bias; groups divide both channel counts. Integral numeric JSON canonicalizes; Boolean numeric aliases, extra keys and list forms are refused.

[PyTorch2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose1d.html) describes a transposed linear convolution operator, not a general inverse. Weight shape is [Cin,Cout/groups,K]. Only zero boundary mode is implemented by this module, so no false multi-choice padding mode is exposed. Ordinary Conv1d same/valid policies do not apply. Authored output_padding controls output geometry; it is not a promise to append zero-valued samples. Dynamic forward output_size is outside this static node contract; explicit output_padding provides authored shape control.

Output length=(Lin−1)×stride−2×padding+dilation×(kernel−1)+output_padding+1. Require input/output lengths1…4096. Padding may exceed dilation×(kernel−1) when the resulting length is positive; this represents legitimate cropping and must not be rejected by reusing ordinary convolution padding rules. Channel/rank checks precede allocation.

## Cross-device validity and resource limits

Pinned2.14.1 probes (`.local/conv-transpose1d-probe.log`) show CPU/MPS forward/backward working for stride2/padding1/output_padding1; stride1/dilation2/output_padding1; and padding4 with kernel3 producing shorter positive outputs. CPU rejects output_padding2 with stride2/dilation1, but MPS executes it. Product validation must require output_padding < stride OR output_padding < dilation on every device; never accept the backend-specific invalid expansion just because MPS runs it. Independent numerical/gradient verification is still required during runtime integration.

Parameters=Cin×(Cout/groups)×kernel + optionalCout≤2M. Conservative work estimate=Lin×Cin×(Cout/groups)×kernel≤128M multiply-add contributions before cropping. Input/output sample limits remain100M elements. Uncropped output extent=(Lin−1)×stride+dilation×(kernel−1)+output_padding+1; require Cout×uncroppedExtent≤100M to make crop-related temporary extent explicit. This is a conservative geometry bound, not a peak-memory estimate. Positive bounded inputs keep arithmetic within Int64; no allocation occurs in normalization/resolution.

## Acceptance

Swift/Python tests first: exact fields/types and canonical copies; grouped/depthwise counts; bias crossing2M parameters; independent stride/dilation/outputpadding calculations; actualCPU shape comparisons; outputpadding accepted via dilation even when not belowstride; negative/excessive output refusal; padding larger than effectivekernel; and workbounds unaffected by cropping. Preserve ordinary convolution regression evidence.

Then graph/runtime/portable source/checkpoint integration with explicit indexed scatter-add output/gradient oracle, actual CPU/MPS numerical checks and saved-state lifecycle. Strict discovery/sharedpalette, prepared example/math and freshpackagednative controls/run/export/restart follow. Explain adjoint versus inverse and output-size ambiguity alongside the component. No gradedlesson/fullproductiongate claim from a prepared example alone.


## Foundation evidence — 2026-10-07

Python RED missingmodule (`.local/conv-transpose1d-contract-python-red.log`); GREEN261tests including ordinary1D/2D/3Dcontracts,0skip, expectedordinaryConv3dsamepaddingcopywarning (`.local/conv-transpose1d-contract-python-green.log`). Swift REDmissingAPI (`.local/conv-transpose1d-contract-swift-red.log`); GREEN17Coretests,0fail/skip including ordinaryconvolutionconfiguration (`.local/conv-transpose1d-contract-swift-green.log`).

Cases cover analytical/actualCPU output shapes, grouped/depthwise weights, both valid branches of the outputpaddingORrule, largepadding with positiveoutput, strictninekeys/types and canonicalcopies, biascrossingparametercap, outputlength and input-basedworklimits. Review confirms boundedInt64arithmetic. The100Muncropped guard is defense in depth: validoutput≤4096,padding≤4096 imply uncropped≤12288, and Cout≤512 implies≤6,291,456 elements. Tests do not pretend to isolate an unreachable100Mrefusal under these simultaneousbounds.

Local constructorprobe additionally confirmed zerosaccepted while reflect/replicate/circular are refused by pinnedPyTorch. No fake modechoices exposed. Next graph/runtime/independentscatter-add gradients/export/checkpoint tests; nativecandidateb87955e remainslastaccepted/quit. Allverificationterminal, productiongatesunchanged.


## Graph/runtime and checkpoint evidence — 2026-10-07

The private nine-field component now supplies schema-2 shape inference, float32 admission and aggregate parameter accounting. Separate transposed-convolution dispatch maps preserve ordinary convolution behavior. Runtime uses actual `nn.ConvTranspose1d`; standalone generated source retains output padding and checkpoint keys, and both paths reject empty or incorrectly shaped batched inputs.

Swift RED failed on the missing component API; GREEN passed 18 Core tests with no failures or skips (`.local/conv-transpose1d-graph-swift-green.log`). Python graph RED: 22 failed, 6 passed for missing runtime support. Graph GREEN: 147 passed, no skips, including existing convolution graph regressions (`.local/conv-transpose1d-graph-python-green.log`). An independent indexed scatter-add oracle verifies values and all input/parameter gradients across grouping, bias, stride, dilation, output padding and large positive cropping. Actual MPS values and gradients agree with CPU across five settings. Allocation refusal intercepts the actual constructor dispatch map.

Lifecycle RED: 4 failed, 7 passed, exposing missing optimizer parameter ordering. The minimal schedule change includes ConvTranspose1d while preserving prior ordinary and convolution signatures. Final combined GREEN: 229 passed, no skips (`.local/conv-transpose1d-lifecycle-green.log`), covering transposed configuration/graph/lifecycle, ordinary convolution graph/lifecycle and shared checkpoint regressions. One expected ordinary Conv1d same-padding copy warning remains. Evidence includes exact CPU continuation with RNG-consuming callbacks, captured weight gradients, frozen/export parity, weights-only fine-tuning reset, initialized/trained cache reuse and stale refusal, plus real MPS training/preview.

All verification handles terminated successfully. The component is not yet publicly exposed; next delivery is capability negotiation, shared palette and prepared example before fresh packaged native acceptance. Last accepted candidate remains `b87955e`, quit. Production gates remain open.


## Discovery and prepared example evidence — 2026-10-07

The live `conv_transpose1d` descriptor binds scalar configuration, rank-2 float32 samples, zero boundaries, the output-padding OR rule and resource limits. Native decoding requires an exact descriptor; optional absence/malformed support clears prior state and prevents graph execution without disabling legacy components. The shared registry matches the nine reviewed fields. Channels/length ports and learned state are declared by the capability bridge.

The prepared universal flow constructs [N,2,7] from two measurements, applies grouped ConvTranspose1d with kernel3/stride2/padding1/output-padding1, then ReLU, flatten56 and a two-score classifier. Its 172 parameters include projection, transposed kernels/bias and classifier. A separate Learn view explains indexed scatter-add, shared/grouped weights, output-size ambiguity, the qualified adjoint identity and constructed-data limitations. The shared Layers rose icon depicts expansion. No graded lesson is claimed.

Python discovery RED: 3 missing API/key failures; GREEN: 48 passed, no skips, including live CLI, actual exported execution and graph/lifecycle regressions (`.local/conv-transpose1d-discovery-python-green.log`). Swift RED observed missing discovery and example APIs in separate terminal runs. First GREEN attempt exposed an incorrect test expectation: PyTorch normalizes output_padding to `(1,)`; the generated source correctly preserves that tuple. Corrected assertion, no product workaround. Final GREEN: 24 tests (16 Core, 8 App), no failures/skips (`.local/conv-transpose1d-discovery-swift-green.log`), including actual code before run, training and verified History in 3.188 seconds, optional/stale discovery and project reopen.

Fresh payload `.local/conv-transpose1d-runtime-20261007` matches 74 canonical worker files (`.local/conv-transpose1d-payload.log`). Packaging and native checks follow; all automated verification handles are terminal.


## Packaged native acceptance — 2026-10-07

Candidate `2d56dd9` uses the fresh 74-file canonical payload. Packaging verified current-source integrity before and after local signing (`.local/conv-transpose1d-package.log`); no distribution-signing claim. Native Learn rendered the scatter-add/output-length equations and complete adjoint/data caveats after disclosure expansion and scrolling. The example opened with declared [N,2,7]→[N,4,14] connectors. Output padding changed 1→0 with Return and Undo restored1; the incompatible downstream width invalidated model contracts until Undo. Bias was disabled and persisted. All nine controls fit the 1440×900 inspector; the Layers rose expansion icon appeared in the ribbon and the independent Results dock stayed at the bottom. Code before training showed actual nn.ConvTranspose1d with stride(2,), padding(1,), output_padding(1,), groups2 and biasFalse.

Real three-epoch run `196FF58B-1492-4873-B9D9-D19880AE05C8` verified 29 archived artifacts. Request `5cb72f5421e73bba8ab14327f7039ef89018093303a1f9dbcaf724379a7c6826`; started2026-10-07T10:57:13.401Z, finished10:57:18.555Z. Preparation computed6/reused0. Final validation loss0.1412 and100% accuracy apply only to this tiny teaching fixture, not a generalization claim.

Native export saved `.local/KwipML-model-196FF58B`, explicitly disclosing prepared float32[N,2] signal/context inputs and exclusion of source rows/fitted preprocessing. Independent `.venv/bin/python inference.py --example --batch-size 2` exited0 (`.local/conv-transpose1d-export-inference.log`), yielding finite logits [[0.0106970891,−0.0725120008],[−0.9887951612,0.7074102759]] and classes[0,1].

After quitting/reopening, the launcher resumed this project with output padding1 and biasFalse; History reverified the same run and29artifacts. Candidate quit again and process absence was confirmed. This closes the scoped ConvTranspose1d delivery; graded curriculum, broader interaction/device/accessibility matrices and full production gates remain outstanding.
