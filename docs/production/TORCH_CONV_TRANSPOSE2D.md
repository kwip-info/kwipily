# TORCH-01 · ConvTranspose2d contract

Status: configuration, graph/runtime, portable source and checkpoint lifecycle verified. Public discovery, prepared example and scoped packaged native acceptance verified. Broader PyTorch and production gates remain open.

## Authored configuration and geometry

Actual `nn.ConvTranspose2d`, batched float32 [N,C,H,W]. Nine exact settings: in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, groups, bias. Five spatial settings use explicit [height,width] integer lists; scalar aliases are intentionally canonicalized by authoring both entries, not accepted in graph JSON. Channels/groups1…512; kernel/stride/dilation1…4096; padding/output padding0…4096. Boolean bias. Groups divide both channel counts. No fake padding policies/modes: this module supports zero boundaries only. Dynamic forward output_size is outside this static contract.

[PyTorch2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html) defines per-axis output as (input−1)×stride−2×padding+dilation×(kernel−1)+output_padding+1. Weights have shape [Cin,Cout/groups,Kh,Kw]. Output padding selects an otherwise ambiguous extent; it does not append zero-valued output. Transposed convolution is not generally an inverse. Each axis must independently satisfy output_padding<stride OR output_padding<dilation. Large padding may legitimately crop more than the effective kernel extent when output remains positive.

## Admission and resource policy

Input/output axes1…4096, sample volume≤100M. Parameters=Cin×(Cout/groups)×Kh×Kw + optionalCout≤2M. Conservative work=Hin×Win×weightCount≤128M, measured before cropping. Uncropped axis=(input−1)×stride+dilation×(kernel−1)+output_padding+1; Cout×Uheight×Uwidth≤100M is a conservative geometry admission bound, not a peak-memory measurement. Unlike the1D contract, this bound can independently reject an otherwise tiny final output.

Validate parameters before work multiplication: although the raw maximum product would exceed Int64, parameter admission bounds weightCount≤2M and spatial input area≤4096², so admitted arithmetic≤33,554,432,000,000. Validate output axes before multiplying uncropped volume; output≤4096 and padding≤4096 imply each uncropped axis≤12288. Ordinary output-volume rejection also exceeds uncropped volume because U≥output; output is checked first for a useful diagnostic, not falsely claimed as an independent bound.

## Pinned runtime observations

Local2.14.1CPU/MPS probes (`.local/conv-transpose2d-probe.log`) passed forward/backward for rectangular stride/output padding, dilation admitting output padding≥stride, and large positive crops. Example input[2,5,6] with kernel[2,3],stride[1,2],padding[1,1],output_padding[1,2],dilation[2,3],Cout4 gives[4,6,17] on both. CPU rejects output_padding[2,0] with stride[2,2]/dilation[1,1], while MPS executes it. Product validation must enforce the portable per-axis rule universally. Constructors accept zeros and reject reflect/replicate/circular. These probes do not substitute for independent numerical/gradient runtime verification.

## Planned acceptance

Strict normalization, canonical independent lists, groups/bias/parameter limits, rectangular output math, actualCPU shape checks, per-axis OR validity, legitimate cropping and preallocation/resource boundary tests in Swift/Python. Then private graph/runtime/source/checkpoint support with an independent2D scatter-add oracle and actualMPS gradients; strict discovery, prepared example/math, then fresh packaged native edit/run/export/reopen. No public component or production gate is accepted by configuration alone.


## Foundation evidence — 2026-10-07

Swift RED missing configuration API; GREEN19Core tests/no failures or skips, including existing1D-transposed and ordinary2D/3D configurations (`.local/conv-transpose2d-contract-swift-green.log`). Python RED missing module; GREEN294passed/no skips with the same prior configuration families (`.local/conv-transpose2d-contract-python-green.log`). Expected ordinaryConv2d same-padding copy warning only.

Independent geometry tests compare rectangular predictions with actualCPU shapes. Strict pairs/types/exact fields, group/depthwise/bias behavior and valid large crops are covered. Exact128M input-based work passes and the next wider case fails. Exact100M uncropped volume passes;100.1M fails despite only600 final output elements. The uncropped guard can independently reject tiny outputs; the output100M guard overlaps it by definition and is checked first. Swift admits parameter count before computing work and uses a division guard; reviewed bound ordering avoids Int64 overflow.

All verification handles terminal. Existing1D implementation remains unchanged. Next private graph/runtime/export and checkpoint integration, with independent2D scatter-add values/gradients and realMPS numerical coverage. Lastaccepted2d56dd9 candidate remains quit; production gates unchanged.


## Graph/runtime and checkpoint evidence — 2026-10-07

Private component schema declares all nine fields and exact H/W pairs. Swift resolves channels-first float32 graphs, schema-2 normalization and trainable/aggregate parameter counts. Python adds2D to the existing transposed constructor/resolver/normalizer maps; actual execution and standalone export reuse shared paths, including exact nonempty batched-shape guards. Shape errors now display real constructor names rather than title-cased internal IDs, improving both1D/2D messages without altering admission.

Swift graph RED missing component API; GREEN21Core tests/no failures or skips (`.local/conv-transpose2d-graph-swift-green.log`). Python graph RED27failed/6passed for missing support; graph GREEN137passed/no skips with preceding convolution families (`.local/conv-transpose2d-graph-python-green.log`). An independently indexed2Dscatter-add calculation verifies outputs and all input/parameter gradients across five geometries with both bias modes. Rectangular stride, grouping, dilation-admitted output padding and large crops are included. Generated source is executed with transferred state and matching gradients; user label “tuple” does not break shape guards. Ten actualMPS comparisons cover the same geometry/bias combinations. Aggregate parameter refusal intercepts the real constructor dispatch map before allocation.

Lifecycle RED4failed/8passed exposed absent optimizer parameter ordering. Added2D to the schedule activation/membership sets; previous ordinary and1D-transposed schedules remain unchanged. Final combined GREEN328passed/no skips (`.local/conv-transpose2d-lifecycle-green.log`): transposed2D configuration/graph/lifecycle, prior convolution graphs/lifecycles and common checkpoint regression. One expected ordinaryConv1d same-padding warning. Tests cover exactCPU continuation with RNG-consuming callbacks, captured [Cin,Cout/G,Kh,Kw] gradients, frozen/export parity, fine-tune reset, cache reuse/stale refusal and actualMPS training/preview.

All handles terminal; candidate2d56dd9 remains last accepted and quit. Next discovery/shared palette and prepared example before fresh packaged native acceptance. Public catalog and production gates remain unchanged.


## Discovery and prepared example evidence — 2026-10-07

Live `conv_transpose2d` capability advertises exact H/W pairs, rank3 float32 samples, zero boundaries, per-axis output-padding validity and reviewed limits. Native optional decoding rejects missing/changed/extra fields and clears stale support; unsupported graphs cannot execute. Shared registry matches all nine component fields; capability ports expose channel/height/width axes and learned state.

Prepared flow projects two measurements to24features, reshapes[2,3,4], applies grouped kernel[3,3]/stride[2,2]/padding[1,1]/output_padding[1,0], then ReLU/flatten168/classifier2. Output[4,6,7];450 total parameters with bias. A modular Learn entry explains2Dscatter-add,1/2/4interior overlap counts, independent output axes, weight grouping and the qualified adjoint relation. It clearly identifies grids as learned constructions, not real source images. Layers rose spatial-expansion icon shares palette/canvas identity. No graded lesson claimed.

Python discovery RED3missing APIs/keys → GREEN55passed/no skips, including liveCLI, executable exportedORgeometry and previous discovery/graph/lifecycle coverage (`.local/conv-transpose2d-discovery-python-green.log`). Swift RED separately observed missing discovery and example APIs; GREEN26tests (18Core/8App), no failures/skips (`.local/conv-transpose2d-discovery-swift-green.log`), including actual code-before-run/training/verifiedHistory3.358s, strict optional support and project reopening.

Fresh `.local/conv-transpose2d-runtime-20261007` payload matches75canonical worker files (`.local/conv-transpose2d-payload.log`). All automated verification terminal. Next fresh packaged native acceptance; previous2d56dd9 candidate remains lastaccepted/quit.


## Packaged native acceptance — 2026-10-07

Fresh75-file payload passed current-source integrity and local signing. Initial candidated303cd3 rendered full Learn equations/overlap/adjoint/data caveats, opened the example, and showed [N,2,3,4]→[N,4,6,7] connectors. Paired output padding[1,0]→[0,0] committed with Return; Undo restored[1,0] and valid downstream contracts. Biasoff persisted. All nine controls fit the1440×900 inspector and Results remained an independent bottom dock.

Native review found both long transposed-convolution ribbon captions truncated identically. First shortening still clipped; final commit`e254950` uses `ConvT 1D`/`ConvT 2D`, visibly distinct and untruncated. Full names remain in tooltips and accessibility labels. Final packaged candidate uses the same verified75-file worker; `.local/conv-transpose2d-caption-final-package.log` records successful source verification/signing. Code before execution correctly showed nn.ConvTranspose2d, paired settings, output_padding(1,0), groups2 and biasFalse.

Real three-epoch run `6B8E0A86-553D-4CC9-96A5-7F52446CD26B` verified29artifacts. Request `35377d3d9c89fdacddd19f6eaa449ab5a847d6e595423ed7bb0366527286fef4`, started2026-10-07T11:19:36.023Z, finished11:19:41.179Z. Preparation computed6/reused0. Loss0.024659/100% accuracy describe only the tiny teaching fixture.

Native export saved `.local/KwipML-model-6B8E0A86`, disclosing preparedfloat32[N,2] signal/context inputs and omitted source rows/fitted preprocessing. Independent `.venv/bin/python inference.py --example --batch-size 2` exited0 (`.local/conv-transpose2d-export-inference.log`), returning finite logits [[−0.0822392255,0.0046998039],[−1.8099482059,1.6671004295]] and classes[1,1]. Synthetic examples are execution demonstrations, not an accuracy assertion.

After quit/reopen, output padding[1,0] and biasFalse remained; the same archived run and29artifacts reverified. Candidate quit again and process absence confirmed. Scoped ConvTranspose2d is closed. Broader accessibility/device/interaction matrices, graded curriculum and production gates remain outstanding.
