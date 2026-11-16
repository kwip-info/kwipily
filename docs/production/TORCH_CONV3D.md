# TORCH-01 · Conv3d delivery contract

Status: scoped Conv3d delivery accepted locally through packaged native run/export/restart; broader convolution and production acceptance remain open. Owner: Codex/root. Builds/tests/native acceptance remain root-owned.

## Authored geometry and semantics

Actual nn.Conv3d with batched float32 [N,C,D,H,W]. Exact ten settings follow Conv2d, but kernel_size, stride, dilation and padding are explicit three-integer arrays [depth,height,width]. Scalar, two-element, Boolean or extra-key representations are refused; integral numeric JSON canonicalizes to integer arrays. Channels/groups1…512, spatial dimensions/kernel/stride/dilation1…4096, explicitpadding0…4096. Groups divide both channel counts; bias is Boolean. Policies explicit/valid/same and modes zeros/reflect/replicate/circular. Same requires stride[1,1,1]; nonexplicit policies require padding[0,0,0].

Reuse the existing per-axis geometry resolver for all three axes. Effective kernel=dilation×(kernel−1)+1. Same splits totalpadding=effective−1 with smaller half before; explicit pads symmetrically; valid0. Resolve front/back,top/bottom,left/right explicitly. Reflection sides must be smaller than the corresponding input extent; circular sides may equal but not exceed it. Each resolved side≤4096; each output extent1…4096. Reject negative output numerator before integer division.

[Official PyTorch Conv3d documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv3d.html) establishes cross-correlation and depth/height/width ordering. Local pinned2.14.1 probe (`.local/conv3d-runtime-probe.log`) passed forward/backward on CPU and actual MPS for all12policy/mode combinations, using groups2,channels2→4,kernel[2,3,2],input[2,4,5,6]. Explicitpad[1,1,1] yields[4,5,5,7], valid[4,3,3,5], same[4,4,5,6]. This probe establishes feasibility, not complete numerical/backend acceptance; independent output/gradient comparisons belong to the runtime unit.

## Allocation bounds

Parameters=Cout×(Cin/groups)×kD×kH×kW + optionalCout≤2M. Input/output sample volumes and conservatively padded inputvolume≤100M elements. Estimatedmultiplyadds=Dout×Hout×Wout×Cout×(Cin/groups)×kD×kH×kW≤128M per observation. Large stride must not bypass padded-volume refusal. These limits are not peak-memory predictions; batch, gradients and backend workspaces add cost.

Use bounded multiplication/order in Swift: parameter count is checked before resolution; validate outputvolume before multiplying by fan-in (already≤2M). Do not extend an unbounded six-dimensional work product by another axis and risk Int overflow. Geometry remains allocation-free. Include adversarial large authored triples, all-axis padding bounds, asymmetricsame, nonsymmetric stride/dilation, grouped/depthwise counts, strict types and negative-numerator fixtures.

## Delivery sequence

Configuration tests RED→implementation→Swift/Python GREEN with Conv1d/2d regressions. Then graph/runtime/export/checkpoint integration, independently indexed three-dimensional output/gradient oracle and actual CPU/MPS tests. Strict device/discovery contracts, prepared volume example/math explanation, fresh packaged native controls/run/History/export/restart precede scoped closure. No video/image/medical volume source adapter is implied by a Conv3d operator, and no graded course or full production gate is claimed by a prepared example.


## Foundation evidence — 2026-10-07

Swift RED missingConv3dConfiguration (`.local/conv3d-contract-swift-red.log`); GREEN13tests including Conv1d/2dconfiguration,0fail/skip (`.local/conv3d-contract-swift-green.log`). New5tests cover grouped/depthwise arithmetic, all6paddingfaces, each-axis validation, canonicaltriples and adversariallimits. Reviewed arithmetic: configured parameterproduct≤512²×4096³ fitsInt64; paddedvolume≤512×12288³ fitsInt64 before cap. Output layout admission precedes work; division-based limitcheck avoids overflow before multiplication.

Python RED missingmodule (`.local/conv3d-contract-python-red.log`); GREEN231tests including Conv1d/2dcontracts,0skip and one expected PyTorch same-padding copy warning (`.local/conv3d-contract-python-green.log`). Cases compare all12policy/mode shapes to actualCPU and independently check nonsymmetric axes, strictcanonicalJSON, exactparameterboundary, each-axisreflection/circular, paddedvolume despite largestride, output and worklimits. Shared axis resolver reused unchanged.

Next graph/runtime/export/state integration with independently computed outputs/gradients and actualMPS numerical comparison. The separate24case CPU/MPS feasibilityprobe is not a substitute for that evidence. Lastacceptedcandidate449ac8e remainsquit; no productiongates closed.


## Graph, portable code and lifecycle — 2026-10-07

Added private ten-field Conv3d component, strict schema2 float32 volumetric graph contracts and aggregate/trainable parameter accounting. Shared reviewed convolution maps dispatch normalization, geometry and actual nn.Conv1d/2d/3d constructors. Export retains named state, exact authored nonempty batch/sample guards and correct dimensional constructor. No public palette entry yet.

Swift RED missingConv3dComponent; GREEN15Coretests0fail/skip including existing convolutiongraph and newconfigregressions (`.local/conv3d-graph-swift-green.log`). Python graphRED32failed/6passed; initialGREEN119graph tests. Independent indexed3Dcross-correlation oracle checks outputs, input/allparametergradients, bothbias and all4boundarymodes; additional tests cover noncubicstride/dilation/groups/depthwise, actualMPS12policy/mode numericalcomparisons, exactshape/dtype/schema/resource refusal and portable source/state/gradients including userlabel `tuple`.

LifecycleRED4failed/6passed identified missing/omitted Conv3dparameterNodeOrder (`.local/conv3d-lifecycle-red.log`). Minimal membershipchange binds newfamily registration order, preserving priorordinary/Conv1d/Conv2dschedules and lazy-onlyinitialization. Tests verify exactCPU4epoch vs2+resume model/optimizer/RNG/metrics with RNG-consumingcallbacks; gradientinventory, frozen/exportparity, finetune reset, initialized/trainedcache reuse/stalerefusal and actualMPStrain/preview.

Review corrected allthree convolution allocationguard tests to intercept the new constructor map, rather than obsolete direct nn attributes. Final combined261Pythonpass0skip, oneexpected same-paddingcopywarning (`.local/conv3d-runtime-accepted.log`). Allverificationterminal. Next strictdiscovery/sharedregistry/preparedexample, then freshpackage/nativeacceptance. Lastacceptedcandidate449ac8e remainsquit; no productiongateclosure.


## Discovery and prepared example — 2026-10-07

Exact optional native/worker descriptor now binds rank4 samples and three spatial entries, float32, padding/group choices and all reviewed resource bounds. Stale reset/refusal and component counts include Conv3d. The bridge describes C,D,H,W with exactly one batch axis, learned state and the real prepared example. Shared registry matches the ten-field component definition.

Prepared example projects48features→[2,2,3,4]→groupedConv3d2→4kernel[2,2,3]same→ReLU→flatten96→classifier2. Rose volume icon follows Layers grouping. Learn explains cross-correlation, group indices, weight sharing, each-axis output equation, six padding faces and48weights+4optionalbiases. Constructed volumes are distinguished from medical/video data; no new source adapter or gradedlesson implied.

Python RED2fail missingdescriptor/API; GREEN52pass0skip with existingdiscovery and Conv3dgraph/lifecycle checks after publiccatalog changes (`.local/conv3d-discovery-catalog-green.log`), expected samepaddingcopywarning. SwiftRED missingruntimecapability API (`.local/conv3d-discovery-example-red.log`); GREEN30tests23Core/7App,0fail/skip (`.local/conv3d-discovery-example-green.log`). Includes strictdecoder/bridge/staleclear/liveworker, bothbiasvolumeshapes, layout/codec/privatebinding/reopen, and actualcode-before-run/preparation/training/verifiedHistory3.283s.

Next freshpackagedcandidate and native authoring/Learn/run/export/restart. Existingaccepted449ac8e remainsquit until replacement; no productiongateclosure from automatedchecks.


## Packaged native acceptance — 2026-10-07

Candidateb87955e used fresh `.local/conv3d-runtime-b87955e`;73canonicalworkerfiles and pre/post-sign integrity passed (`.local/conv3d-payload.log`, `.local/conv3d-package.log`). Local ad-hoc signing only; previous candidate retained.

Observed Learn volume explanation/equation fully rendered, real example opened, connector shapes[N,2,2,3,4]→[N,4,2,3,4]. Triple kernel2,2,3→1,2,3 committed withReturn; Undo restored2,2,3. Biasoff and Reflect selected; all ten controls visible in the inspector at1440×900 with bottomResults remaining independent. Codebeforetraining displayed nn.Conv3d(2,4,(2,2,3),stride=(1,1,1),padding='same',dilation=(1,1,1),groups=2,bias=False,padding_mode='reflect'). This is scoped observed interaction, not a full keyboard/VoiceOver matrix.

Actual3epoch run **2EEE5AEA-A7A9-467B-800A-D7A498D0A392**, request **12c03f31e4a20713b322aa215c392419ff12de4b06a810fb898c77b166275a47**, started2026-10-07T10:35:15.058Z, finished10:35:20.296Z. Historyverified29artifacts,6computed/0reused, validationloss0.0022012/100%smallfixtureaccuracy; no broadergeneralizationclaim.

NativeHistory→Savedmodelsource→trainedexport saved `.local/KwipML-model-2EEE5AEA`. Reviewdisclosed preparedfloat32[N,2] inputs, signal/contextorder and excluded source rows/fittedpreprocessing. Independent `.venv/bin/python inference.py --example --batch-size2` exited0 (`.local/conv3d-export-inference.log`), logits[[0.0024987310,-0.1481397897],[-3.1710872650,2.6459686756]],classes[0,1].

Quit/relaunch resumed savedtriple2,2,3,Same,Reflect,biasoff; Historyreverified the identical completedarchive. Candidatequit and processabsence confirmed. Scoped Conv3d closed; next inventoried convolution unitConvTranspose1d. No gradedlesson or productiongateclosure implied.
