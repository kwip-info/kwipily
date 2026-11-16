# Native POC execution and polish UAT — 2026-10-04

These are actual observations from the native ad-hoc signed KwipML POC, operated through computer use. Automated evidence is separate. The owner's active exploration interrupted the CSV-mapping/course replay; those scenarios are explicitly distinguished below.

## Gate C — real execution

| Scenario | Observed result |
|---|---|
| Open clean reference flow | Eight blocks/eight connections, graph valid, all stages initially Not evaluated |
| Preview through Combine | Six actual computed stages; bounded float32 tensor [8,16] and eight numeric values visible |
| Run from Combine | Exactly five upstream stages labeled Cached; Combine/Classifier/Prediction labeled Computed; footer reports 3 computed / 5 cached |
| Train from native Run flow | Actual MPS worker; 100/100 epochs; held-out loss0.0659 and accuracy99.5% on bundled synthetic example; measured charts and predictions |
| Generated source | Code tab displays real nn.ModuleDict Linear layers, ReLU/Tanh, torch.cat and forward traversal |
| Parameter invalidation | Input features + changes2→3, precise mismatch shown, Run disabled, all captured values marked Stale; Undo restores valid graph and current result |
| Invalid numeric draft | Entering `nope` and Return leaves draft visible with range error; committed graph stays2features and valid |

This accuracy measures only the bundled synthetic demonstration. It is not a language-model result or product benchmark.

## Gate D — persistence, stage inspection and interaction

| Scenario | Observed result |
|---|---|
| Quit/relaunch | App opens Projects and lists saved recent diagrams |
| Open saved Code & artifacts | Prior model.py is restored without rerunning |
| Restored Run view | Prior100epochs, loss0.0659, accuracy99.5%, actual MPS and restored-manifest event visible |
| Preview from restored checkpoint | Explicit `Inference · trained checkpoint`; real [8,16] values match previous training capture |
| Run from trained checkpoint | 3computed / 5cached; both branches show Cached |
| Grouped component palette | Data, Layers, Activations, Merge with purpose subtitles; all seven square semantic icons and distinct muted theme colors |
| Square-icon drag | Drag Linear icon into graph adds Linear4,9blocks/8wires; inspector opens; unconnected-input validation disables run; Undo restores8blocks |
| Background drag | Graph moves by drag delta; node positions stay intact; minimap viewport follows; Fit resets viewport |
| Hidden native scrollers | Final canvas screenshot has no persistent native bars; drag-to-pan still visibly moves graph after native-scroller fix |
| Tensor formatting | Captured tensor now displays `[8, 16]`, replacing multiline NSArray description |
| CSV source import | Native picker imported .local/fixtures/classification.csv; source800rows/3columns and first8actualrows shown; custom mapping controls visible |

CSV mapping/train and full course replay were not yet completed through computer use at this point because the owner navigated into a guided project and began editing it. Do not count owner actions as agent-controlled UAT. Native-store integration and model/runner tests cover those code paths separately; final replay status must be appended after verification.

## Findings corrected before final audit

- SwiftUI hidden-scrollbar preference did not override this Mac's persistent-scrollbar setting. Applied an app-local native scroll policy; no system preferences changed.
- Cache freshness initially checked only graph/source/tensor existence. Added sidecar/checkpoint artifact hashes and prefix eligibility; worker independently performs authoritative graph/source/device/seed/weights/integrity checks.
- Source hashing ran repeatedly during canvas redraw. Hashes now memoize by path/size/modification time; worker always hashes actual CSV bytes for provenance.
- Restore previously set completed before validating snapshot/artifacts. New regression tests require decodable graph, nonempty model/checkpoint and valid finite metrics before publishing completion.
- Cached source provenance now comes from actual worker-loaded data, avoiding a pre-launch read race.

## Final replay additions

After the owner made the window available for final checks, created a separate `POC acceptance · course and CSV` project. Guided build replay observed0→1→2→8blocks, automatic correct connections and changing lesson copy, ending8blocks/8wires and graph valid. The existing owner-explored project was untouched.

A10,000epoch MPS run completed successfully while the automation was obtaining a slow live-chart observation. The attempted cancellation reached an expired element; therefore no cancellation pass is claimed for that attempt. This exposed excessive chart work. Bounded sampled rendering/live publication was added; all original epoch measurements remain in metrics.json. Native cancellation is retested below.

Verified local manifests independently of the UI:
- `.local/runs/36842680-7128-4210-ACE8-B3ED0125B8BF/artifacts/manifest.json`:100epochs, mps, validation200, accuracy0.9950000047683716, loss0.06589192152023315.
- `.local/runs/381E8547-51E3-4F4F-A41F-03258B102E82/artifacts/manifest.json`:10,000epochs, mps, validation200. This long demonstration is not used as a product-quality benchmark.

Automated native-store integration independently exercised real40-row CSV training for5CPUepochs and cancellation after measured progress. Both passed with isolated temporary project/artifact roots; they supplement but do not replace computer-use evidence.

## Final acceptance — passed

- Custom CSV choices: native computer use opened each popover, selected measurement_a, measurement_b and category; each selection closed and updated its label. Compute popover selected explicit CPU. No native menu tracking loop remains.
- Cancellation retest: native10,000epoch mapped-CSV MPS run showed measured progress (last retained epoch2000), then Cancel run returned controls and footer `Run cancelled. No completed result published.` No Reveal artifacts/completed state was offered. Cancellation label refinement now distinguishes partial metrics from errors; its native-store regression observed missing-property red then passed.
- Final complete flow: course-built8-node graph + mapped800-row CSV +100CPUepochs completed. UI showed validation loss0.0506, held-out accuracy99.4%, real charts/predictions, CPU and saved exports.
- Independent final manifest `.local/runs/4892A214-98EC-4D2D-A7B1-22FCB957CA27/artifacts/manifest.json`: CPU,100epochs,640training/160validation observations, featureColumns measurement_a/measurement_b, target category, content hash800943bee8caa02753f5f8506b750c0ff6287bfe6521f01a13b0bc59451bf2f9. Accuracy0.9937499761581421 and validationLoss0.05055354908108711 match displayed rounding.
- Compact window tested at approximately1082×730. Canvas Fit showed complete graph; Cmd-Option-1 toggled palette, Cmd-S saved. Found wrapped Examples label in crowded toolbar; final refinement fixes tab labels to one line and removes duplicate graph counts from that toolbar (counts remain in overview).
- Shared quiet scrolling verified in final Data, Settings, Run and inspector views; no persistent bars in screenshots. Native file chooser intentionally keeps system chrome. Palette scroll remains available by wheel/trackpad.

Final automated verification:38Swift tests (22Core +16App including2real native-runner integration tests) and50Python tests. Native build and ad-hoc signing succeed. These establish the one-flow POC on this Mac, not App Store readiness or compatibility across untested devices.

Gates A–E accepted for the POC scope. Earlier failed attempts above are retained as evidence of the refinement process; pending labels in earlier historical sections are superseded by this acceptance section.
