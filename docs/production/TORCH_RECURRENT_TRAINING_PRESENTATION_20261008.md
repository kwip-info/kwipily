# Recurrent training presentation repair — 2026-10-08

Scoped automated closure; native replay and family acceptance remain root gates.

The actual LSTM classifier completed worker training but publication refused with “Training tensor previews exceed their display contract.” The shared presentation verifier treated recurrent named outputs as one tensor. Native evidence is recorded in TORCH_RECURRENT_NATIVE_20261008.md for project43E0D7C3-27E5-4E14-ABA5-C41337101520/runC719CBAF-9A80-4B45-8A58-5B793D2D045F.

Only WorkflowModelTrainingExecution.swift and new RecurrentTrainingPresentationTests.swift changed. The recurrent branch resolves effective configuration and graph output contracts, verifies each connected input/state preview (including an upstream named hidden port), requires identical positive observation counts, resolves every output shape, and checks exact roles, float32 dtype, bounded finite values and existing capture status/privacy policy. Unused hidden/cell roles receive identical validation. Cell result/hidden previews must be exact aliases; sequence result and final hidden are distinct. Common tensor verification and prior split/chunk/pooling branches remain unchanged. Saved-artifact/event preview equality remains unchanged.

Actual TDD used Xcode toolchain and four jobs. Initial default CommandLineTools invocation failed compilation in unrelated existing expressions; it is retained separately and is not semantic RED. The first Xcode test harness required a String-to-JSONValue fixture correction, with assertions preserved. Actual semantic RED then executed four methods:19 assertion failures, zero unexpected,1.301seconds. Valid all-six named captures, privacy/status and named supplied-state examples reproduced the native error; malformed negative cases already refused. After the branch repair, seven scoped methods passed. Root requested the exact cell alias boundary; a new method then captured nine actual failures for differing values, counts or missing samples, followed by the production alias guard.

Final actual run: five new recurrent methods plus three existing TensorRoutingCaptureTests methods, eight passed, zero failures/zero unexpected,0.333seconds. It covers all six captured-style output schemas, malformed unused states/roles/shapes/dtypes/value bounds, captured/omitted policy/status, supplied hidden from a named upstream GRUCell output, mismatched observations and exact cell aliases. This is presentation-contract evidence, not a new native training or runtime numerical measurement.

| Evidence | SHA-256 |
| --- | --- |
| .local/recurrent-training-presentation-semantic-red.log | 2a519cc31b96c57643530c64bf9764cf01b1b65cdcd0893728e75e153bf62d45 |
| .local/recurrent-training-alias-red.log | cd2a428432a5324c7c082d7888d0989344ab3e155c6bb8b5fd532ee2528b693a |
| .local/recurrent-training-presentation-closed-green.log | a026ea7abc92a480bc4385196f20805a05706fa8013aa06356940e4d6701f53a |
| Sources/KwipMLApp/WorkflowModelTrainingExecution.swift | f3cd47e8a5d1e31564038a5c3984adfa47d5d02ef6de7db642c65f65c0066e77 |
| Tests/KwipMLAppTests/RecurrentTrainingPresentationTests.swift | 9061e6a0f57bec84193c392bb3459cf4e7027152d88bdbb431fa8e256110f172 |
