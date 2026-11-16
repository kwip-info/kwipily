# Remaining normalization actual native BN1 export · 2026-10-08

This records independent verification of the actual UI-saved prepared BatchNorm1d classifier supplied by root. It is a finite native artifact check, not blanket framework, supported-machine or distribution acceptance. The helper ran only with the host `.venv/bin/python`; neither the signed bundled runtime nor Swift/UI/build tools were invoked by this agent. Package/archive source bytes remained read-only.

Package: `.local/KwipML-model-15F870F2`. Archive: App container workflow-runs project `056FF2E4-2C6A-4E0E-81B5-4EC1A1B6A7B0`, directory `F3F3695C-71E6-453A-ADB4-58A09EB6F928`. Logical run `15F870F2-0FAA-47B7-B5CE-6C9FB89880B3`, request `63c4b08e9c91618a4e011d16d271586a2653f14f7139acc148ec36047b25d7f9`, revision0. Actual training device mps; saved completed epoch 3, exact int64 training-call counter 6. Export/capture ownership, source manifest hash, inventory byte counts/hashes, model/checkpoint equivalence, checkpoint/prediction/activation archived hashes, preparation manifest digest and actual prepared held-out inputs/labels all match.

Architecture was verified against the independent canonical reference: Input2 → learned Linear2→4 → reshape[N,2,2] → tracked BatchNorm1d2 with learned scale/offset → flatten4 → learned Linear4→2 → Output. Saved state contains exact expected projection/head/normalization learned parameters, float32[channel] running means/variances and int64 scalar counter; values are dense/finite, variance nonnegative and the counter valid. No arbitrary integer learned state was admitted.

Independent expected forward arithmetic uses plain Python scalar affine operations and saved running statistics: subtract each channel mean, divide by the square root of saved variance plus epsilon, apply learned channel scale/offset, then evaluate the saved head. Expected values never invoke engine forward or a native normalization module. All 16 actual captured held-out rows matched independent class decisions and saved prediction classes. Every captured stage's available value prefix matched the scalar reference.

Saved portable model was imported and evaluated on both CPU and MPS in full and one-observation batches; all model state/buffers remained unchanged. CPU maximum scalar error 1.11635360955e-07, MPS 2.24968228379e-07; maximum batch-one/full difference 1.19209289551e-07. Maximum captured stage error 1.37073640438e-07. Actual saved inference.py was executed in two fresh host subprocesses with the actual held-out JSON inputs and batch1/full; both terminal0, all classes match, maximum CLI scalar error 1.50307466029e-07. Tolerance absolute2e-5/relative2e-5. The parent verifier itself completed terminal0 on its first actual-artifact run; no failing fixture assumption or numerical threshold repair occurred.

Limits: one actual native canonical tracked BatchNorm1d export only. Other four operators and non-default controls have separate host/runtime/lesson evidence; this receipt does not convert the sample into general current-stat BatchNorm microbatch invariance, native parity for all units, clean-machine acceptance or release readiness. Root's UI/History observations and final scoped Swift replay remain separate evidence.

Evidence:

- `.local/verify-stateful-normalization-native-bn1.py` SHA-256 `03e4825ef92476aff1d8900cacce6f32c02905c944158e65be69e88186ee120a`.
- `.local/stateful-normalization-native-bn1-verification.log` SHA-256 `0dbacbd53220bb6181ac790d2a672c336841aa7ae3a5ec6aa3e2f3e29c02a859`.
- `.local/stateful-normalization-native-bn1-20261008/verification.json` SHA-256 `f194258d7bf71417bf324e94ab5053c90a75cbfeff4bd993f9a4abadb856f686`.
- `.local/stateful-normalization-native-bn1-20261008/cli-1.log` SHA-256 `4948cf1a08cbcfd6722d13e2d56e4754c70aa4d562fe79585cdb7a4add8429d6`.
- `.local/stateful-normalization-native-bn1-20261008/cli-16.log` SHA-256 `627cdf7075e6261349ccef03a1e3b5f091ab32b41f3e6c40e89f1594f2081513`.

Actual archived model manifest SHA-256 `e2bc9e087edac8117c40fa875911c8774732da1ce253e7c1ef9287f61f29a6bb`; preparation manifest SHA-256 `f400d534dfb6d6c959ce9b3fd54fa4c330ca29987b7ed4f40da297c001f4d49b`. Package file identities/hashes and complete typed state shapes are retained in verification.json.
