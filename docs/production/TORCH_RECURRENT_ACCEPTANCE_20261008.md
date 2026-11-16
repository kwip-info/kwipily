# Recurrent family finite acceptance — 2026-10-08

**Frozen finite acceptance.** RNN, LSTM, GRU, RNNCell, LSTMCell and GRUCell meet the declared bounded v1 exit checks. All four original reviewed modes remain: inference, training, input-gradients and state-roundtrip. This accepts six bundles; final accounting is guarded separately. Prior59 accepted units remain unchanged in meaning. Desired resulting checkpoint65/88,23 remaining; no claim of consolidated UAT or release readiness.

## Delivered contract

Six configurable generated bundles share configuration-v1 and schema2 float32 graph execution. Explicit zero or supplied starting-state connections, named sequence/hidden/cell outputs, observation-first state layout and independent per-observation memory are enforced. Cells expose out/hidden as one physical alias. Native layer/direction/gate parameter names/order/shapes are validated; hidden state is never a learned parameter. Allocation/recurrence-work budgets apply before native allocation. Sequence dropout acts between layers during training; single-layer nonzero dropout is refused. RNN activation, bias, layers, direction and LSTM projection controls are supported. Projected LSTM is CPU-only: requested MPS is refused actionably before allocation; no fallback or silent implicit state carry. See frozen `TORCH_RECURRENT_CLOSURE.md`.

Weights/checkpoints and standalone Python model/inference export preserve real recurrent operators. CPU exact epoch-boundary continuation restores optimizer/random state; MPS supports explicit weights-only restart and refuses exact continuation. Bidirectional state carry is not claimed equivalent to whole-sequence reverse processing. Six original image-generated rose-family icons are copied into app resources with source/hash/alpha provenance; palette and canvas share each asset. Six no-code lessons ship with matching genuine editable classifier flows, learner-owned connections/configuration, checker-owned varied values/learned tables, independent scalar references and influence checks. Public guidance defines memory and influence; private grading withholds numerical answer tables and cold-reverifies saved evidence.

## Measured exit checks

- Runtime215 new checks passed0fail/0skip: real CPU/MPS scalar/gradient comparisons, named states, optional controls, strict learned-state admission, budgets, actual export, CPU continuation, MPS refusal/reset and projected CPU proof. Adjacent831 checks passed separately.
- Bundle authoring173 scoped checks and1384 shared declaration checks passed; prior63 ordered component definitions retain their exact hash, with69 total after the six additive definitions.
- Actual lesson Python136 checks passed; source-qualified App checks include all six guided/practice/private/cold worker receipts, wrong-bias failure/repair, editor draft/Undo, exact capability admission/revocation and independent learned-state ownership. Final new App8 checks passed0fail/0skip in53.507s.
- Canonical12 genuine Core example CPU/MPS runs passed, with48 training/16 held-out rows, independent scalar/named-state/full/micro/helper inference, strict checkpoint pins, six CPU exact continuations and six MPS refusal/weights-only resets. Maximum canonical error9.661e-8.
- Prior59 bound runner replay:5313 passed0fail/0skip across76 prior test files;167 frozen host inputs,72 prior receipts and110 original images remain unchanged. Final Core965 reported tests/5 explicit historical skips/0fail; prior shared App51/1 benchmark skip/0fail.
- Native testing caught a missing named-preview verifier branch. Actual19-failure RED and9-failure cell-alias RED led to strict source/output/state/capture/alias admission. Final five new +three routing tests passed; subsequent adjacent actual-worker App66/1 benchmark skip/0fail in27.399s. Earlier compiler-only failures and an invocation with absent worker environment are not represented as semantic or actual-worker passes.
- Corrected signed132-file candidate passed root-only native LSTM guided/private/cold grading, actual preparation/training, distinct sequence/hidden/cell tables, verified History, actual UI package save and cold History verification. Logical runB1BF7D99-5525-4411-B3B1-B228EFF956F8; project43E0D7C3-27E5-4E14-ABA5-C41337101520; requestc51c4f7e9843989f07c647d74ccd41a22d817c424fcceb7a7cc07e5aad867428; revision0;30 saved artifacts.
- Independent actual UI-export proof verifies archive/source/model/checkpoint/file pins, actual MPS training, CPU/MPS full/micro logits/classes and all named states, strict saved Adam state and actual inference CLI batches1/16. Maximum named-state error8.397e-8; actual export/archive bytes unchanged. A copied scheduler verifier version assertion was corrected to the observed implicit-recipe capture1/compiler1; its failed verifier log is preserved separately and was not a product failure.

## Evidence and practical limits

Scoped authoring/runtime/learning/canonical/prior-replay/root-integration/training-presentation/native/native-export receipts are retained under `docs/production/TORCH_RECURRENT_*`. Source and artifact hashes below pin actual finite proof outputs; local binary archives/caches remain excluded from Git. Native representative coverage complements all-six scientific and worker checks. In-flight animation, full machine/theme/accessibility/novice UAT and distribution stay in their own release tasks; no absent captured values or broader model quality are inferred.

| Actual proof artifact | SHA-256 |
|---|---|
| `.local/recurrent-complete-final-green.log` | `bdd60407df0862c96ced097be574b9c5ee3da46891c7bdcc819e5cdab59637c5` |
| `.local/recurrent-prior-runner-result.json` | `270ef4fb6d25c6f109f11c19b0dc71125ef6552253ef7de21ccad94d56a7604e` |
| `.local/recurrent-canonical-host-20261008-final/receipt.json` | `ddedae3e708a8a4319fd8a9bb58bbb2f62586593c117513f16ef05e8f5eedfa9` |
| `.local/recurrent-final-core-shared-app-green.log` | `c7d1c90c3c3a1bb425d4f93ba2e39bf0241e6ed1202eaf8458292b0d92742453` |
| `.local/recurrent-final-presentation-actual-shared-app-green.log` | `5c1f63f463731bf6fdf3b217a6b8ccb5479cfb1c3ba03729f6bdf9e29527aeb7` |
| `.local/recurrent-final-new-app-green.log` | `41d323d661b8fd5af4424fe8aa03bbfce6686e6d88de07d058260ecae9b49b3b` |
| `.local/recurrent-native-export-20261008-final/verification.json` | `06dee9c60987b0382196b635fdd400a407d8710cf56e7dbb50468ee71e1b8d6e` |
| `.local/recurrent-preview-fixed-packaged-build.log` | `7e6eda525e48b0e28158950b4b10d8a21e83e7db3438f28efbe892ae72214a1f` |
