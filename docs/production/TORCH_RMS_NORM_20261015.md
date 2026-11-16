# RMSNorm in the consolidated Normalization tool

Owner authorized2026-10-15. Additive `rms_norm` graph-schema2 primitive; existing
normalization IDs/defaults/projects remain unchanged. One family toolbar entry,
Conditioning artwork and inspector; Method RMS,1–7 trailing sample dimensions,
explicit normal float32 epsilon default1e-5 and optional elementwise scale.
No mean subtraction, bias, running statistics or batch reduction. Float32 CPU/MPS;
actual `nn.RMSNorm` runtime capability must be present. Learned scale starts at one.
This is the next Llama-style architecture primitive, not the full architecture.

## Verified development checkpoint

`codex/rms-normalization`, based on generation6e1efdb7. Shared Normalization toolbar,
olive Conditioning artwork and stacked inspector expose RMS. The existing shape grid,
Apply/Revert, epsilon, Learn scale and weight initialization/trainability controls are
reused. RMS omits bias and statistics controls. Changing a method refuses removal of
explicit authored bias/weight policies; existing primitive defaults and IDs are intact.
Layer↔RMS carries applicable shape/epsilon/scale settings, not checkpoint state.

Strict authored bundle, graph shape/dtype/configuration, real runtime discovery,
learned-parameter inventory, ordered checkpoint signature, native artifact verification
and generated model source all understand scale-only RMS. Explicit epsilon avoids
runtime-dependent dtype defaults. No new dependency or service was introduced.

## Evidence

- RED: `.local/rms-evidence/red.log` refuses the missing RMS primitive before implementation.
- Python family:474 passing cases in4.62s, `.local/rms-evidence/python-family-final.log`.
  Includes19 RMS-specific cases: independent non-centered formula and weighted gradients,
  CPU/MPS, trailing group/whole-sample scope, zero/constant inputs, actual-module finite
  differences, affine-off/scale-only state, train/eval/batch independence, standalone
  source replay, real training, exact CPU resume/model/optimizer/RNG/metrics, unchanged
  parent checkpoint, captured gradient inventory, cache reuse and stale-config refusal.
  Existing LayerNorm/stateful runtime and authored-bundle mutation checks remain passing.
- Swift affected batch:12 Core and15 App checks passed; the one new managed integration
  initially checked before asynchronous model verification finished. Its wait was fixed,
  and the real integration passed independently in3.526s (`swift-integration.log`).
  Total distinct final passing affected cases:28, no unresolved failure. Exact config,
  shape/parameter inventory, Codable round-trip, explicit policy preservation, descriptor
  mutation, selector/shape drafts, lessons, family toolbar and existing discovery covered.
- Native development build succeeded (`native-build.log`). Learn search/expanded explanation
  and Open RMS normalization practice worked. Actual RMS→Layer→RMS Apply retained wires
  and applicable fields; RMS omitted bias. Epsilon0 blocked Apply and Run Flow; Revert
  restored valid settings. The real canvas run completed with captured `[8,2,3]` float32
  output in the standard tensor viewer. Model Code showed `nn.RMSNorm([3],eps=1e-5,
  elementwise_affine=True)`. Save/quit/relaunch retained RMS/shape/epsilon/scale and canvas
  position. Practice project75F716CC-BF29-40A4-ABC7-24E806957033; native completed model
  F2F8918C-0FF… is local fixture evidence, not a model quality claim.
- Generated bundles/indexes and whitespace checked at closure. Logs/captured synthetic
  projects stay ignored; no hosted or exhaustive full-suite/release-baseline claim.

## Learning and remaining scope

Normalization overview now links the RMS bundle-authored lesson/practice, explaining
root mean square, no centering, trailing sample axes, epsilon and scale-only state,
with `[3,4]` worked values and an editable classifier comparison. It explicitly remains
unscored; historical LayerNorm and channel-normalization assessments are unchanged and
no new assessment completion is awarded.

Owner review is pending. No develop/main merge or release. RoPE, decoder blocks and
faithful tiny Llama architecture/demo remain later slices; this primitive alone does
not establish their correctness.
Reference: https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html
