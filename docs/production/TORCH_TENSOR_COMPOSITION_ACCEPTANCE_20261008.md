# Tensor composition acceptance · 2026-10-08

Frozen finite acceptance for `pytorch.torch.cat`, `pytorch.torch.add`, `pytorch.torch.stack` and `pytorch.torch.permute`. Each owns forward, input-gradients and shape-contract modes through one authored bundle. These parameterless blocks use surrounding learned layers for checkpoint/trained export. All88 reviewed units remain the owner delivery target; this receipt does not establish release acceptance.

## Reviewed component contracts

| Component | Configuration | Independently assessed lesson |
|---|---|---|
| Concatenate | Signed sample direction, default−1; ordered A then B | `assessment.concat-sample-axis` |
| Add | Finite signed source multiplier, default1; optional sample-only broadcasting, defaultfalse | `assessment.add-shared-pattern` |
| Stack | Signed insertion direction, default−1; equal sources create a size2 sample direction | `assessment.stack-new-direction` |
| Permute | Complete unique direction ordering, default[0,2,1]; observations first | `assessment.permute-sample-order` |

Dense production float32 uses a separate nonempty observation batch, full rank2–8; Stack inputs rank2–7. Exact source/output allocation is bounded at100million elements before allocation. Dtypes, devices and batches match. Batch direction0 cannot be concatenated/stacked/reordered or broadcast. Add broadcasting right-aligns sample directions only; both source batches remain identical. Alpha0 still validates both sources. Nonfinite, scalar, sparse/ragged/empty, unknown settings, boolean/fractional axes and shape mismatches refuse. Inputs are not mutated.

Int64 values/types/shape are preserved on tested CPU/MPS paths without an integer-gradient claim. Integer Add permits only historical alpha1/equal-shape behavior. Permute normalizes signed aliases, requires every direction exactly once with observation direction first, and reorders values rather than reshaping. Concatenate/Stack have two ordered input ports; larger compositions use explicit chains, not arbitrary arity.

Historical Concatenate/Add schema1/2 and empty configurations normalize to exact defaults; their legacy-equivalent semantic/cache/checkpoint identities are retained. Nondefault settings remain identity-bearing. New Stack/Permute require schema2. Per-operator runtime admission compares exact authored descriptors; absent/stale discovery grants no support, including legacy Concat/Add graphs. Existing tensor-layout descriptor remains unchanged.

## Actual automated verification

Scoped worker157 independent configuration/engine checks, authoring561 and new lesson worker81 checks pass with no skips. Runtime covers actual CPU/MPS values and weighted influence on both ordered inputs, sample broadcasting, permutation value ordering, signed aliases, integer outputs/indices, allocation/rank/dtype/refusal boundaries, portable exports and surrounding learned training/checkpoint behavior. Scalar references do not call the operation being tested. [Runtime](TORCH_TENSOR_COMPOSITION_RUNTIME_20261008.md), [authoring](TORCH_TENSOR_COMPOSITION_AUTHORING_20261008.md) and [learning](TORCH_TENSOR_COMPOSITION_LEARNING_20261008.md) retain exact scoped commands and limits.

Core configuration semantic RED preceded actual strict implementation. Integrated Swift58 checks passed, including all four real guided/practice/private worker lifecycles and cold receipts, wrong-valid Add settings→measured failure→repair, metadata/artwork and exact discovery/admission. Native findings then produced actual lesson-polish RED39assertion failures across4methods; scoped33Swift checks passed after concise controls, actionable missing construction and reactive readiness fixes. No grading or private projection guard was weakened.

Prior31 accepted source-bound Python replay:3,521passed, no skips, three existing warnings,59files. Final106-suite Swift replay:115Core+200App=315passed, zero failures/skips; LayerNorm real-runtime opt-in enabled. Earlier interrupted RED and normal terminal315/3RED are recorded honestly. Explicit synthetic fixtures now supply independently authored Concat/Add descriptors; stale/missing discovery negatives remain absent. Exact inventory assertions include the real four new examples and previously shipped convolution lessons. Prior semantic assertions remain. [Replay](TORCH_TENSOR_COMPOSITION_PRIOR_REPLAY_20261008.md). Generator drift and diff whitespace checks pass.

## Lessons, icons and native app evidence

Four approachable no-code tasks explain arrays, directions, observations, pattern reuse and how small source changes affect outputs. Guided controls are concise; definitions and optional mathematics remain separate. Distinct runnable captured-data classifiers and independent seeded guided/practice/private grading ship for every member. Productionfloat32 and assessmentfloat64 remain distinct. Original approved Concatenate/Add icons are retained; Stack's original generated blue diamond and Permute's purple rectangular reorder artwork are shared by palette/canvas. Unedited prompts/source/hash provenance: `docs/design/component-icons-v2/tensor-composition-provenance-20261008.json`.

Native Stack lesson project90A5470F-4716-4CD9-8DEA-37BE9457C705 assembled two Inputs/Stack/Output using ordered ports. Default direction−1 produced an incorrect target shape and no measured success. Direction0 refused with an observation-axis message; repaired direction2 inferred[N,2,2,3] and passed all three criteria in guided, practice and private modes. Private mode withheld measured fixtures. Final candidate cold restore retained direction2, source connections and verified private receipt with all criteria passed. Concise controls and ready assessment were visible without a graph edit. Learn search Stack now groups its lab/example under Tensors & linear algebra. Original Stack art and the new Permute palette art were observed in their blue/purple groups. Theme repetition moves to final UAT.

Prepared Stack project5AD80A83-DA45-416F-8E1B-DC181CEE9EFD trained three epochs on configured MPS. Logical run6579324A-6326-4D12-A09F-8AD321CF1D0A, requestdbcf010b79eaec3e2e4d25cfef9b856d81f3909678ce1ac0f6aec0f4719e95dd, revision1. History verified30artifacts, six computed preparation stages and saved source. Native export saved `.local/KwipML-model-6579324A` with weights, source, strict helper, prepared-input contract and hashes. Raw rows/fitted preparation are excluded; external callers must reproduce preparation.

Independent saved-weight scalar branches→Stackdim2→flatten→head across all16 held-out observations agree with strict CPU export within3.95442444567351e−7; all classes match. Inference helper/direct parity is0; batch1/full error≤2.980232238769531e−7. Package hashes, capture identity and archive model/checkpoint bytes match; synthetic CLI exits0. Captured previews retain eight values per stage, not full arrays. Teaching-set metrics do not imply generalization. [Saved export](TORCH_TENSOR_COMPOSITION_NATIVE_EXPORT_20261008.md) records exact hashes and reproducibility.

## Final packaged confirmation and limits

Final canonical109-worker payload integrity, same-Xcode watchdog identity, copied worker hashes, local signing and deep strict package checks passed. Final executable SHA256 `71f8320586a6b22e5d946c4657308f49f331480b7ddb59ac90dfc152516f5924`; log `.local/tensor-composition-final-packaged-build.log`. Final candidate quit/relaunch and saved private assessment confirmation passed. No distribution signing or clean-machine claim.

Representative native Stack construction/training/export supplements automated real-worker/export/assessment checks for all four; it does not claim all four manually trained. Formal release theme/accessibility/clean-machine UAT and supported-runtime distribution remain open. All declared finite bundle checks are satisfied. Source-bound accounting must preserve the prior31 semantic rows and17 frozen receipts while binding these four; that result belongs in separate accounting evidence. This receipt is immutable after creation. Next batches, DATA-02, remaining curriculum and UAT/DIST remain separate gates.
