# Objective export review copy · 2026-10-09

Native MSE export correctly saved a schema2 numeric model package, but review UI claimed two logits and a class index. This slice repairs that consumer-facing false statement using the already validated captured output contract, never width-based class inference, current registry or disk reparsing.

TrainedModelInputContract retains validated objective output geometry and dtype. Shared review renders numeric model values, actual [N,...] shape, and explains inference does not compute the training score or assign labels. Legacy initializer and exact two-logit sentence remain unchanged.

Actual `.local/loss-export-review-red.log`:2 methods,40 assertion failures, zero unexpected. Actual `.local/loss-export-review-green.log`:6 App methods passed, zero failures/skips,61.626s. Includes all8 compiled generic objective summary cases, legacy copy, typed input contracts and actual saved objective exports. Final log SHA256affed25446ea47a0627a2cdaee35bab0ef2946348c18bedbf0df7f7d07d9fcd5. Source pins `.local/loss-export-review-pins.json`. No native-copy recheck claimed yet; root is rebuilding candidate.
