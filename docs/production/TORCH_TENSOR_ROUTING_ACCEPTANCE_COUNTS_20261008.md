# Tensor routing acceptance count clarification · 2026-10-08

Frozen additive clarification to TORCH_TENSOR_ROUTING_ACCEPTANCE_20261008.md; its bytes remain unchanged. Only the reporting of the final workflow-projection test counts is corrected. Operator semantics, tests, runtime, native observations, exports and acceptance are unchanged.

The workflow-port regression log reports **8 Core checks, one skipped, zero failures**, and **21 App checks, zero failures**:29 reported,28 passed,1 optional benchmark skipped. The original receipt incorrectly calls the7 passing Core checks “reported.” The separate enabled benchmark replay reports **8 Core checks passed, zero skipped**, rather than7. It also records512 descriptors/4096 reads in1.636ms. These counts are independently visible in the actual logs; no result is fabricated or reclassified.

Actual final prior-source replay remains169Core+216App=385passed, zero skipped; broad Python remains4947passed. This clarification changes no framework numerator or production-gate claim.

Workflow regression log SHA256784d3743c96b65d3e29c0283f9b507fd03bde285f0fbfb339147738d79163183.
Enabled benchmark log SHA256733c781ab51d7ce3033df16764cc9527fee74bf56b9cdbd5e9031492628fbb58.
Original frozen receipt SHA256b7eebbed25fe0e4a5bbae8cf8a56603d66ea982e65332a62237a5a5c5059f896.
