# Standard optimizer archive repair · 2026-10-08

Native prepared AdamW Train failed before worker launch because WorkflowArchivePlan still admitted compiler versions1–4 and omitted the selected optimizer. The existing schema6/compiler5 mapping alone was insufficient. Root reproduced this with genuine compiled Core examples: three initial methods failed six assertions (three unexpected errors), recorded in `.local/optimizer-archive-projection-red.log`.

The bounded repair admits compiler5 and records an optional canonical optimizer recipe plus its qualified control source. Historical compiler versions reject explicit optimizer plans instead of silently discarding them; their implicit plans omit the new field and retain canonical bytes. Decoding normalizes equivalent JSON integer/decimal scalar representations, requires every declared configuration control, rejects unknown fields through the archive codec, and checks the effective execution rate against the recipe. Matching compares algorithm, controls and source rather than only rate. No model-forward contract or historical version pair was broadened.

Four new methods cover all five genuine examples, recipe/rate/source changes, historical3/3,4/4,5/4 byte roundtrips and injection refusal, compiler5 implicit Adam, incomplete/unknown recipe fields, wrong source domain and conflicting effective rate. Actual scoped replay executed61 methods:60 passed, one older actual-History fixture failed because it explicitly disabled discovery and did not await required execution-contract verification. Root added the existing actual runtime-discovery helper before its run, without granting fixture support or weakening admission. Final Archive/History replay passed13 methods with zero failures/skips, including the expanded missing-control/conflicting-rate negatives. Across these runs all61 scoped methods pass; this is not a single full61-method final-source rerun. Native replay remains pending; this document is not coverage acceptance.

- `.local/optimizer-archive-repair-green.log` SHA256 `a961eb4f50d08093d1e829ae2b37adb1efaa1771d3fdddebc221187c47dfb9f5`.

- `.local/optimizer-archive-history-final-green.log` SHA256 `120f5a22884fbd030ee3b1f8f21d8b76fba3cc7f8352e39b9ffe4c1192bb677a`.

## Native completed-outcome inventory boundary

The repaired compiler5 candidate completed actual AdamW training, but saving its History outcome refused the unrecognized training_recipe.py path. Its training artifacts and capture are retained; no completed saved History receipt is claimed for that run. A new outcome test reproduced the exact admission error (one unexpected failure), then a narrow allowlist entry admitted only training/artifacts/training_recipe.py with the existing8192-byte limit. Tests admit1/8192 bytes and reject8193 bytes and unknown filenames. Final Archive/Codec/History/optimizer-artifact replay passed30 methods, zero failures/skips; no other phase artifact admission changed. Native replay of the next signed candidate is pending.

- `.local/optimizer-outcome-recipe-red.log` SHA256 `ce791e944814d2affb4ddc29bd492d2760172f23b32c6557b36dcb9aaca39de1`.

- `.local/optimizer-outcome-recipe-final-green.log` SHA256 `259c1a8708238f944c922bad016e1f5fba73288797b77c50f9373e27ff359f07`.
