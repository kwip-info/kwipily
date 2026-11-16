# Load model checkpoint · October 18, 2026

Implementation branch: `codex/load-model-import-export`, based on `3f3371c4`.
Owner review checkpoint only; no develop merge, version change or publication.
Maintained contract: [Load model](LOAD_MODEL.md).

## Delivered and preserved

Saved checkpoint is presented as Load model, with the same internal ID and empty Saved run
configuration. Local folder and explicitly pinned Hub sources add reviewed GPT-2/Llama causal
adapters, model-matched token/ID/mask preview, bounded generation and safe Hugging Face folder
export. Existing package/checkpoint saves, private saved-run bindings and History remain readable.
Toolbar, Information, shared Practice rail and MCP discovery use the consolidated identity.

Source hashing/copying, storage admission and large output verification run off the main actor.
Progress remains local to the inspector; workspace reservations prevent competing training while
keeping selection available. Terminal operations remove owned source copies. New private model
snapshots are retained only after binding publication; duplicate immutable assets reuse identity.

## Automated evidence

These are affected batches, not a full-suite acceptance claim. Repeated batches are not additive.
All paths below are ignored checkout-local evidence.

- `.local/load-model-final-affected-v2.log`: 36 App tests passed, including prior saved-model,
  publication, MCP, diagnostics and lesson behavior; corresponding Core/network batches passed.
- `.local/load-model-final-retention-check.log`: final three real integration tests passed in
  13.182s: isolated import/tokenize/generate/export/reopen, reservations/cancellation/cleanup,
  schema20 coexistence with classifier training and independently verified saved History,
  and changed-byte/symlink refusals.
- `.local/load-model-cancellation-final.log`: four Core and four App checks passed, including
  captured tensor identity across different layers in one run.
- `.local/load-model-real-weights-red.log` / `-green.log`: actual public safetensors download
  failed before the CDN correction; afterward three network and four Core checks passed.
  Truncated/oversized responses, wrong ranges and authentication failures are refused.
- `.local/load-model-backend-expanded.log`: eight backend tests passed, including upstream
  GPT-2/Llama offline export/reload and unsafe model/shape/vocabulary cases.
- `.local/load-model-packaging-checks-v2.log`: 20 authorization, notice and helper packaging
  checks passed. Final helper-specific repeat: two passed. A notice inventory is not complete
  commercial redistribution clearance.
- `.local/load-model-review-build.log`: coherent private development build and deep/strict
  signature verification passed. Owner explicitly approved the existing Developer ID for the
  same POC app/NetworkService identities after an ad-hoc/container mismatch. No container reset or
  sandbox bypass, new key, production replacement or notarization.

## Observed native checks

Explicit folder selection loaded a synthetic safe GPT-2 model; token preview returned the actual
IDs/masks. Connected generation emitted three tokens into the standard output table. Native Save
model exported a complete safe folder, independently reloaded offline with upstream libraries:
identical logits and IDs (`.local/load-model-native-export-verification.json`). Saved project,
private selection and captured generation output survived quit/reopen.

The actual Hub path resolved and downloaded seven reviewed files from
`hf-internal-testing/tiny-random-gpt2`, commit `71034c5d8bde858ff824298bdedc65515b97d2b9`.
The isolated worker verified GPT-2, 1,000 vocabulary IDs and 512-token context; native token
preview completed. Initial helper and CDN failures were visibly attributed to the component and
Diagnostics, then corrected. Private/gated-repository sign-in was not exercised with real credentials.

Information → Read the lesson → Open on canvas opened the shared Load model Practice rail and
real next-token workflow. Forward navigation, cards and source instructions rendered. Whole-window
zoom and corner resize with rail/Results visible preserved layout; Fit recomputed its camera.
Canvas scroll and Fit remained functional. This is observed behavior, not input-to-paint timing.

## Matched interaction measurements

Unchanged baseline `.local/interaction-profiles/load-model-before-64/report.json` and two final
captures `load-model-review-after-64/report.json` / `load-model-review-repeat-64/report.json` use
the same release harness, machine, 64 nodes and eight iterations. Both qualified with zero dropped
samples and working-changes hash `24845c0878fc7581c9bdecf1488ba9e5ea424f7eb8e2a6fbd24542965fcdc065`.
Only receipt/index documentation, trailing-whitespace cleanup and the helper-signing packaging
correction below follow those application-source checks.

| Host-layout p95, ms | Baseline | First | Repeat |
|---|---:|---:|---:|
| Pan drag | 23.504 | 22.905 | 23.517 |
| Wheel zoom | 26.797 | 26.328 | 27.422 |
| Camera restore | 46.849 | 44.436 | 47.089 |

Insertion/deletion queue-delay p95 varied 67.316 → 80.258 → 56.563ms. Retain both comparisons;
do not select the faster run or claim a universal speedup. These host-wall observations show
similar camera/layout costs in this bounded fixture; they do not measure GPU, frames, large
imported-model workloads or physical input latency. Earlier source-changing/stopped captures are
not qualified evidence. No other app, model operation or test competed during the final captures.

A later reopen exposed a separate MCP container prompt because the first signing option also
signed unrelated helpers. Scoped secinit evidence identified MCPService (old ad hoc, new Developer
ID); owner cancelled the ambiguous prompt. Packaging was narrowed to leave MCP and Database
helpers on their original ad-hoc identities, with only app/NetworkService using the approved
existing Developer ID. No container or consent records were altered. The corrected
private app passed deep/strict verification. Independent codesign metadata verified MCP/Database
remain ad hoc and NetworkService retains team MT68X9Z6NQ
(`.local/load-model-helper-identity-verification.json`). On actual reopen, Settings showed the
existing MCP service Running without changing its settings; native Hub resolution also completed.
Scoped secinit readback during this corrected reopen contained no new KWIP container prompt.
Earlier successful Hub/runtime checks remain valid; no new distribution acceptance is implied.

## Current boundaries

Reviewed safe folders require config.json, tokenizer.json and safetensors. GGUF/LM Studio
quantization, arbitrary/custom remote-code models, imported fine-tuning/Evaluate, Hub upload and
Laya need separate adapters. Imported generation has verified private output/provenance and
reopening, but not the standard trained-run History list or standalone imported-generation replay.
Whole-workflow Code refuses unsupported terminals; supported safe model export remains available.
Full suites and release/runtime redistribution audits remain separate integration gates.
