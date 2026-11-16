# Load model

Load model consolidates the existing Saved checkpoint placement. Its internal component ID
`workflow.frozenModel`, frozen-model edge kind and private saved-run references remain stable.
An empty configuration retains Saved run. External sources use versioned LoadModelConfiguration
and workflow schema20; changing the label or position does not change model identity.

## Sources and current capabilities

This source contract includes unreleased development adapters. It does not change the shipped1.3.2 installer.

| Source | Acquisition | Supported consumers |
|---|---|---|
| Saved run | Existing independently verified project History selection | Existing Evaluate, Generate text where compatible, and package/checkpoint Save model |
| Local folder | Explicit folder selection at any accessible location; no download | Reviewed task adapters: GPT-2/Llama causal; Whisper speech recognition; AST sound classification; ModernBERT Laya typed decisions. Consumers depend on verified task capabilities. |
| Hugging Face | Explicit repository/revision resolution, immutable commit selection and bounded download | Same reviewed adapters as Local folder |

Local folder can reuse a model installed by another application when that folder contains the
supported format. GGUF/LM Studio quantized models, pickle weights, remote repository code,
quantization adapters and arbitrary architectures are not supported. Reviewed causal fine-tuning and the separate CPU [Laya typed-decision adapter](../dataflow/TYPED_DECISIONS.md) have explicit task contracts.
No model executes during project opening, configuration rendering, panning or zooming.
Imported task-specific Train/Evaluate and standard-table consumers are defined in the [fine-tuning contract](../EXTERNAL_MODEL_FINE_TUNING.md); inference-only speech/audio adapters do not inherit training. Whole-workflow Code explicitly refuses
unsupported model/evaluation terminals; use the supported local model export. These are capability
boundaries, not a claim that importing a file delivers every upstream library feature.

## Model and tokenizer authority

Causal folders contain config.json, tokenizer.json and safetensors weights, optionally a
validated shard index and reviewed tokenizer/configuration/license data files. Other tasks require their own complete matching processor or reviewed nested encoder/tokenizer profile; Laya never substitutes a causal tokenizer preview. Native snapshotting
uses held directory descriptors, regular-file checks, exclusive writes, cancellation, streamed
SHA256 and identity/time rechecks. Admission is bounded to128 files and2GB. The worker verifies
exact pinned bytes, allowed architecture/head, contiguous matching vocabulary IDs, tensor types,
keys and shapes before reporting supported capabilities. Loading stays offline with safe weights
and repository code disabled. CPU/MPS generation observes the project's captured time, storage
and memory policies; a preflight memory estimate is an admission bound, not measured peak RSS.

Tokenizer preview retains model ID ordering. It exposes maximum tokens, explicit refusal or
truncation, padding and padding side, with real token/ID/mask cells. The inspector shows at most32
sample tokens and labels truncated display. Generation receives unpadded real prompt tokens,
uses an explicit context policy and the model's special-token IDs, and emits the standard table
with source prompt, generated text, token sequence, count and stop reason. No tokenizer recipe
node or hidden vocabulary refitting is introduced.

## Network and private state

Hub traffic uses the existing authenticated network-only XPC helper. The app and offline worker
receive no new network entitlement. The DEBUG helper has finite read-only access to its
exact enclosing POC bundle to inspect the actual peer signature; production helper rights remain
unchanged. macOS may require owner consent to reuse that helper’s own container after an ad-hoc
development rebuild. Do not delete its container or bypass that system access decision.
An existing Developer ID container cannot silently become an ad-hoc identity: with explicit owner
authorization, `KWIPML_DEVELOPMENT_SIGNING_IDENTITY` makes `scripts/build-app.sh` sign the private
POC app and NetworkService consistently using that existing identity. Other helpers retain their
prior ad-hoc identities; changing those would needlessly request new container consent. The default stays ad hoc; no
notarization or publication is implied. Helper/app signatures are verified after assembly.
Requests accept repository IDs and reviewed data filenames,
not arbitrary URLs. Immutable commit URLs, bounded ranges and exact response lengths are checked;
redirects are HTTPS allowlisted and bearer authorization is stripped on foreign CDN hosts.
The explicit list includes the current US AWS/GCP download edges documented by
[Hugging Face](https://huggingface.co/docs/hub/models-downloading); arbitrary subdomains remain
refused. The real-network regression downloads the pinned public safetensors file as well as
metadata, because metadata alone does not exercise CDN redirects.
Credentials entered by the owner stay in Keychain. Paths, credentials and private model-selection
bindings are not workflow configuration or MCP catalog data. Opening a project never downloads.
MCP discovers the same typed configuration schemas and describes the private source-binding step;
it does not invent an alternate model catalog or bypass the app's source grant.

Verified snapshots are project-owned private assets with content identity deduplication. Download
staging and task source copies are removed through their held ownership on terminal paths. Managed
operation outputs/receipts remain subject to storage accounting; no blanket cache purge or silent
model deletion occurs. Imported generation records pin their own model/source and captured workflow,
without pretending to be an app-trained checkpoint. Existing saved-run History and replay remain
unchanged. The latest verified imported output can be restored without reopening the original folder.

## Export

Save model retains portable package and checkpoint formats for app-trained models. Imported models
use Hugging Face folder: original dtype, safe weights, configuration, matching tokenizer, retained
notices and a kwip-manifest.json file inventory/provenance. Shard size controls file layout, not
weights. A single tensor may exceed a chosen shard target. Publication uses an explicit parent
folder grant, a validated new child name, held scope, verified streaming and exclusive final commit.
Existing destinations are never overwritten. Publishing to the Hub is outside this slice.

## Interaction standards

Source copy/hash/model work occurs outside SwiftUI rendering. Download writes and large file
verification run off the main actor. Only the narrow inspector observes per-file progress; the
canvas receives semantic job transitions and completed bindings. Tensor rendering can reuse an
immutable captured identity, never just a node ID; selection, accent and capture revision must
invalidate presentation. Heatmap scaling computes its bounded range once per render.

Use affected model, saved-run, publication, authorization, discovery and networking checks for a
component checkpoint. Compare matched release interaction fixtures with no competing work, and
check actual canvas/Results/lesson-rail/window drags separately. Host-wall timings are not GPU,
display-frame or pointer-to-paint measurements. Full suites remain required at develop integration.
