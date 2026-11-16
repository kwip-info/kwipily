# Recurrent canonical workflow host proof — 2026-10-08

## Scope and provenance

This is an independent **host** proof of the six genuine prepared-classifier Core factories, not a native interaction, App Store release, or accepted-coverage claim. `RecurrentCanonicalExampleTests` compiles the actual public factories with compiler 6, verifies one configured zero-state recurrent member with implicit Adam and no scheduler, and emits the original workflow plus compiled model, dataflow, and settings only under `KWIPML_RECURRENT_CANONICAL_FIXTURES`. The emitted original workflow is schema 1; compilation performs the supported migration rather than manufacturing a newer source document.

The proof uses the existing sealed two-feature preparation archive:

`/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/056FF2E4-2C6A-4E0E-81B5-4EC1A1B6A7B0/F3F3695C-71E6-453A-ADB4-58A09EB6F928`

The loader requires its recorded manifest SHA, dataflow digest, qualified training/validation outputs, feature columns, target column, float32 input contract and two-feature shape. Every actual emitted Core dataflow must match that preparation digest. Source pins cover the runtime engine, recurrent helper, lifecycle verifier, preparation loader, immutable proof script, genuine Core factory and portable inference helper. The scalar oracle is an exact frozen plain-Python copy of `scalar_reference` and `leaves` from runtime test source SHA `8a81605bd56dd21688882b4e6644c79f3ba208ebdab8a374650718729c4ae118`; the copied functions SHA is `6f9294a276105c43f3017d538ddc3965b3b99abf738d995346a8c7894a3c862d`. This removes unrelated mutable test additions from the scientific reference while preserving its provenance.

## Checks

Each RNN, LSTM, GRU, RNN Cell, LSTM Cell and GRU Cell classifier trains for its actual factory settings on CPU and MPS. The independent expected values use plain Python scalar matrix products, sigmoid/tanh gates and recurrent updates from `runner/tests/test_recurrent_runtime.py`, followed by a scalar learned linear head. No PyTorch recurrent operation, runtime adapter or exported helper computes expected values.

For every trained checkpoint the proof checks completed runtime identity, exact model snapshot/hash, checkpoint SHA/size, metric-history SHA and checkpoint contract, copied preparation manifest, implicit Adam policy and absence of an invented scheduler. It measures exported full and one-row inference on both CPU and MPS against those scalar logits; separately measures the portable native layer's `out`, `hidden` and LSTM `cell` against the scalar state oracle. Class choices must match. Genuine portable model source must not import the product runtime.

A host-created portable inventory uses genuine worker `model.py` and `checkpoint.pt` plus the application's actual `inference.py` and explicit already-prepared input contract. CLI batches of one and all held-out rows must match scalar logits/classes. A changed checkpoint must be refused even after a successful earlier load. These directories are host-created proof packages; they are not represented as native UI exports.

CPU continuation compares a full run against a one-epoch checkpoint followed by exact resume: model weights, optimizer state, RNG and complete metrics are exactly equal. MPS exact resume must refuse; explicit fine-tuning must report weights-only lineage, epoch zero and reset optimizer/RNG/epoch/metrics. This canonical zero-state matrix supplements the runtime agent's unit/state-mode/projection matrix; it does not duplicate a broader projected-CPU matrix.

## Actual chronology

- First attempt refused before training because the proof mistakenly assumed original workflow schema 7. The genuine factory source is schema 1; the proof assertion was corrected to its actual contract. Preserved `.local/recurrent-canonical-host-schema-assumption-red.log`.
- Second attempt completed all **12** training-device cases, including scalar, named-state, portable inference, CLI, integrity refusal and continuation checks, but its final source-freeze guard correctly refused an overall passing receipt because the independently imported runtime-test oracle source changed during the run. Production source pins did not change. `.local/recurrent-canonical-host-20261008-r2/receipt.json` remains `passed-so-far`, and `.local/recurrent-canonical-host-r2.log` records the terminal failure. No source hash was silently refreshed and no credit is claimed from that partial receipt.
- Final fixed-source replay: **terminal exit 0, all 12 cases passed**, no skips or warnings. Every production/script pin was unchanged across capture. Actual PyTorch `2.14.1`, 48 training rows and 16 held-out rows. Maximum scalar error across full/micro logits, named recurrent outputs and CLI logits: `9.661341637823284e-08`.

Final evidence:

| Evidence | SHA-256 |
| --- | --- |
| `.local/recurrent-canonical-host-20261008-final/receipt.json` | `ddedae3e708a8a4319fd8a9bb58bbb2f62586593c117513f16ef05e8f5eedfa9` |
| `.local/recurrent-canonical-host-final.log` | `df2d7c51124760849be0c09ba3bb3ec193aec18ace0e000361f0c296804cf2ee` |
| `.local/verify-recurrent-canonical-workflows.py` | `0de7b0d0e818e76439c8ea635e4708a3d084d56aa5b7624220137a747ddd515c` |

The final JSON retains each fixture hash, real run ID, graph/model/checkpoint/runtime pins, named parameter inventory, all measured errors and continuation manifest SHA. All six CPU resumes match complete weights/optimizer/RNG/metrics exactly. All six MPS exact resumes refuse and all six explicit fine-tunes report the reviewed reset lineage. Native UI export/cold History and accepted accounting remain parent-owned gates; this passed host receipt grants no coverage by itself.

## Commands

Core emission is parent-owned and serialized. This agent did not run Swift, a package build, native UI or signed Python.

```sh
.venv/bin/python .local/verify-recurrent-canonical-workflows.py \
  --fixtures .local/recurrent-canonical-examples \
  --archive '/Users/developer/Library/Containers/info.kwip.KwipML.POC/Data/Library/Application Support/KwipML/workflow-runs/056FF2E4-2C6A-4E0E-81B5-4EC1A1B6A7B0/F3F3695C-71E6-453A-ADB4-58A09EB6F928' \
  --output .local/recurrent-canonical-host-20261008-final
```

The proof refuses to replace an existing output directory and refuses its final passed status if any pinned source changes while evidence is captured.
