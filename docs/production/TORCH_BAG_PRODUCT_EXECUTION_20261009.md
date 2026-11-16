# Typed-bag product execution adapter · 2026-10-09

Scoped implementation evidence only. This does not publish a runtime descriptor, register components, admit captured artifacts, implement training/History consumers, or grant capability coverage. Root owns those integration gates.

## Input and inspection contract

`ProductExecutionAdapter` admits one actual `input`, `index_input` or `bag_input` source. Ordinary tensor input attributes retain their prior behavior; its new `input_contract` is exactly `{dtype, sampleShape}`. Bag input sets legacy `input_shape`, `input_dtype` and `features` to `None`: packed item coordinates have no dense observation-first sample shape or single packet dtype.

Bag `input_contract` is fresh JSON-compatible metadata with `kind:bagPacket`, `schemaVersion:1`, actual `sourceID`, the four normalized source configuration values, and named `ports` containing each exact role and dtype. Index/offset coordinates are int64; optional item weights are float32. It does not carry measured tensor values or invented sample shapes. Mutating a returned metadata dictionary cannot change the retained admission contract.

`validate_input(x, *, check_finite=True, device=None, context='Model input')` returns the original packet after the real `execute_bag_source` geometry/control checks and explicit device validation. A supplied device permits prepared CPU partition validation before movement to an MPS model; the default remains the model device. Item weights retain their live autograd connection. The trusted hot path still validates packet geometry and finite item weights.

`observation_count(x)` validates on the model device and returns whole bags, including empty packed bags. Root training code may separately use the model-input helper to count admitted CPU partitions. Public forward and inspected ordinary outputs use bag count rather than item count or `len(packet)`.

Captured source stages require exactly the declared named ports, independently match source identity/configuration/coordinate roles/dtypes, reconstruct the typed packet without flattening, and validate actual packet geometry/device/observation count. They never access a packed coordinate `.sample`. Later changes to source controls, role layouts or dtype metadata refuse before the native model forward. This qualification checks the current model's source contract, not artifact provenance or permission to execute an arbitrary captured source.

## Actual RED and GREEN

New tests first ran against the existing importable adapter: **29 failed**, terminal1, `.local/embedding-bag-product-execution-red.log`. Bag cases encountered the unsupported-source `StopIteration`; the ordinary metadata case encountered the absent new API. These are executed failures, not missing-file or compiler failures.

After implementation, the new tests plus existing product-execution tests terminated0: **48 passed, zero skipped**, 0.94 seconds, `.local/embedding-bag-product-execution-green.log`. The single warning is the existing upstream nested-tensor prototype warning. Installed PyTorch exercised actual CPU and MPS cases; no device case was replaced with a mock.

The new matrix covers fixed/packed and weighted/unweighted graphs, literal independent outputs and every table/item-weight influence, mixed empty bags and all-empty packed observations, original tensor identity in inspected named ports, malformed packet dtype/offset/weight geometry/finiteness/terminal flags, declared source limit, captured source config/role/dtype/source identity tamper refused before forward, captured source missing/extra ports/row mismatch, explicit CPU-before-MPS validation and wrong-device refusal, fresh metadata isolation and trusted-path finite checks. Existing ordinary source/output/parameter/stage validation regression remains passing. Test registries are injected fixtures; no production registry or discovery publication is implied.

The final source hashes below were recorded after the observed GREEN. No before/after source-freeze manifest was captured for that command; it is not claimed retroactively. Root can bind these sources into its later frozen integration replay.

| Source or local receipt | SHA256 |
| --- | --- |
| `runner/product_execution.py` | `e4c9190d4fe20b6c2c28b99a154c7b4ff82c13ea74b83ce6316683501d078a8d` |
| `runner/tests/test_bag_product_execution.py` | `79419c48d702610bae8b75e8a8f5b0221793667ed5f396b974fb30a84cfa9535` |
| `.local/embedding-bag-product-execution-red.log` | `1b2a6e4ba33ccfbfc91bc59ee9994e319c10a03906f4a405998270ec0f884f6f` |
| `.local/embedding-bag-product-execution-green.log` | `ad2c65239b6310d80a5a40662214a608a0fa4772025eef8a417c5722e0ad35b3` |
