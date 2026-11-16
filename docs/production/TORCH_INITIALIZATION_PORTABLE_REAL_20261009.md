# Initialization portable learned inference — 2026-10-09

This finite test closes the selected-helper inference gap identified by a read-only review of `InitializationHistoryExportTests`, `test_initialization_portable_inventory.py` and prior initialization engine tests. It is host evidence, not native UI export or capability acceptance.

## Actual result

```sh
.venv/bin/python -m pytest -q runner/tests/test_initialization_portable_real.py
```

**1 passed, zero failures/skips, 1.89s** on the first actual baseline execution. The existing production implementation already satisfied this missing proof. No production defect, production repair or invented RED is claimed. Log `.local/initialization-portable-real-baseline.log`, SHA256 `b3ae31be6e4a8db7a6dc34a6435e6b00ad06346c7a88e3edfe15604cec7c473e`.

## Measured boundaries

The test trains a genuine CPU two-input/two-output affine graph for two epochs with 64 source observations, batch16 and SGD rate0.05. A selected initialization recipe uses seed42, normal weight center0.15/spread0.2, and zero bias. Both saved learned weights and bias are asserted to differ from independent starting values. The starting-weight oracle derives the owner seed from literal UTF8 SHA256 JSON and uses the independently probed tiny-array CPUfloat64 normal primitive rounded to float32; it does not invoke the production initializer.

The actual generated model, checkpoint and optional initializer are copied beside the production `inference.py` and an explicit input contract into a separate package. Every file is pinned; the selected qualified policy is retained in the package manifest. The original run is deleted before inference.

All 11 prepared rows are compared against plain Python scalar affine arithmetic from actual saved learned weights. Production reusable inference passes batches1,3,11; actual standalone inference CLI passes batch1 and batch11. The loaded state equals every saved model tensor exactly, global CPU RNG remains unchanged, and the optional initializer file is unchanged. Predictions use relative tolerance1e-6/absolute1e-7 for float32 affine accumulation.

Explicit fresh use is tested separately: importing the real helper does not change a fresh model; calling its initialization function explicitly reproduces independently calculated starting values, which remain distinct from learned weights. This is a fresh-model operation, not continuation or loaded inference.

A consistently rehashed initializer containing a top-level exception still permits correct inference. Therefore this real inference path does not import or apply that helper. Changing it without updating its inventory causes integrity refusal before model loading. Package hashes demonstrate internal consistency, not third-party authorship or permission to execute arbitrary exports.

## Pins and limits

- New test `runner/tests/test_initialization_portable_real.py`: `acf7a1ed3b43262828a7c7c66825b967e77c9ba5447ba783e003315114420a59`.
- Production inference resource, read only: `79d768fa618a3d2a720fb6d465ad0b9d018fcfcc0cd268b828c6972459fa075f`.
- Engine, read only: `4c3eea4f36461292b1070c7e5cb0cdc6a29fbd46ac817def1dd9ba55ac221d44`.
- Initialization helper, read only: `75f5bf637106d32481850b9d049d562d33df2d095585d426854f3d6a452a8253`.
- Machine pins: `.local/initialization-portable-real-pins.json`.

Only this new test and receipt were added. This proof covers production convenience inference on CPU. CPU/MPS fresh initialization and continuation retain their separate prior runtime evidence; this test does not imply an MPS convenience-inference feature or native UI execution. No Swift, build, package, catalog, tracker or previous frozen receipt was modified.
