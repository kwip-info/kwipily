#!/usr/bin/env python3
"""Real CPU backward/capture controls, not a trained-model performance corpus.

Creates a new local fixture directory. All gradients are obtained via autograd;
no parameter.grad assignments fabricate the measured observations.
"""
import argparse
import hashlib
import json
import platform
import sys
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.gradient_capture import GradientCapture

CASES = [
    ("healthy", None),
    ("missing", "gradient.missing.v1"),
    ("zero", "gradient.zero.v1"),
    ("nonfinite", "gradient.nonfinite.v1"),
    ("growth", "gradient.growth.v1"),
    ("relu-zero-sample", "activation.zero-sample.v1"),
    ("tanh-saturated-sample", "activation.saturation-sample.v1"),
    ("healthy-activations", None),
]


def generate(destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    records = []
    for name, expected in CASES:
        model = torch.nn.Module()
        # Match the real worker's named Linear parameter inventory. The strict
        # native codec intentionally refuses arbitrary Python parameter names.
        model.layers = torch.nn.ModuleDict({"control": torch.nn.Linear(10, 1, bias=False)})
        torch.nn.init.ones_(model.layers["control"].weight)
        if name == "missing":
            model.layers["unused"] = torch.nn.Linear(2, 1, bias=False)
        weight = model.layers["control"].weight
        run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "kwip:diagnostic-controls:" + name)).upper()
        epochs = 10 if name == "growth" else 3
        capture = GradientCapture(model, {"schemaVersion": 1, "everyEpochs": 1, "maxBatchesPerEpoch": 1, "maxCaptures": epochs},
                                  run_id=run_id, initial_epoch=0, final_epoch=epochs, device="cpu")
        for epoch in range(1, epochs + 1):
            model.zero_grad(set_to_none=True)
            if name == "zero":
                loss = (weight * 0).sum()
            elif name == "nonfinite":
                loss = (weight * torch.tensor(float("nan"))).sum()
            elif name == "growth":
                loss = (weight * (0.01 if epoch <= 5 else 2)).sum()
            else:
                loss = weight.square().sum() / 2
            loss.backward()
            capture.begin_epoch(epoch, total_batches=1, total_rows=1)
            capture.observe(batch_index=0, training_rows=1)
            capture.end_epoch()
        artifact = capture.artifact()
        data = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        filename = name + ".json"
        (destination / filename).write_bytes(data)
        activations = []
        if name == "relu-zero-sample":
            activations.append({"nodeID": "relu-control", "component": "relu", "values": torch.relu(-torch.ones(8)).tolist()})
        elif name == "tanh-saturated-sample":
            activations.append({"nodeID": "tanh-control", "component": "tanh", "values": torch.tanh(torch.full((8,), 10.0)).tolist()})
        elif name == "healthy-activations":
            activations.extend([
                {"nodeID": "relu-control", "component": "relu", "values": torch.relu(torch.ones(8)).tolist()},
                {"nodeID": "tanh-control", "component": "tanh", "values": torch.tanh(torch.full((8,), 0.5)).tolist()},
            ])
        records.append({"id": name, "file": filename, "sha256": hashlib.sha256(data).hexdigest(), "runID": run_id,
                        "expectedRules": [expected] if expected else [], "activations": activations})
    manifest = {"schemaVersion": 1, "corpusID": "kwip.pytorch-backward-controls.v1", "runtime": {"pytorch": torch.__version__, "python": platform.python_version()},
                "limitations": "Controlled backward-pass and tiny activation fixtures, not independent training runs, representative population calibration, or accuracy evidence.", "cases": records}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"directory": str(destination.resolve()), "cases": len(records), "corpusID": manifest["corpusID"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    generate(parser.parse_args().destination)
