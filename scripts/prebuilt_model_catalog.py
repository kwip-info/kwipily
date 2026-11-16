#!/usr/bin/env python3
"""Synchronize the trusted Core algorithm catalog to the isolated Python worker."""
import argparse
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
source=root/'Sources/KwipMLCore/Resources/prebuilt-models.json';target=root/'runner/prebuilt-models.json'
if args.check:
    if source.read_bytes()!=target.read_bytes():p.exit(1,'Prebuilt model catalog drift: run scripts/prebuilt_model_catalog.py\n')
else:target.write_bytes(source.read_bytes())
