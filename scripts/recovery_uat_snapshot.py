#!/usr/bin/env python3
"""Create a separately sealed diagnostic source tree for native recovery UAT.

This does not build, launch, or alter an installed application. Build the resulting
snapshot using its own runtime_package.py and build-packaged-app.sh; their normal
current-worker integrity checks then bind to this recorded diagnostic source.
"""
import argparse
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

_brand_spec = importlib.util.spec_from_file_location('recovery_uat_brand', Path(__file__).with_name('product_brand.py'))
_brand = importlib.util.module_from_spec(_brand_spec)
_brand_spec.loader.exec_module(_brand)

PREPARATION_GATE = '''
def _recovery_uat_pause(output, node_id, emit):
    # Diagnostic snapshot only: actual supervisor/native watchdog remain active.
    import time
    import uuid
    arm = output.parent / 'recovery-uat-arm'
    if not arm.exists() and not arm.is_symlink():
        return
    raw = bounded_regular_read(arm, 256)
    nonce = raw.decode('ascii')
    if str(uuid.UUID(nonce)).lower() != nonce.lower():
        raise ValueError('Invalid recovery UAT nonce')
    emit({'event': 'recoveryUATPaused', 'stage': 'duringPreparation',
          'nodeID': node_id, 'nonce': nonce})
    deadline = time.monotonic() + 300
    resume = output.parent / 'recovery-uat-resume'
    while time.monotonic() < deadline:
        if resume.exists() or resume.is_symlink():
            if bounded_regular_read(resume, 256) == raw:
                return
        time.sleep(0.05)
    raise RuntimeError('Recovery UAT preparation pause timed out')

'''


def overlay_worker(source):
    anchors = ['def run_dataflow(', '    def node_complete(node_id,ports):',
               '        emit(event)\n    try:']
    if '_recovery_uat_pause' in source or any(source.count(a) != 1 for a in anchors):
        raise ValueError('Diagnostic overlay requires the exact reviewed preparation anchors')
    source = source.replace(anchors[0], PREPARATION_GATE + anchors[0], 1)
    source = source.replace(anchors[1], '    recovery_uat_consumed = False\n' + anchors[1] + '\n        nonlocal recovery_uat_consumed', 1)
    source = source.replace(anchors[2], '        emit(event)\n        if not recovery_uat_consumed:\n            recovery_uat_consumed = True\n            _recovery_uat_pause(output, node_id, emit)\n    try:', 1)
    compile(source, 'diagnostic/dataflow_cli.py', 'exec')
    return source


def overlay_build(source):
    anchor = f'<key>CFBundleDisplayName</key><string>{_brand.NAME} Development</string>'
    display_setter = 'Set :CFBundleDisplayName $PRODUCT_DISPLAY_NAME Development'
    name_setter = 'Set :CFBundleName $PRODUCT_DISPLAY_NAME Development'
    if source.count('swift build --package-path') != 2:
        raise ValueError('Unexpected build command; review diagnostic flag injection')
    if source.count(anchor) != 1 or source.count(display_setter) != 1 or source.count(name_setter) != 1:
        raise ValueError('Unexpected display identity; review diagnostic naming')
    return source.replace('swift build --package-path', 'swift build -Xswiftc -DKWIPML_RECOVERY_UAT --package-path').replace(
        anchor, f'<key>CFBundleDisplayName</key><string>{_brand.NAME} Recovery UAT</string>').replace(
        display_setter, 'Set :CFBundleDisplayName $PRODUCT_DISPLAY_NAME Recovery UAT').replace(
        name_setter, 'Set :CFBundleName $PRODUCT_DISPLAY_NAME Recovery UAT')


def create_snapshot(destination):
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        raise ValueError('Commit reviewed source before creating the diagnostic snapshot')
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Snapshot destination must not exist')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    archive = subprocess.check_output(['git', 'archive', '--format=tar', revision], cwd=root)
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(destination, filter='data')
    edits = {}
    def change(relative, transform):
        path = destination / relative
        before = path.read_bytes()
        after = transform(before.decode()).encode()
        path.write_bytes(after)
        edits[relative] = {'baselineSHA256': hashlib.sha256(before).hexdigest(),
                           'diagnosticSHA256': hashlib.sha256(after).hexdigest()}
    change('runner/dataflow_cli.py', overlay_worker)
    change('scripts/build-app.sh', overlay_build)
    record = {'kind': 'recoveryUATDiagnosticSnapshot', 'baselineRevision': revision,
              'purpose': 'REC-08F/G/H native crash timing only; not production acceptance by itself',
              'changes': edits}
    (destination / 'RECOVERY_UAT_SNAPSHOT.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(create_snapshot(args.destination), indent=2))
