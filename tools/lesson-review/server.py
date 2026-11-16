#!/usr/bin/env python3
"""Loopback-only lesson copy review. Decisions never edit product source or run code."""
import argparse
import hashlib
import json
import re
import secrets
import threading
import textwrap
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CATALOG = ROOT / 'Sources/KwipMLApp/LessonContentCatalog.swift'
REFERENCE = ROOT / 'Sources/KwipMLCore/ModelOutputPracticeExample.swift'
GUIDE = ROOT / 'Sources/KwipMLApp/ClassifierLessonGuide.swift'
PLAN = ROOT / 'Sources/KwipMLCore/WorkflowGuidedBuildPlan.swift'
INPUT_COPY = ROOT / 'Sources/KwipMLApp/ModelInputLessonContent.swift'
GUIDED_COPY = ROOT / 'Sources/KwipMLApp/GuidedLessonCopy.swift'
AFFINE_REFERENCE = ROOT / 'Sources/KwipMLCore/AffineReference.swift'
AFFINE_VIEW = ROOT / 'Sources/KwipMLApp/AffineLearningView.swift'
AFFINE_TABLES = ROOT / 'Sources/KwipMLApp/AffineTaskReferenceView.swift'
RAIL = ROOT / 'Sources/KwipMLApp/LessonRailView.swift'
AFFINE_STORE = ROOT / 'Sources/KwipMLApp/AffineLearningStore.swift'
AFFINE_TASKS = ROOT / 'Sources/KwipMLCore/AffineLessonTasks.swift'
AFFINE_LAUNCH = ROOT / 'Sources/KwipMLApp/ProjectWindowRegistry.swift'
AFFINE_PARAMETERS = ROOT / 'Sources/KwipMLApp/MatrixParameterEditor.swift'
SOURCE_INPUTS = (CATALOG,GUIDE,PLAN,INPUT_COPY,GUIDED_COPY,AFFINE_REFERENCE,AFFINE_VIEW,AFFINE_TABLES,RAIL,AFFINE_STORE,AFFINE_TASKS,AFFINE_LAUNCH,AFFINE_PARAMETERS)
COMPOSITION_COPY = ROOT / 'Sources/KwipMLApp/AffineCompositionLessonCopy.swift'
RECOVERY_REFERENCE = ROOT / 'Sources/KwipMLApp/AffineRecoveryReferenceView.swift'
SOURCE_INPUTS += (COMPOSITION_COPY,RECOVERY_REFERENCE)
RECOVERY_COPY = ROOT / 'Sources/KwipMLApp/AffineRecoveryLessonCopy.swift'
BACKWARD_REFERENCE = ROOT / 'Sources/KwipMLCore/AffineBackwardChainReference.swift'
SOURCE_INPUTS += (RECOVERY_COPY,BACKWARD_REFERENCE)
BACKWARD_COPY = ROOT / 'Sources/KwipMLApp/AffineBackwardLessonCopy.swift'
FROZEN_REFERENCE = ROOT / 'Sources/KwipMLCore/FrozenStandardizationReference.swift'
SOURCE_INPUTS += (BACKWARD_COPY,FROZEN_REFERENCE)
FROZEN_COPY = ROOT / 'Sources/KwipMLApp/AffineFrozenLessonCopy.swift'
RELU_REFERENCE = ROOT / 'Sources/KwipMLCore/ReLULessonReference.swift'
SOURCE_INPUTS += (FROZEN_COPY,RELU_REFERENCE)
RELU_COPY = ROOT / 'Sources/KwipMLApp/AffineReLULessonCopy.swift'
SOURCE_INPUTS += (RELU_COPY,)
ACTIVE_ENTRY = 'affine-relu'
STATE = ROOT / '.local/lesson-review'
MAX_BODY = 80_000
LOCK = threading.Lock()
TOKEN = secrets.token_urlsafe(32)
STRING = r'"((?:[^"\\]|\\.)*)"'
STEP = re.compile(r'LessonContent\.step\(' + STRING + r'\s*,\s*' + STRING + r'\s*,\s*' + STRING + r'(?:,kind:\.(\w+))?')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def decode_steps(section):
    result = []
    for match in STEP.finditer(section):
        ident, title, text = [json.loads('"' + v + '"') for v in match.groups()[:3]]
        result.append(dict(id=ident, title=title, text=text, kind=match.group(4) or 'explanation'))
    return result


def read_current(root=ROOT, entry_id=None):
    entry_id = entry_id or ACTIVE_ENTRY
    def read(path):
        return (root / path.relative_to(ROOT)).read_text()
    source = read(CATALOG)
    if entry_id == 'model-input':
        copy = read(INPUT_COPY)
        steps = decode_steps(copy)
        if [s['id'] for s in steps] != ['boundary', 'meaning', 'follow', 'reflect']:
            raise ValueError('Model input source structure changed; refresh its adapter.')
        return dict(steps=steps, sourceHash=digest(copy), sourcePath=str(INPUT_COPY.relative_to(ROOT)))
    if entry_id == 'affine-composition':
        return affine_current(root)
    if entry_id == 'affine-recovery':
        return recovery_current(root)
    if entry_id == 'affine-backward':
        return backward_current(root)
    if entry_id == 'affine-frozen':
        return frozen_current(root)
    if entry_id == 'affine-relu':
        return relu_current(root)
    if entry_id != 'guided':
        raise ValueError('No reviewed source adapter for this lesson.')
    start = source.index('        if entryID=="guided" {')
    end = source.index('        if let entry=componentEntries', start)
    overview = source[start:end]
    plan, guide = read(PLAN), read(GUIDE)
    if 'version:Int=5' not in plan:
        raise ValueError('Guided course version changed; reconcile the adapter.')
    calls = re.compile(r'step\(\.(data|model|control),' + STRING + r',' + STRING + r',' + STRING + r'\)')
    stages, order = {}, []
    for match in calls.finditer(plan):
        domain = match.group(1)
        ident, title, fallback = [json.loads('"' + v + '"') for v in match.groups()[1:]]
        key = (domain, ident)
        if key not in stages:
            order.append(key)
        stages[key] = dict(title=title, fallback=fallback)
    # Explicit version5 admission removes the old model carrier and mapping stages.
    order = [key for key in order if key not in [('control', 'model'), ('control', 'mapping')]]
    expected = ['observations', 'cast', 'split', 'fit', 'training', 'validation', 'input', 'branch-a', 'activation-a', 'branch-b', 'activation-b', 'merge', 'classifier', 'output', 'train']
    if [key[1] for key in order] != expected:
        raise ValueError('Guided placement order changed; reconcile the adapter.')
    copy = read(GUIDED_COPY)
    if 'GuidedLessonCopy.stages[nodeID]' not in guide or 'deck(GuidedLessonCopy.overview)' not in overview:
        raise ValueError('Guided copy wiring changed; reconcile the adapter.')
    authored = re.compile(STRING + r':\.init\(title:' + STRING + r',text:' + STRING + r'\)')
    staged = {}
    for match in authored.finditer(copy):
        ident, title, text = [json.loads('"' + v + '"') for v in match.groups()]
        staged[ident] = dict(title=title, text=text)
    steps = []
    for index, (domain, ident) in enumerate(order):
        if ident not in staged:
            raise ValueError('Missing approved guided explanation for ' + ident)
        steps.append(dict(id='placement.' + str(index), title=staged[ident]['title'],
                          text=staged[ident]['text'], kind='canvasAction', nodeID=ident,
                          address=domain + '/' + ident, group='Guided canvas · 15 placement stages'))
    fallback = decode_steps(copy)
    if [s['id'] for s in fallback] != ['meet', 'inspect', 'repair', 'train']:
        raise ValueError('Guided overview source changed; reconcile the adapter.')
    for step in fallback:
        step['group'] = 'Library overview · 4 fallback sections'
    return dict(steps=steps + fallback, sourceHash=digest(plan + '\0' + guide + '\0' + copy + '\0' + overview),
                sourcePath=' + '.join(str(p.relative_to(ROOT)) for p in (GUIDED_COPY, PLAN, CATALOG)))


def literal(pattern, text):
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError('Assessment source structure changed; reconcile the adapter.')
    return json.loads('"' + match.group(1) + '"')


def source_text(root, path):
    return (root / path.relative_to(ROOT)).read_text()


def public_maps(root):
    reference=source_text(root,AFFINE_REFERENCE)
    maps=[]
    pattern=r'name: "(First map|Second map)", weight: LearningTensor\(shape: \[2, 2\], values: (\[[^\]]+\])\),\s*bias: LearningTensor\(shape: \[2\], values: (\[[^\]]+\])\)'
    for match in re.finditer(pattern,reference):
        values=json.loads(match.group(2));bias=json.loads(match.group(3))
        maps.append(dict(name=match.group(1),weight=[values[:2],values[2:]],bias=bias))
    if len(maps)!=2 or any(len(m['weight'][0])!=2 or len(m['weight'][1])!=2 or len(m['bias'])!=2 for m in maps):
        raise ValueError('Public map shapes changed; reconcile the adapter.')
    return maps


def affine_current(root):
    copy=source_text(root,COMPOSITION_COPY)
    steps=decode_steps(copy)
    if [step['id'] for step in steps]!=['goal','try.0','inspect']:
        raise ValueError('Composition rail copy changed; reconcile the adapter.')
    for step in steps:
        step['group']='Assessment rail · 3 sections'
    pattern=re.compile(STRING+r':\.init\(title:'+STRING+r',text:'+STRING+r'\)')
    refs={}
    for match in pattern.finditer(copy):
        ident,title,text=[json.loads('"'+value+'"') for value in match.groups()]
        refs[ident]=dict(id=ident,title=title,text=text)
    kinds={'reference.goal':'explanation','reference.workflow':'canvasAction','reference.maps':'visual','reference.criteria':'observation','reference.notation':'detail'}
    if list(refs)!=list(kinds):
        raise ValueError('Composition reference sections changed; reconcile the adapter.')
    steps += [{**refs[ident],'kind':kind,'group':'Task reference · 5 sections'} for ident,kind in kinds.items()]
    paths=(COMPOSITION_COPY,AFFINE_REFERENCE,AFFINE_VIEW,AFFINE_TABLES,RAIL,AFFINE_STORE,AFFINE_TASKS,AFFINE_LAUNCH,AFFINE_PARAMETERS)
    return dict(steps=steps,publicMaps=public_maps(root),sourceHash=digest('\0'.join(source_text(root,p) for p in paths)),sourcePath=str(COMPOSITION_COPY.relative_to(ROOT)))


def recovery_current(root):
    copy=source_text(root,RECOVERY_COPY)
    steps=decode_steps(copy)
    if [step['id'] for step in steps]!=['goal','try.0','inspect']:
        raise ValueError('Recovery rail copy changed; reconcile the adapter.')
    for step in steps: step['group']='Assessment rail · 3 sections'
    pattern=re.compile(STRING+r':\.init\(title:'+STRING+r',text:'+STRING+r'\)')
    refs={}
    for match in pattern.finditer(copy):
        ident,title,text=[json.loads('"'+value+'"') for value in match.groups()]
        refs[ident]=dict(id=ident,title=title,text=text)
    kinds={'reference.goal':'explanation','reference.workflow':'canvasAction','reference.probes':'visual','reference.criteria':'observation','reference.notation':'detail'}
    if list(refs)!=list(kinds): raise ValueError('Recovery reference sections changed; reconcile the adapter.')
    steps += [{**refs[ident],'kind':kind,'group':'Task reference · 5 sections'} for ident,kind in kinds.items()]
    tasks,reference=[source_text(root,p) for p in (AFFINE_TASKS,RECOVERY_REFERENCE)]
    probe_match=re.search(r'let probes = zip\(\["zero", "basis-0", "basis-1"\], (\[\[.*?\]\])\)\.map',tasks)
    if not probe_match: raise ValueError('Public probe layout changed; reconcile the adapter.')
    inputs=json.loads(probe_match.group(1))
    if inputs!=[[0,0],[1,0],[0,1]]: raise ValueError('Public probe inputs changed; reconcile the adapter.')
    probes=[dict(id=ident,title=literal(r'case "'+ident+r'": return '+STRING,reference),input=values) for ident,values in zip(['zero','basis-0','basis-1'],inputs)]
    paths=(RECOVERY_COPY,AFFINE_TASKS,RECOVERY_REFERENCE,AFFINE_VIEW,RAIL,AFFINE_STORE,AFFINE_LAUNCH,AFFINE_PARAMETERS)
    return dict(steps=steps,publicObservations=probes,projectSpecificValues=True,sourceHash=digest('\0'.join(source_text(root,p) for p in paths)),sourcePath=str(RECOVERY_COPY.relative_to(ROOT)))


def backward_current(root):
    copy=source_text(root,BACKWARD_COPY)
    steps=decode_steps(copy)
    if [step['id'] for step in steps]!=['goal','try.0','inspect']:
        raise ValueError('Backward rail copy changed; reconcile the adapter.')
    for step in steps: step['group']='Assessment rail · 3 sections'
    pattern=re.compile(STRING+r':\.init\(title:'+STRING+r',text:'+STRING+r'\)')
    refs={}
    for match in pattern.finditer(copy):
        ident,title,text=[json.loads('"'+value+'"') for value in match.groups()]
        refs[ident]=dict(id=ident,title=title,text=text)
    kinds={'reference.goal':'explanation','reference.workflow':'canvasAction','reference.maps':'visual','reference.meaning':'explanation','reference.criteria':'observation','reference.notation':'detail'}
    if list(refs)!=list(kinds): raise ValueError('Backward reference sections changed; reconcile the adapter.')
    steps += [{**refs[ident],'kind':kind,'group':'Task reference · 6 sections'} for ident,kind in kinds.items()]
    paths=(BACKWARD_COPY,BACKWARD_REFERENCE,AFFINE_REFERENCE,AFFINE_TASKS,AFFINE_VIEW,AFFINE_TABLES,RAIL,AFFINE_STORE,AFFINE_LAUNCH,AFFINE_PARAMETERS)
    return dict(steps=steps,publicMaps=public_maps(root),sourceHash=digest('\0'.join(source_text(root,p) for p in paths)),sourcePath=str(BACKWARD_COPY.relative_to(ROOT)))


def frozen_current(root):
    copy=source_text(root,FROZEN_COPY)
    steps=decode_steps(copy)
    if [x['id'] for x in steps]!=['goal','try.0','inspect']: raise ValueError('Frozen rail copy changed; reconcile the adapter.')
    for x in steps: x['group']='Assessment rail · 3 sections'
    refs={}
    for match in re.finditer(STRING+r':\.init\(title:'+STRING+r',text:'+STRING+r'\)',copy):
        ident,title,text=[json.loads('"'+v+'"') for v in match.groups()];refs[ident]=dict(id=ident,title=title,text=text)
    kinds={'reference.goal':'explanation','reference.workflow':'canvasAction','reference.statistics':'visual','reference.scope':'callout','reference.criteria':'observation','reference.notation':'detail'}
    if list(refs)!=list(kinds): raise ValueError('Frozen reference sections changed; reconcile the adapter.')
    steps += [{**refs[i],'kind':kind,'group':'Task reference · 6 sections'} for i,kind in kinds.items()]
    reference=source_text(root,FROZEN_REFERENCE)
    statistics={name:json.loads(re.search(r'public let '+name+r' = (\[[^\]]+\])',reference).group(1)) for name in ['means','scales']}
    paths=(FROZEN_COPY,FROZEN_REFERENCE,AFFINE_TASKS,AFFINE_VIEW,RAIL,AFFINE_STORE,AFFINE_LAUNCH,AFFINE_PARAMETERS)
    return dict(steps=steps,publicStatistics=statistics,sourceHash=digest('\0'.join(source_text(root,p) for p in paths)),sourcePath=str(FROZEN_COPY.relative_to(ROOT)))


def relu_current(root):
    copy=source_text(root,RELU_COPY)
    steps=decode_steps(copy)
    if [x['id'] for x in steps]!=['goal','try.0','inspect']: raise ValueError('ReLU rail copy changed; reconcile the adapter.')
    for x in steps: x['group']='Assessment rail · 3 sections'
    refs={}
    for match in re.finditer(STRING+r':\.init\(title:'+STRING+r',text:'+STRING+r'\)',copy):
        ident,title,text=[json.loads('"'+v+'"') for v in match.groups()];refs[ident]=dict(id=ident,title=title,text=text)
    kinds={'reference.goal':'explanation','reference.workflow':'canvasAction','reference.examples':'visual','reference.configure':'canvasAction','reference.inspect':'observation','reference.criteria':'observation','reference.notation':'detail'}
    if list(refs)!=list(kinds): raise ValueError('ReLU contextual sections changed; reconcile the adapter.')
    steps += [{**refs[i],'kind':kind,'group':'Task reference · 7 sections'} for i,kind in kinds.items()]
    examples=[dict(input=json.loads(a.replace('−','-')),output=json.loads(b)) for a,b in re.findall(r'Input (\[[^\]]+\]) (?:becomes|stays) (\[[^\]]+\])',refs['reference.examples']['text'])]
    if examples!=[dict(input=[-2,0],output=[0,0]),dict(input=[1,2],output=[1,2])]: raise ValueError('Authored public ReLU examples changed; reconcile the adapter.')
    paths=(RELU_COPY,RELU_REFERENCE,AFFINE_TASKS,AFFINE_VIEW,RAIL,AFFINE_STORE,AFFINE_LAUNCH,AFFINE_PARAMETERS)
    return dict(steps=steps,publicExamples=examples,sourceHash=digest('\0'.join(source_text(root,p) for p in paths)),sourcePath=str(RELU_COPY.relative_to(ROOT)))


def packet(root=ROOT):
    current = read_current(root)
    proposal = json.loads((HERE / ('proposals/' + ACTIVE_ENTRY + '.json')).read_text())
    if proposal['baselineHash'] != current['sourceHash']:
        raise ValueError('Current lesson changed. Reconcile the proposal before reviewing.')
    if [s['id'] for s in proposal['steps']] != [s['id'] for s in current['steps']]:
        raise ValueError('Proposal does not cover the exact current lesson sections.')
    revision = digest(json.dumps(proposal, sort_keys=True))
    return {**proposal, 'current': current, 'revision': revision}


def reference():
    text = REFERENCE.read_text()
    body = re.search(r'public static let instructions = """\n(.*?)\n    """', text, re.S)
    if not body:
        raise ValueError('Accepted Foundations reference unavailable.')
    return textwrap.dedent(body.group(1))


def decision(data, root=ROOT, state=STATE):
    with LOCK:
        active = packet(root)
        if data.get('revision') != active['revision'] or data.get('sourceHash') != active['current']['sourceHash']:
            raise ValueError('This page is stale. Reload before recording a decision.')
        if data.get('action') not in ('approve', 'request_changes'):
            raise ValueError('Choose an approval or revision request.')
        note = data.get('note', '')
        if not isinstance(note, str) or len(note) > 8000:
            raise ValueError('Review notes must be at most 8,000 characters.')
        if data['action'] == 'request_changes' and not note.strip():
            raise ValueError('Add a short note describing the changes you want.')
        state.mkdir(parents=True, exist_ok=True)
        path = state / (active['entryID'] + '-decision.json')
        record = dict(entryID=active['entryID'], action=data['action'], note=note.strip(),
                      revision=active['revision'], sourceHash=active['current']['sourceHash'],
                      recordedAt=datetime.now(timezone.utc).isoformat(),
                      scope='Wording review only; app integration and native chapter checks pending.',
                      proposal=active)
        if path.exists():
            prior = json.loads(path.read_text())
            if all(prior.get(k) == record.get(k) for k in ('action', 'note', 'revision', 'sourceHash')):
                return prior
            archive = state / (active['entryID'] + '-decision-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
            archive.write_text(path.read_text())
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
        temp.replace(path)
        return record


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # no review text or credentials in access logs

    def respond(self, status, body, mime='application/json'):
        raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', mime + '; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(raw)

    def trusted(self):
        return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

    def do_GET(self):
        if not self.trusted():
            return self.respond(403, {'error': 'Use the local 127.0.0.1 address.'})
        try:
            if self.path == '/api/review':
                payload = packet()
                payload['token'] = TOKEN
                payload['reference'] = reference()
                path = STATE / (payload['entryID'] + '-decision.json')
                payload['decision'] = json.loads(path.read_text()) if path.exists() else None
                return self.respond(200, payload)
            assets = {'/': ('index.html', 'text/html'), '/style.css': ('style.css', 'text/css'), '/app.js': ('app.js', 'text/javascript')}
            if self.path in assets:
                name, mime = assets[self.path]
                return self.respond(200, (HERE / name).read_bytes(), mime)
            return self.respond(404, {'error': 'Not found'})
        except (ValueError, OSError) as error:
            return self.respond(409, {'error': str(error)})

    def do_POST(self):
        if not self.trusted() or self.headers.get('Origin') != f'http://127.0.0.1:{self.server.server_port}' or self.headers.get('X-Review-Token') != TOKEN:
            return self.respond(403, {'error': 'Review actions must come from this local viewer.'})
        if self.path != '/api/decision':
            return self.respond(404, {'error': 'Not found'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                return self.respond(413, {'error': 'Review request is too large or empty.'})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Expected a review decision object.')
            return self.respond(200, decision(data))
        except (ValueError, OSError) as error:
            return self.respond(409, {'error': str(error)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8781)
    args = parser.parse_args()
    packet()  # refuse stale source before opening the listener
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Lesson review: http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever()
