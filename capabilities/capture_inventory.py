#!/usr/bin/env python3
"""Extract official documentation symbol indexes. No framework import is needed.

Sources must be downloaded explicitly; this script never fetches or executes content.
Sphinx snapshots enumerate documented Python objects, including methods and private APIs.
TensorFlow's captured navigation is a page-link inventory, not runtime introspection.
"""
import argparse
from collections import defaultdict
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
import zlib

class TFLinks(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=set()
    def handle_starttag(self,tag,attrs):
        if tag!='a':return
        value=dict(attrs).get('href','')
        parsed=urlparse(urljoin('https://www.tensorflow.org',value))
        if parsed.netloc=='www.tensorflow.org' and re.fullmatch(r'/api_docs/python/tf(?:/[A-Za-z0-9_]+)*',parsed.path):
            self.links.add(parsed.path)

def sphinx(path,base,prefix):
    parts=path.read_bytes().split(b'\n',4)
    if parts[0]!=b'# Sphinx inventory version 2':raise ValueError('Unsupported Sphinx inventory')
    version=parts[2].decode().removeprefix('# Version: ')
    records={}
    for line in zlib.decompress(parts[4]).decode().splitlines():
        match=re.match(r'^(\S+) (py:\S+) (-?\d+) (\S+) (.*)$',line)
        if not match:continue
        symbol,kind,_,uri,_=match.groups()
        if not (symbol==prefix or symbol.startswith(prefix+'.')):continue
        if uri.endswith('$'):uri=uri[:-1]+symbol
        records[symbol]={'symbol':symbol,'kind':kind,'doc_url':urljoin(base,uri)}
    return version,list(records.values())

def tensorflow(path):
    text=path.read_text(); parser=TFLinks();parser.feed(text)
    version=re.search(r'TensorFlow v([0-9.]+)',text).group(1)
    return version,[{'symbol':link.removeprefix('/api_docs/python/').replace('/','.'),'kind':'documented-page','doc_url':'https://www.tensorflow.org'+link} for link in sorted(parser.links)]

def normalize(records):
    groups=defaultdict(list)
    for row in records:groups[row['doc_url']].append(row['symbol'])
    for row in records:
        candidates=groups[row['doc_url']]
        fragment=urlparse(row['doc_url']).fragment
        canonical=fragment if fragment in candidates else min(candidates,key=lambda value:(len(value),value))
        row['canonical_symbol']=canonical
        row['capability_ids']=[]
        if canonical!=row['symbol']:
            disposition,reason='alias','Same canonical documentation target; alias does not create another capability unit.'
        elif any(part.startswith('_') for part in row['symbol'].split('.')):
            disposition,reason='excluded','Documented private/internal or special-method symbol. Retained for audit; not an independent user operation in proposed v1.'
        elif row['symbol'].startswith(('tf.compat.v1','tf.raw_ops')):
            disposition,reason='excluded','Legacy compatibility or low-level raw operator API excluded from proposed visual v1; retained in the full captured inventory.'
        elif row['kind'] in ('py:module','py:method','py:attribute','py:property','py:data','py:exception'):
            disposition,reason='infrastructure','Namespace, object lifecycle/member or metadata surface; reviewed with its owning operation, not counted as an independent visual primitive.'
        else:
            disposition,reason='unsupported','Documented API not yet selected/implemented; retained for future family expansion and denominator review.'
        row.update(disposition=disposition,reason=reason)
    return sorted(records,key=lambda row:row['symbol'])

def main():
    root=Path(__file__).resolve().parent
    sources=[('pytorch','pytorch.inv','https://docs.pytorch.org/docs/2.14/objects.inv','https://docs.pytorch.org/docs/2.14/','torch'),
             ('sklearn','sklearn.inv','https://scikit-learn.org/1.9/objects.inv','https://scikit-learn.org/1.9/','sklearn'),
             ('tensorflow','tensorflow-layers.html','https://www.tensorflow.org/api_docs/python/tf/keras/layers',None,None)]
    metadata=[]
    (root/'inventories/v1').mkdir(parents=True,exist_ok=True)
    for framework,filename,url,base,prefix in sources:
        path=root/'snapshots'/filename
        version,records=sphinx(path,base,prefix) if base else tensorflow(path)
        destination=root/'inventories/v1'/f'{framework}.json'
        destination.write_text(json.dumps(normalize(records),indent=2)+'\n')
        metadata.append({'framework':framework,'source_url':url,'captured_on':'2026-10-05','declared_documentation_version':version,
                         'raw_snapshot':str(path.relative_to(root.parent)),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'inventory_file':str(destination.relative_to(root.parent)),'symbol_count':len(records),
                         'parser':'capabilities/capture_inventory.py','notes':'Captured documentation surface, not a claim of complete runtime API enumeration or implementation.'})
        print(framework,version,len(records))
    (root/'sources.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':main()
