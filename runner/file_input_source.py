"""Bounded File Input adapter: immutable manifests or explicit portable sources.

Managed operations consume only admitted manifest/member inventory. Original
folders are never inferred from metadata. Portable replay captures its explicit
external binding with the same deterministic no-follow collection semantics.
"""
from contextlib import contextmanager
import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

if __package__:
    from .file_collection_contract import parse_manifest,validate_manifest,MAX_MANIFEST_BYTES
    from .file_input_options import validate_configuration
    from .dataflow_disk import disk_scope,current_scope
else:
    from file_collection_contract import parse_manifest,validate_manifest,MAX_MANIFEST_BYTES
    from file_input_options import validate_configuration
    from dataflow_disk import disk_scope,current_scope

MAX_ENUMERATED_ENTRIES=10000
METADATA_TYPES={'file_id':'string','name':'string','relative_path':'string','extension':'string','detected_type':'string','byte_size':'int64','modified_at':'string','created_at':'string','content_sha256':'string'}
EXTENSIONS=json.loads(Path(__file__).with_name('file-input-options.json').read_text())['extensions']
READERS={extension:reader for reader,values in EXTENSIONS.items() for extension in values}


SELECTION_KEYS=('sourceMode','extensions','includeSubfolders','includeHidden','includePatterns','excludePatterns','symlinks','readMode','unknownFiles','maxFiles','maxDepth','maxFileBytes','maxTotalBytes','errorPolicy')

def selection_configuration(configuration):
    config=validate_configuration(configuration)
    return {key:config[key] for key in SELECTION_KEYS}


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def _hash(value):return hashlib.sha256(value).hexdigest()
def _identity(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
def _timestamp(value):return datetime.datetime.fromtimestamp(value,datetime.timezone.utc).isoformat().replace('+00:00','Z')

def _matches(path,patterns):
    return any(fnmatch.fnmatchcase(path,pattern) or (pattern.startswith('**/') and fnmatch.fnmatchcase(path,pattern[3:])) for pattern in patterns)


def _safe_root(path,directory=False):
    path=Path(path).absolute()
    # Resolve only the explicit binding's ancestors; the selected entry itself
    # and every descendant are opened O_NOFOLLOW below.
    path=path.parent.resolve(strict=True)/path.name
    flags=os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK
    if directory:flags|=os.O_DIRECTORY
    fd=os.open(path,flags)
    info=os.fstat(fd)
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        os.close(fd);raise ValueError('File Input requires a regular file or directory, never a link or special device.')
    return fd


def _read_member(parent,member,limit,check):
    parts=member['capturePath'].split('/')
    directory=os.open(parts[0],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
    try:fd=os.open(parts[1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
    finally:os.close(directory)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size!=member['bytes'] or before.st_size>limit:raise ValueError('Captured file differs from its regular-file byte authority.')
        blocks=[];count=0;hasher=hashlib.sha256()
        while True:
            check();block=os.read(fd,min(262144,limit+1-count))
            if not block:break
            count+=len(block)
            if count>limit:raise ValueError('File Input member exceeds its byte limit.')
            blocks.append(block);hasher.update(block)
        if count!=member['bytes'] or hasher.hexdigest()!=member['sha256'] or _identity(before)!=_identity(os.fstat(fd)):raise ValueError('Captured file changed or its SHA-256 does not match.')
        return b''.join(blocks)
    finally:os.close(fd)


@contextmanager
def _portable_capture(binding,config,check,preview,sampling):
    context=current_scope();budget=context[1];reserved=0
    with tempfile.TemporaryDirectory(prefix='.file-input-',dir=context[0]) as temporary:
        target=Path(temporary);(target/'members').mkdir(mode=0o700)
        selected=[];visited=0
        is_directory=config['sourceMode']=='directory'
        rootfd=_safe_root(binding,directory=is_directory)
        def collect(directory,relative='',depth=0):
            nonlocal visited
            with os.scandir(directory) as listing:
                names=[]
                for entry in listing:
                    check();visited+=1
                    if visited>MAX_ENUMERATED_ENTRIES:raise ValueError('File Input directory enumeration exceeds its bounded entry budget.')
                    names.append(entry.name)
            for name in sorted(names):
                check()
                if not config['includeHidden'] and name.startswith('.'):continue
                path=relative+'/'+name if relative else name
                info=os.stat(name,dir_fd=directory,follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    if config['symlinks']=='error':raise ValueError('File Input encountered a symbolic link; links are never followed.')
                    continue
                if stat.S_ISDIR(info.st_mode):
                    if not config['includeSubfolders']:continue
                    if depth>=config['maxDepth']:raise ValueError('File Input directory exceeds the configured depth.')
                    child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
                    try:collect(child,path,depth+1)
                    finally:os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    extension=Path(name).suffix[1:].lower()
                    if config['extensions'] and extension not in config['extensions']:continue
                    if not _matches(path,config['includePatterns']) or _matches(path,config['excludePatterns']):continue
                    selected.append((path,info))
                    if len(selected)>config['maxFiles']:raise ValueError('File Input selection exceeds the configured file count; narrow the filter.')
                else:raise ValueError('File Input encountered a special filesystem entry; only regular files are supported.')
        try:
            if is_directory:collect(rootfd)
            else:selected=[(Path(binding).name,os.fstat(rootfd))]
            selected.sort(key=lambda item:item[0])
            total=0;members=[]
            # Sampling chooses files before capture/extraction, not expanded units.
            if preview:selected=selected[:8]
            chosen={item[0] for item in _select_files(selected,sampling if not preview else None,True)}
            for relative,known in selected:
                check();extension=Path(relative).suffix[1:].lower();status='metadata'
                member={'relativePath':relative,'capturePath':None,'bytes':known.st_size,'sha256':None,'extension':extension,'status':status,'modifiedAt':_timestamp(known.st_mtime)}
                if hasattr(known,'st_birthtime'):member['createdAt']=_timestamp(known.st_birthtime)
                reader=READERS.get(extension)
                if reader=='audio' and config['readers']['audio']['format']=='disabled':reader=None
                if reader is None and config['unknownFiles']=='error':
                    member.update(status='error',error='Unsupported file type; choose metadata or skip for unknown files.')
                    members.append(member);continue
                if reader is None and config['unknownFiles']=='skip':continue
                if relative in chosen and config['readMode']=='content' and reader is not None and not (preview and reader in ('image','audio')):
                    if known.st_size>config['maxFileBytes']:
                        if config['errorPolicy']=='fail':raise ValueError('Selected file exceeds the configured per-file byte limit.')
                        member.update(status='error',error='Selected file exceeds the configured per-file byte limit.');members.append(member);continue
                    if total+known.st_size>config['maxTotalBytes']:raise ValueError('Selected files exceed the configured aggregate capture byte limit.')
                    current=rootfd;owned=[]
                    try:
                        if is_directory:
                            for part in relative.split('/')[:-1]:
                                child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=current);owned.append(child);current=child
                            fd=os.open(relative.split('/')[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=current)
                        else:fd=os.dup(rootfd)
                        try:
                            before=os.fstat(fd)
                            if not stat.S_ISREG(before.st_mode) or _identity(before)!=_identity(known):raise ValueError('Selected file changed before capture.')
                            capture='members/'+str(len(members)).zfill(6)+('.'+extension if extension else '')
                            hasher=hashlib.sha256();count=0
                            with (target/capture).open('xb') as sink:
                                os.chmod(target/capture,0o600)
                                while True:
                                    check();block=os.read(fd,262144)
                                    if not block:break
                                    count+=len(block)
                                    if count>config['maxFileBytes'] or total+count>config['maxTotalBytes']:raise ValueError('Selected file grew beyond its byte allowance.')
                                    budget.resize(reserved,reserved+len(block));reserved+=len(block)
                                    sink.write(block);hasher.update(block)
                            if count!=known.st_size or _identity(before)!=_identity(os.fstat(fd)):raise ValueError('Selected file changed while being captured.')
                            total+=count;member.update(capturePath=capture,sha256=hasher.hexdigest(),status='content')
                        finally:os.close(fd)
                    finally:
                        for descriptor in reversed(owned):os.close(descriptor)
                members.append(member)
            document={'schemaVersion':1,'kind':'kwipFileCollection','sourceMode':'directory' if is_directory else 'file','readMode':config['readMode'],'complete':not preview,'selection':selection_configuration(config),'members':members}
            raw=_canonical(validate_manifest(document));manifest=target/'manifest.json';manifest.write_bytes(raw)
            yield manifest,raw,document
        finally:
            os.close(rootfd)
            if reserved:budget.resize(reserved,0)


def _select_files(members,sampling,complete):
    if __package__:from .preparation_demand_configuration import normalize_sampling,sampling_row_limit,is_complete_sampling
    else:from preparation_demand_configuration import normalize_sampling,sampling_row_limit,is_complete_sampling
    policy=normalize_sampling(sampling)
    if not complete and (is_complete_sampling(policy) or policy['mode']=='percentage'):raise ValueError('File Input needs a complete captured collection; refresh the source before full execution.')
    limit=sampling_row_limit(policy,len(members))
    if limit is not None and not complete and limit>len(members):raise ValueError('Captured file sample is smaller than the requested sample; refresh the source.')
    return members if limit is None else members[:limit]


@contextmanager
def collection(binding,config,check,preview=False,sampling=None):
    explicit=isinstance(binding,dict) and set(binding)=={'manifest'}
    path=Path(binding['manifest'] if explicit else binding)
    managed=path.name=='manifest.json' and re.fullmatch('[0-9a-f]{64}',path.parent.name) is not None
    if explicit or managed:
        fd=_safe_root(path)
        try:
            info=os.fstat(fd)
            if info.st_size>MAX_MANIFEST_BYTES:raise ValueError('File collection manifest exceeds its byte limit.')
            raw=os.read(fd,MAX_MANIFEST_BYTES+1)
            if _identity(info)!=_identity(os.fstat(fd)):raise ValueError('File collection manifest changed during read.')
        finally:os.close(fd)
        document=parse_manifest(raw)
        if document.get('selection')!=selection_configuration(config):raise ValueError('Captured file selection differs from the requested configuration; select or capture the source again.')
        if managed and _hash(raw)!=path.parent.name:raise ValueError('File collection manifest directory does not match its SHA-256.')
        yield path,raw,document
    else:
        # Managed authority may never fall back to an original directory/file.
        authority=getattr(check,'__self__',None)
        if authority is not None and hasattr(authority,'request'):raise ValueError('Managed File Input requires an admitted collection manifest.')
        with _portable_capture(path,config,check,preview,sampling) as value:yield value


def _message(error):return ('File reader failed: '+str(error).replace('\0',' ').replace('\n',' '))[:1000]


def source_files(binding,configuration=None,check_cancelled=lambda:None,*,sampling=None,_preview=False):
    config=validate_configuration(configuration or {})
    if __package__:
        from .dataflow import Table,derive,row_store
        from .image_artifacts import artifact_scope,publish_pixels,publish_encoded
        from . import media_artifacts as media
        from .file_input_audio import decode_audio
    else:
        from dataflow import Table,derive,row_store
        from image_artifacts import artifact_scope,publish_pixels,publish_encoded
        import media_artifacts as media
        from file_input_audio import decode_audio
    check=check_cancelled
    with disk_scope(check_cancelled=check),collection(binding,config,check,_preview,sampling) as (manifest,raw,document):
        if document['sourceMode']!=config['sourceMode']:raise ValueError('Captured collection mode differs from File Input configuration.')
        eligible=[]
        for member in document['members']:
            relative=member['relativePath']
            if config['sourceMode']=='directory':
                if config['extensions'] and member['extension'] not in config['extensions']:continue
                if not config['includeHidden'] and any(part.startswith('.') for part in relative.split('/')):continue
                if not config['includeSubfolders'] and '/' in relative:continue
                if len(relative.split('/'))>config['maxDepth']+1:raise ValueError('Captured collection exceeds the configured directory depth.')
                if not _matches(relative,config['includePatterns']) or _matches(relative,config['excludePatterns']):continue
            eligible.append(member)
        if len(eligible)>config['maxFiles']:raise ValueError('Captured collection exceeds the configured file count.')
        selected=eligible[:8] if _preview else _select_files(eligible,sampling,document['complete'])
        image_column=config['readMode']=='content' and not _preview and (any(READERS.get(m['extension'])=='image' for m in selected) or (config['readers']['pdf'].get('renderMode')=='pageImages' and any(m['extension']=='pdf' for m in selected)))
        schema=[{'name':name,'dtype':METADATA_TYPES[name],'nullable':name in ('modified_at','created_at','content_sha256')} for name in config['metadataFields']]
        if config['readMode']=='content':schema.append({'name':'text_content','dtype':'string','nullable':True})
        if config['readMode']=='content' and not _preview and config['readers']['audio']['format']=='pcmWav' and any(m['extension']=='wav' for m in selected):schema.append({'name':'audio_content','dtype':'audio','nullable':True})
        if image_column:schema.append({'name':'image_content','dtype':'image','nullable':True})
        if image_column and config['readers']['pdf'].get('renderMode')=='pageImages':schema += [{'name':'pdf_map_'+str(i),'dtype':'float64','nullable':True} for i in range(9)]
        schema += [{'name':name,'dtype':'int64','nullable':True} for name in ('unit_index','page_number','frame_index')]
        schema += [{'name':'reader_status','dtype':'string','nullable':False},{'name':'reader_message','dtype':'string','nullable':True}]
        store=row_store(tuple(schema),check);count=0;diagnostics=[];parent=_safe_root(manifest.parent,directory=True)
        def append(destination,member,unit,status,message=None):
            identity=member_identity
            values={**{'pdf_map_'+str(i):None for i in range(9)},'file_id':identity,'name':Path(member['relativePath']).name,'relative_path':member['relativePath'],'extension':member['extension'],'detected_type':READERS.get(member['extension'],'unknown'),'byte_size':member['bytes'],'modified_at':member.get('modifiedAt'),'created_at':member.get('createdAt'),'content_sha256':member['sha256'],'text_content':None,'image_content':None,'audio_content':None,'unit_index':0,'page_number':None,'frame_index':None,'reader_status':status,'reader_message':message,**unit}
            suffix='error' if status=='error' else str(values['unit_index'])
            destination.append(tuple(values[column['name']] for column in schema),identity+':'+suffix)
        try:
            for member in selected:
                check();reader=READERS.get(member['extension'])
                if reader=='audio' and config['readers']['audio']['format']=='disabled':reader=None
                member_identity=_hash(_canonical(['kwip.file.v1',member['relativePath'],member['sha256'],member['bytes'],member.get('modifiedAt'),member.get('createdAt')]))
                if reader is None and config['unknownFiles']=='skip':continue
                # Fail-fast readers can stream directly into the operation store.
                # Record/skip readers stage per file so partial units remain atomic.
                stage=store if config['errorPolicy']=='fail' else row_store(tuple(schema),check)
                stage_reserved=0;pixel_files={};context=current_scope();budget=context[1]
                with tempfile.TemporaryDirectory(prefix='.file-pixels-',dir=context[0]) as pixel_directory:
                    def stage_pixel(content,reference):
                        nonlocal stage_reserved
                        check();name=reference['sha256']
                        if name in pixel_files:return
                        budget.resize(stage_reserved,stage_reserved+len(content));stage_reserved+=len(content)
                        target=Path(pixel_directory)/name
                        with target.open('xb') as sink:target.chmod(0o600);sink.write(content)
                        pixel_files[name]=(target,reference)
                    try:
                        failure=None
                        # Capture authority failures are never a recoverable reader error.
                        content=_read_member(parent,member,config['maxFileBytes'],check) if (member['status']=='content' and reader is not None and config['readMode']=='content' and not (_preview and (reader in ('image','audio') or (reader=='pdf' and config['readers']['pdf'].get('renderMode')=='pageImages')))) else None
                        try:
                            if member['status']=='error':raise ValueError(member['error'])
                            if reader is None and config['unknownFiles']=='error':raise ValueError('Unsupported file type; choose metadata or skip for unknown files.')
                            if config['readMode']=='metadata' or reader is None or (_preview and (reader in ('image','audio') or (reader=='pdf' and config['readers']['pdf'].get('renderMode')=='pageImages'))):
                                append(stage,member,{},'metadata')
                            else:
                                if member['status']!='content':raise ValueError('Content is not captured; refresh the source before reading this file.')
                                if reader=='image':
                                    if __package__:from .file_input_images import decode_images
                                    else:from file_input_images import decode_images
                                    options=dict(config['readers'][reader])
                                    if options['frames']!='range':options.pop('frameStart',None);options.pop('frameEnd',None)
                                    with artifact_scope(stage_pixel):
                                        for frame in decode_images(content,options,check):
                                            reference=publish_pixels(frame['pixels'],width=frame['width'],height=frame['height'],channels=frame['channels'],dtype=frame['dtype'],layout=frame['layout'],policy=frame['policy'],check_cancelled=check)
                                            notices='; '.join(item['message'] for item in frame.get('notices',[]))[:1000] or None
                                            append(stage,member,{'image_content':reference,'unit_index':frame['frame_index'],'frame_index':frame['frame_index']},'content',notices)
                                elif reader=='pdf' and config['readers']['pdf'].get('renderMode')=='pageImages':
                                    if __package__:from .ocr import render_pdf
                                    else:from ocr import render_pdf
                                    options=config['readers']['pdf']
                                    with artifact_scope(stage_pixel):
                                        for page in render_pdf(content,pages=options['pages'],dpi=options['renderDPI'],maximum_pages=options['maximumRenderPages'],check=check):
                                            pixels=page['pixels'];h,w,_=pixels.shape
                                            reference=publish_pixels(pixels.tobytes(),width=w,height=h,channels=3,dtype='uint8',layout='HWC',policy=dict(orientation='stored',colorMode='RGB',palette='expandRGBA',actualMode='RGB'),check_cancelled=check)
                                            append(stage,member,{'image_content':reference,'unit_index':page['page']-1,'page_number':page['page'],'frame_index':page['page']-1,**{'pdf_map_'+str(i):value for i,value in enumerate(page['pixelToPage'])}},'content')
                                elif reader=='audio':
                                    decoded=decode_audio(content,config['readers']['audio'],check)
                                    samples=decoded.pop('samples')
                                    encoded,reference=media.encode_asset(samples,kind='audio',**decoded)
                                    stage_pixel(encoded,reference)
                                    append(stage,member,{'audio_content':reference},'content')
                                else:
                                    if __package__:from .file_input_documents import extract_document
                                    else:from file_input_documents import extract_document
                                    for unit in extract_document(content,reader,config['readers'][reader],check):
                                        if _preview and store.count+(0 if stage is store else stage.count)>=8:break
                                        append(stage,member,unit,'content')
                        except Exception as error:
                            cancellation_classes=tuple(cls for name in ('__main__','runner.main','main') if isinstance(cls:=getattr(sys.modules.get(name),'Cancelled',None),type))
                            if isinstance(error,(InterruptedError,*cancellation_classes)):raise
                            check()
                            if config['errorPolicy']=='fail':raise
                            failure=_message(error)
                            diagnostics.append({'kind':'fileReadError','message':failure})
                        if failure is not None:
                            if config['errorPolicy']=='record':append(store,member,{'unit_index':None},'error',failure);count+=1
                        else:
                            # Publication/storage failures are not parsing errors:
                            # they must fail the operation under every error policy.
                            for target,reference in pixel_files.values():
                                check()
                                publisher=media.publish_encoded if reference.get('kind')=='audioSamplesRef' else publish_encoded
                                publisher(target.read_bytes(),reference,check_cancelled=check)
                            if stage is store:count=store.count
                            else:
                                rows,ids=stage.finish()
                                for row,identity in zip(rows,ids):
                                    check();store.append(row,identity);count+=1
                    finally:
                        if stage is not store:stage.close()
                        if stage_reserved:budget.resize(stage_reserved,0)
                if _preview and count>=8:break
            rows,ids=store.finish()
            return derive(Table(tuple(schema),rows,ids,_hash(raw),diagnostics=tuple(diagnostics[-16:])))
        except BaseException:store.close();raise
        finally:os.close(parent)


def preview_files(binding,configuration=None,check_cancelled=lambda:None):
    config=validate_configuration(configuration or {})
    with disk_scope(check_cancelled=check_cancelled),collection(binding,config,check_cancelled,True) as (manifest,raw,document):
        table=source_files({'manifest':manifest},config,check_cancelled,_preview=True)
        try:
            rows=[list(row) for row in table.rows[:8]]
            def display(value):return '' if value is None else str(value).lower() if type(value) is bool else str(value)[:2000]
            return {'event':'dataPreview','columns':[c['name'] for c in table.columns],'columnSchema':list(table.columns),'rows':[[display(value) for value in row] for row in rows],'rowCount':len(rows),'nullCells':[[i,j] for i,row in enumerate(rows) for j,value in enumerate(row) if value is None],'sourceSHA256':_hash(raw),'parsing':{'options':config,'validatedRows':len(rows),'previewRows':len(rows),'sampleOnly':True},'collection':{'complete':document['complete'],'selectedFiles':min(len(document['members']),8),'observedFiles':len(document['members']),'sourceMode':document['sourceMode'],'readMode':document['readMode'],'imagePixelsIncluded':False}}
        finally:table.rows.store.close()
