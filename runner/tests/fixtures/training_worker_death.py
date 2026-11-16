"""Test-only supervisor injection: kill the retained real worker after epoch one."""
import json
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import threading


def main():
    arguments = json.loads(sys.argv[1])
    proof = Path(sys.argv[2])
    index = arguments.index('--kwip-supervise')
    entry = Path(arguments[index - 1])
    group, nonce, intent_pin, ownership_pin = arguments[index + 1:]
    sys.path.insert(0, str(entry.parent))
    import worker_supervisor
    # Bootstrap itself performs the real claim. This pre-read only chooses the
    # expected test target; it grants no authority and never edits launch files.
    intent = json.loads((Path(group) / ('launch-' + nonce + '.json')).read_bytes())
    if intent['ordinal'] != 2 or intent['phase'] != 'training':
        raise RuntimeError('Worker-death fixture received the wrong pinned phase')
    original = subprocess.Popen
    children = []
    readers = []
    observation = {}

    def launch(*args, **kwargs):
        if children:
            raise RuntimeError('Expected exactly one supervisor-owned worker')
        kwargs['stdout'] = subprocess.PIPE
        kwargs['bufsize'] = 0
        child = original(*args, **kwargs)
        children.append(child)
        def relay():
            for raw in iter(child.stdout.readline, b''):
                event = json.loads(raw)
                if event.get('event') == 'epoch' and not observation:
                    observation.update(epoch=event['epoch'])
                    # Popen retains/reaps this direct child; never signal a
                    # parsed event PID or a previously reaped orphan.
                    child.kill()
                sys.stdout.buffer.write(raw)
                sys.stdout.buffer.flush()
        reader = threading.Thread(target=relay)
        readers.append(reader);reader.start()
        return child

    worker_supervisor.subprocess.Popen = launch
    sys.argv = [str(entry)] + arguments[index:]
    code = 0
    try:
        runpy.run_path(str(entry), run_name='__main__')
    except SystemExit as error:
        code = error.code
    finally:
        worker_supervisor.subprocess.Popen = original
        for reader in readers:
            reader.join(timeout=5)
            if reader.is_alive():
                raise RuntimeError('Worker output relay failed to finish')
    if len(children) != 1 or children[0].returncode != -signal.SIGKILL or observation.get('epoch') != 1:
        raise RuntimeError('Did not kill the actual training child after its first epoch')
    proof.write_text(json.dumps(dict(nonce=nonce, ordinal=intent['ordinal'], phase=intent['phase'],
                                    epoch=observation['epoch'], returncode=children[0].returncode)))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
