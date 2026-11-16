"""Allowlisted optional backend dispatch; never imports arbitrary plugin code."""
import hashlib
import json
if __package__:
    from .prepared_data import bounded_read
else:
    from prepared_data import bounded_read


def execute_request(path,pin,output,emit,*,check_cancelled=lambda:None):
    check_cancelled();raw=bounded_read(path,100_000)
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('Backend request differs from its SHA256 pin.')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate backend request field.')
            result[key]=value
        return result
    request=json.loads(raw,object_pairs_hook=unique)
    if not isinstance(request,dict):raise ValueError('Backend request must be an object.')
    if request.get('backendID')=='sklearn':
        if __package__:from .sklearn_workflow import run_prepared
        else:from sklearn_workflow import run_prepared
    elif request.get('backendID')=='tensorflow':
        if __package__:from .tensorflow_workflow import run_prepared
        else:from tensorflow_workflow import run_prepared
    else:raise ValueError('Unsupported fitting backend.')
    return run_prepared(request,output,emit,check_cancelled=check_cancelled)
