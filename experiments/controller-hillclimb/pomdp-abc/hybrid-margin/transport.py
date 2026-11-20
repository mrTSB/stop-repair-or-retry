"""Retry only infrastructure failures; never retry a completed poor model answer."""

import hashlib
import json
import threading
import time


def resilient(worker,journal,attempts=4):
    lock=threading.Lock()
    def request(path,payload=None):
        for attempt in range(attempts):
            try:
                return worker.request.remote(path,payload)
            except Exception as error:
                name=type(error).__name__;message=str(error).lower()
                transient=(name in ('ConnectionError','TimeoutError') or
                           (name in ('RemoteError','FunctionTimeoutError') and any(s in message for s in
                           ('deadline','timeout','timed out','cancelled by user or a failure','remote end closed','connection reset'))))
                if not transient:raise
                receipt={'time_unix':time.time(),'attempt':attempt+1,'error_type':name,
                         'path':path,'payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
                         'model':(payload or {}).get('model'),'reason':'transport failure; no usable response receipt'}
                with lock:
                    with journal.open('a') as stream:stream.write(json.dumps(receipt)+'\n')
                if attempt+1==attempts:raise
                print(f'Transient {name}; retrying request {attempt+2}/{attempts}',flush=True)
                time.sleep(min(5*2**attempt,30))
    return request
