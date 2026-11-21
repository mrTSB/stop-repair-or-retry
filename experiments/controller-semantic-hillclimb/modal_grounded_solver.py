"""Private, bounded GPU worker serving the exact archived Ollama model digests."""

import modal

APP_NAME = 'srr-grounded-solver-20261120'
MODELS = {
    'qwen3.5:4b': '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd',
    'qwen3.5:35b': '3460ffeede5453ead027dbd2f821b12ad0aa3de54630971993babdb2165221f7',
}
MANIFESTS_B64 = {'qwen3.5:4b': 'eyJzY2hlbWFWZXJzaW9uIjoyLCJtZWRpYVR5cGUiOiJhcHBsaWNhdGlvbi92bmQuZG9ja2VyLmRpc3RyaWJ1dGlvbi5tYW5pZmVzdC52Mitqc29uIiwiY29uZmlnIjp7Im1lZGlhVHlwZSI6ImFwcGxpY2F0aW9uL3ZuZC5kb2NrZXIuY29udGFpbmVyLmltYWdlLnYxK2pzb24iLCJkaWdlc3QiOiJzaGEyNTY6ZGU5ZmVkMjI1MWIzNzI5NWI3NjM3MjdhNTljYTM1Y2Y1Y2ZlNWM3Mzc5YmMzZTIxMDRiMmNlM2MxNDVhYTg4NyIsInNpemUiOjQ3NX0sImxheWVycyI6W3sibWVkaWFUeXBlIjoiYXBwbGljYXRpb24vdm5kLm9sbGFtYS5pbWFnZS5tb2RlbCIsImRpZ2VzdCI6InNoYTI1Njo4MWZiNjBjN2RhYTgwZmMxMTIzMzgwYjk4OTcwYjMyMGFlMjMzNDA5ZjBmNzFhNzJlZDdiOWIwZDYyZjQwNDkwIiwic2l6ZSI6MzM4OTk3MTg0MH0seyJtZWRpYVR5cGUiOiJhcHBsaWNhdGlvbi92bmQub2xsYW1hLmltYWdlLmxpY2Vuc2UiLCJkaWdlc3QiOiJzaGEyNTY6NzMzOWZhNDE4YzlhZDNlOGUxMmU3NGFkMGZkMjZhOWNjNGJlODcwM2Y5YzExMDcyOGE5OTJiMTkzYmU4NWNiMiIsInNpemUiOjExMzU1fSx7Im1lZGlhVHlwZSI6ImFwcGxpY2F0aW9uL3ZuZC5vbGxhbWEuaW1hZ2UucGFyYW1zIiwiZGlnZXN0Ijoic2hhMjU2OjkzNzEzNjRiMjdhNTJhY2FjOWQ4N2Y4OGJkOTNjOWRiMTE3NGQ4ZDZlYzU3ZjY4ODg5MjVjZGMxNzg4ODcxZmYiLCJzaXplIjo2NX1dfQ==', 'qwen3.5:35b': 'eyJzY2hlbWFWZXJzaW9uIjoyLCJtZWRpYVR5cGUiOiJhcHBsaWNhdGlvbi92bmQuZG9ja2VyLmRpc3RyaWJ1dGlvbi5tYW5pZmVzdC52Mitqc29uIiwiY29uZmlnIjp7Im1lZGlhVHlwZSI6ImFwcGxpY2F0aW9uL3ZuZC5kb2NrZXIuY29udGFpbmVyLmltYWdlLnYxK2pzb24iLCJkaWdlc3QiOiJzaGEyNTY6NjA2YWQ5ZjFlY2JjZDBkNDAwYTg5NTcyZjQ5ZDlhOGU4MGIxMWEzNDZiZTc0ODgzMjQ1ZTI0ZWNiYmMzYWM4MSIsInNpemUiOjQ4Mn0sImxheWVycyI6W3sibWVkaWFUeXBlIjoiYXBwbGljYXRpb24vdm5kLm9sbGFtYS5pbWFnZS5tb2RlbCIsImRpZ2VzdCI6InNoYTI1Njo5MDBkZGU2MmZiN2ViZThhNWEyNWUzNWQ1Yjc2MzNmNDAzZjIyNmEzMTA5NjVmZWQ1MWQ1MGY1MjM4YmExNDVhIiwic2l6ZSI6MjM4NjkxNzk4NDB9LHsibWVkaWFUeXBlIjoiYXBwbGljYXRpb24vdm5kLm9sbGFtYS5pbWFnZS5saWNlbnNlIiwiZGlnZXN0Ijoic2hhMjU2OjczMzlmYTQxOGM5YWQzZThlMTJlNzRhZDBmZDI2YTljYzRiZTg3MDNmOWMxMTA3MjhhOTkyYjE5M2JlODVjYjIiLCJzaXplIjoxMTM1NX0seyJtZWRpYVR5cGUiOiJhcHBsaWNhdGlvbi92bmQub2xsYW1hLmltYWdlLnBhcmFtcyIsImRpZ2VzdCI6InNoYTI1Njo5MzcxMzY0YjI3YTUyYWNhYzlkODdmODhiZDkzYzlkYjExNzRkOGQ2ZWM1N2Y2ODg4OTI1Y2RjMTc4ODg3MWZmIiwic2l6ZSI6NjV9XX0='}
image = modal.Image.from_registry('ollama/ollama:0.40.16', add_python='3.12').entrypoint([])
cache = modal.Volume.from_name('srr-pomdp-abc-ollama', create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(image=image, cpu=2, memory=4096, volumes={'/root/.ollama': cache},
              timeout=900, max_containers=1)
def prepare_weights():
    import base64
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import json
    from pathlib import Path
    from urllib.request import urlopen

    blobs = Path('/root/.ollama/models/blobs')
    manifests = Path('/root/.ollama/models/manifests/registry.ollama.ai/library/qwen3.5')
    blobs.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    wanted = {}
    raw_manifests = {}
    for model, encoded in MANIFESTS_B64.items():
        raw = base64.b64decode(encoded)
        if hashlib.sha256(raw).hexdigest() != MODELS[model]:
            raise ValueError('Embedded manifest digest changed')
        obj = json.loads(raw)
        raw_manifests[model] = raw
        for layer in [obj['config'], *obj['layers']]:
            wanted[layer['digest']] = layer['size']

    def fetch(item):
        fingerprint, size = item
        target = blobs / fingerprint.replace(':', '-')
        if target.exists() and target.stat().st_size == size:
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() == fingerprint.split(':')[1]:
                    return
        temporary = target.with_suffix('.partial')
        digest = hashlib.sha256()
        with urlopen('https://registry.ollama.ai/v2/library/qwen3.5/blobs/'+fingerprint, timeout=500) as response, temporary.open('wb') as stream:
            while chunk := response.read(8*1024*1024):
                stream.write(chunk)
                digest.update(chunk)
        if temporary.stat().st_size != size or digest.hexdigest() != fingerprint.split(':')[1]:
            raise ValueError('Downloaded model blob failed cryptographic verification')
        temporary.replace(target)
        print(f'Verified model blob: {size} bytes', flush=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, wanted.items()))
    for model, raw in raw_manifests.items():
        (manifests / model.rsplit(':', 1)[1]).write_bytes(raw)
    cache.commit()
    return MODELS


@app.cls(image=image, gpu='L40S', cpu=8, memory=49152,
         volumes={'/root/.ollama': cache}, timeout=600, startup_timeout=900,
         scaledown_window=300, max_containers=2,
         env={'OLLAMA_HOST': '127.0.0.1:11434', 'OLLAMA_NUM_PARALLEL': '2',
              'OLLAMA_MAX_LOADED_MODELS': '2', 'OLLAMA_CONTEXT_LENGTH': '4096',
              'OLLAMA_FLASH_ATTENTION': '1'})
@modal.concurrent(max_inputs=2, target_inputs=2)
class Ollama:
    @modal.enter()
    def start(self):
        import subprocess
        import time
        import urllib.error

        self.log = open('/tmp/ollama.log', 'a')
        self.process = subprocess.Popen(['ollama', 'serve'], stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+60
        while True:
            try:
                self._request('/api/version')
                break
            except (ConnectionError, urllib.error.URLError):
                if time.monotonic() > deadline or self.process.poll() is not None:
                    raise RuntimeError('Inference server did not start')
                time.sleep(.5)
        actual = {r['name']: r['digest'] for r in self._request('/api/tags')['models']}
        if any(actual.get(model) != fingerprint for model, fingerprint in MODELS.items()):
            raise ValueError('Downloaded model digests differ from the frozen compact solver/reviewer')
        self.version = self._request('/api/version')['version']
        print(f'Fixed weights ready; Ollama {self.version}; L40S GPU worker.', flush=True)

    def _request(self, path, payload=None):
        import json
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        body = json.dumps(payload).encode() if payload is not None else None
        try:
            with urlopen(Request('http://127.0.0.1:11434'+path, data=body,
                                 headers={'Content-Type': 'application/json'}), timeout=500) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(2000).decode(errors='replace')
            raise RuntimeError(f'Inference server HTTP {error.code}: {detail}') from None

    @modal.method()
    def request(self, path, payload=None):
        if path not in ('/api/generate', '/api/tags', '/api/version', '/api/ps', '/api/show'):
            raise ValueError('Unsupported inference request')
        if payload is not None and payload.get('model') not in MODELS:
            raise ValueError('Unapproved model')
        result = self._request(path, payload)
        result['srr_backend'] = {'provider': 'Modal', 'gpu': 'L40S', 'ollama_version': self.version,
                                 'app': APP_NAME, 'max_containers': 2, 'parallel_inputs': 2, 'target_inputs': 2}
        return result

    @modal.method()
    def benchmark(self):
        import time
        from pathlib import Path
        import subprocess
        prompt = '<|im_start|>user\nWhat is 7 times 9? Reply only with the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        results = []
        for i in range(3):
            started = time.monotonic()
            result = self._request('/api/generate', {'model':'qwen3.5:35b', 'prompt':prompt,
                'raw':True, 'stream':False, 'keep_alive':'30m',
                'options':{'temperature':0, 'num_predict':20, 'num_ctx':4096}})
            results.append({'iteration':i, 'wall_seconds':time.monotonic()-started,
                            **{k:result.get(k) for k in ('response','done','load_duration',
                            'prompt_eval_count','prompt_eval_duration','eval_count','eval_duration')}})
            print(results[-1], flush=True)
        gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
        log = Path('/tmp/ollama.log').read_text(errors='replace')
        lines = [line for line in log.splitlines() if any(word in line.lower() for word in
                 ('offload','cuda','memory','tokens per second'))]
        return {'results':results, 'gpu':gpu, 'log': '\n'.join(lines)[-9000:]}

    @modal.method()
    def diagnostics(self):
        from pathlib import Path
        import subprocess
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                              '--format=csv'], capture_output=True, text=True)
        log = Path('/tmp/ollama.log').read_text(errors='replace')[-14000:]
        return {'gpu': gpu.stdout, 'log': log, 'server_pid': self.process.pid,
                'server_exit': self.process.poll()}

    @modal.exit()
    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()


@app.local_entrypoint()
def probe():
    result = Ollama().request.remote('/api/tags')
    print({r['name']: r['digest'] for r in result['models']})
