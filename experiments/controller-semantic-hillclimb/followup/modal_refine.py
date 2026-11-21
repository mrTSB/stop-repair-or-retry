"""Bound one training GPU for the prespecified correctness-focused refinement."""

from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App('srr-large-root-refinement-20261120')
volume = modal.Volume.from_name('srr-semantic-training-20261120')
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_pip_install('uv', 'torch==2.9.0', 'transformers==4.57.1', 'safetensors==0.6.2',
                         'numpy==2.5.4', 'sentencepiece==0.2.1', 'protobuf==5.29.5')
         .add_local_file(HERE / 'network_large.py', '/root/semantic/network_large.py')
         .add_local_file(HERE / 'refine_large.py', '/root/semantic/refine_large.py'))


@app.function(image=image, gpu='L40S', cpu=8, memory=32768, timeout=14400,
              max_containers=1, volumes={'/study': volume})
def train():
    import json
    import subprocess
    import sys
    import time
    process = subprocess.Popen(['uv', 'run', '--no-project', '--no-managed-python', '--python', sys.executable,
                                '/root/semantic/refine_large.py', '--input', '/study/grounded-train.jsonl',
                                '--initial', '/study/verifier-large-v1', '--output', '/study/verifier-large-root-v2'])
    saved = set()
    while process.poll() is None:
        checkpoints = {str(p) for p in Path('/study/verifier-large-root-v2').glob('*/recipe.json')}
        if checkpoints - saved:
            volume.commit()
            saved = checkpoints
        time.sleep(20)
    volume.commit()
    if process.returncode:
        raise RuntimeError(f'Refinement failed: {process.returncode}')
    return json.loads(Path('/study/verifier-large-root-v2/training-summary.json').read_bytes())


@app.local_entrypoint()
def main():
    print(train.remote(), flush=True)
