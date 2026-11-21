"""One bounded GPU for supervised POMDP observation/transition training."""

from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App('srr-grounded-training-20261120')
volume = modal.Volume.from_name('srr-semantic-training-20261120', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('uv', 'torch==2.9.0', 'transformers==4.57.1', 'safetensors==0.6.2',
                      'numpy==2.5.4', 'sentencepiece==0.2.1', 'protobuf==5.29.5')
         .add_local_file(HERE / 'network_grounded.py', '/root/semantic/network_grounded.py')
         .add_local_file(HERE / 'train_grounded.py', '/root/semantic/train_grounded.py'))


@app.function(image=image, gpu='L40S', cpu=8, memory=32768, timeout=14400,
              max_containers=1, volumes={'/study':volume},
              env={'HF_HUB_DISABLE_TELEMETRY':'1', 'HF_HUB_DISABLE_XET':'1'})
def train():
    import json
    import subprocess
    import sys
    import time
    # Use uv for Python execution inside the task, as in the local workspace.
    process = subprocess.Popen(['uv', 'run', '--no-project', '--no-managed-python', '--python', sys.executable,
                    '/root/semantic/train_grounded.py', '--input', '/study/grounded-train.jsonl',
                    '--output', '/study/verifier-grounded-v1', '--cache', '/study/cache'])
    saved = set()
    while process.poll() is None:
        checkpoints = {str(p) for p in Path('/study/verifier-grounded-v1').glob('*/recipe.json')}
        if checkpoints - saved:
            volume.commit()
            saved = checkpoints
        time.sleep(20)
    if process.returncode:
        volume.commit()
        raise RuntimeError(f'Training process exited with code {process.returncode}')
    volume.commit()
    return json.loads(Path('/study/verifier-grounded-v1/training-summary.json').read_bytes())


@app.local_entrypoint()
def main():
    print(train.remote(), flush=True)
