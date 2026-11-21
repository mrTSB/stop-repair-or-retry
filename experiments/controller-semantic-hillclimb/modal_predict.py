"""Batch semantic observations on one GPU; inputs contain no reference answers."""

import modal
from pathlib import Path

HERE=Path(__file__).resolve().parent
volume=modal.Volume.from_name('srr-semantic-training-20261120')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('uv','torch==2.9.0','transformers==4.57.1','safetensors==0.6.2',
                    'numpy==2.5.4','sentencepiece==0.2.1','protobuf==5.29.5')
       .add_local_file(HERE/'network.py','/root/semantic/network.py'))

app=modal.App('srr-semantic-observations-20261120')


@app.function(image=image,gpu='L40S',cpu=8,memory=32768,timeout=1800,max_containers=1,
              volumes={'/study':volume})
def observe(name):
    import json,sys
    from pathlib import Path
    import numpy as np
    import torch
    sys.path.insert(0,'/root/semantic')
    from network import encode,load,predict
    from transformers.utils import logging
    logging.set_verbosity_error();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[json.loads(s) for s in Path('/study/'+name+'.jsonl').read_text().splitlines()]
    assert all(set(r)<= {'id','kind','path','left','right','numeric','input_sha256'} for r in rows)
    model,tokenizer,recipe=load('/study/verifier-v2/full')
    model=model.to('cuda')
    tensors=encode(tokenizer,rows,recipe['max_length'])
    state,plain,guided=predict(model,tensors,device='cuda',batch=16)
    np.savez_compressed('/study/'+name+'.npz',state=state,plain=plain,guided=guided,
                        tokens=tensors['attention_mask'].sum(1).numpy())
    volume.commit()
    return {'nodes':len(rows),'weights':recipe['weights'],'reference_answers_present':False}


@app.local_entrypoint()
def main(name:str='development-inputs'):
    print(observe.remote(name),flush=True)
