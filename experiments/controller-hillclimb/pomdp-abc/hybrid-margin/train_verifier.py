"""Question-grouped cross-fitting of a small neural POMDP observation model."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup

from margin_data import read,wait_complete
from extended_acquire import feedback
from difficulty_pomdp import FIXED,visible
from scoring import correct
from neural_verifier import BASE_MODEL,REVISION,CACHE,Verifier,text_pair,inputs,predict,save_network,load_network

SEED=2026112013


def examples(rows,trees,judges,folds):
    result=[]
    for row in rows:
        tree=trees[row['id']]
        hard=int(np.mean([correct(tree[p],row['answer']) for p in FIXED])<.6)
        fold=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)%folds
        for kind,paths in (('solver',tuple(tree)),('judge',('','repair','retry','guided'))):
            for path in paths:
                node=tree[path] if kind=='solver' else judges[row['id'],path]
                current=tree
                if kind=='judge':
                    node={**node,'text':str(node['verdict'].get('solution',''))+'\nFinal: '+node['answer'],'confidence':None}
                    current={**tree,path:node}
                left,right=text_pair(row,node)
                result.append({'id':row['id'],'kind':kind,'path':path,'left':left,'right':right,
                               'numeric':visible(row,path,current),'label':2*hard+int(correct(node,row['answer'])),
                               'fold':fold,'valid':bool(node['answer'])})
    return result


def train(args):
    torch.set_num_threads(4)
    rows,trees,judges=wait_complete(args.data,'train',args.wait)
    samples=examples(rows,trees,judges,args.folds)
    args.output.mkdir(parents=True,exist_ok=True)
    text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in samples)
    contract={'base_model':BASE_MODEL,'revision':REVISION,'examples_sha256':hashlib.sha256(text.encode()).hexdigest(),
              'epochs':args.epochs,'folds':args.folds,'max_length':args.max_length,'batch':args.batch,
              'encoder_lr':2e-5,'head_lr':3e-4,'seed':SEED,'train_questions':len(rows),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'network_source_sha256':hashlib.sha256(Path(__file__).with_name('neural_verifier.py').read_bytes()).hexdigest()}
    target=args.output/'contract.json'
    if target.exists() and json.loads(target.read_bytes())!=contract:raise ValueError('Changed verifier experiment; use a new output directory')
    target.write_text(json.dumps(contract,indent=2)+'\n');(args.output/'examples.jsonl').write_text(text)
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,revision=REVISION,cache_dir=CACHE,trust_remote_code=False)
    data=inputs(tokenizer,[r['left'] for r in samples],[r['right'] for r in samples],
                [r['numeric'] for r in samples],args.max_length)
    labels=torch.tensor([r['label'] for r in samples],dtype=torch.long)
    folds=np.array([r['fold'] for r in samples]);oof=np.zeros((len(samples),4),dtype=np.float32)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Neural verifier: {len(samples)} training examples / {len(rows)} questions; device={device}; fresh data unused',flush=True)
    for fold in (*range(args.folds),'full'):
        directory=args.output/(f'fold-{fold}' if fold!='full' else 'full')
        train_indices=np.arange(len(samples)) if fold=='full' else np.flatnonzero(folds!=fold)
        if not (directory/'recipe.json').exists():
            seed=SEED+(args.folds if fold=='full' else fold)
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            encoder=AutoModel.from_pretrained(BASE_MODEL,revision=REVISION,cache_dir=CACHE,
                                               trust_remote_code=False,attn_implementation='eager')
            numeric=data['numeric'][train_indices].numpy()
            model=Verifier(encoder,numeric.shape[1],numeric.mean(axis=0),numeric.std(axis=0).clip(min=.05)).to(device)
            optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':2e-5},
                 {'params':list(model.text_norm.parameters())+list(model.head.parameters()),'lr':3e-4}],weight_decay=.01,foreach=False)
            subset=TensorDataset(*(data[k][train_indices] for k in ('input_ids','attention_mask','token_type_ids','numeric')),
                                 labels[train_indices])
            loader=DataLoader(subset,batch_size=args.batch,shuffle=True,num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
            total=len(loader)*args.epochs
            scheduler=get_linear_schedule_with_warmup(optimizer,max(1,total//10),total)
            for epoch in range(args.epochs):
                model.train();loss_sum=0.;seen=0
                for batch in loader:
                    input_ids,mask,types,numbers,y=[v.to(device) for v in batch]
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(input_ids,mask,types,numbers)
                    loss=torch.nn.functional.cross_entropy(logits,y);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step();scheduler.step()
                    loss_sum+=float(loss.detach().cpu())*len(y);seen+=len(y)
                print(f'Verifier fold {fold} epoch {epoch+1}/{args.epochs}: train loss={loss_sum/seen:.4f}',flush=True)
            save_network(model,tokenizer,directory,{**contract,'fold':fold})
            del optimizer,model,encoder
            if device=='mps':torch.mps.empty_cache()
        model,_,recipe=load_network(directory)
        if any(recipe.get(k)!=v for k,v in contract.items()):raise ValueError('Stale verifier fold checkpoint')
        if fold!='full':
            indices=np.flatnonzero(folds==fold)
            oof[indices]=predict(model,{k:v[indices] for k,v in data.items()})
            print(f'Verifier fold {fold}: generated {len(indices)} out-of-question predictions on CPU',flush=True)
        del model
    np.savez_compressed(args.output/'oof-probabilities.npz',probabilities=oof)
    valid=np.array([r['valid'] for r in samples]);truth=labels.numpy()%2;score=oof[:,1]+oof[:,3]
    report={'examples':len(samples),'questions':len(rows),'question_grouped_folds':args.folds,
            'oof_binary_accuracy_valid':float(np.mean((score[valid]>=.5)==truth[valid])),
            'oof_brier_valid':float(np.mean((score[valid]-truth[valid])**2)),
            'parameters':json.loads((args.output/'full/recipe.json').read_bytes())['parameters'],
            'fresh_holdout_used':False}
    (args.output/'training-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--max-length',type=int,default=384)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--wait',type=int,default=43200)
    train(parser.parse_args())
