"""Small discriminative observation model; never generates an answer or selects an action."""

from dataclasses import dataclass,field
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from transformers import AutoConfig,AutoModel,AutoTokenizer

from margin_data import ROOT
from difficulty_pomdp import visible,observation
from extended_acquire import SCIENCE_SUFFIX

BASE_MODEL='sentence-transformers/all-MiniLM-L6-v2'
REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
CACHE=ROOT/'.cache/neural-observer'


def text_pair(row,node):
    task=row['question'].removesuffix(SCIENCE_SUFFIX)
    question,separator,options=task.partition('\n\nOptions:\n')
    candidate=str(node['answer'])
    if separator and candidate in 'ABCDEFGHIJ' and len(candidate)==1:
        match=re.search(r'^'+candidate+r'\.\s*(.*?)(?=^[A-J]\.\s|\Z)',options,re.M|re.S)
        if match:candidate+=': '+match[1].strip()
    return question,'Proposed answer: '+candidate+'\nSolution:\n'+node['text']


class Verifier(nn.Module):
    def __init__(self,encoder,width,mean=None,std=None):
        super().__init__();self.encoder=encoder
        self.register_buffer('numeric_mean',torch.zeros(width) if mean is None else torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('numeric_std',torch.ones(width) if std is None else torch.tensor(std,dtype=torch.float32))
        hidden=encoder.config.hidden_size
        self.text_norm=nn.LayerNorm(hidden)
        self.head=nn.Sequential(nn.Linear(hidden+width,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,4))

    def forward(self,input_ids,attention_mask,token_type_ids,numeric):
        states=self.encoder(input_ids=input_ids,attention_mask=attention_mask,
                            token_type_ids=token_type_ids).last_hidden_state
        mask=attention_mask.unsqueeze(-1).to(states.dtype)
        pooled=(states*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
        numeric=((numeric-self.numeric_mean)/self.numeric_std).clamp(-6,6)
        return self.head(torch.cat((self.text_norm(pooled),numeric),dim=1))


def save_network(model,tokenizer,directory,recipe):
    directory.mkdir(parents=True,exist_ok=True)
    model=model.cpu()
    model.encoder.config.save_pretrained(directory);tokenizer.save_pretrained(directory)
    # Evaluate the exact half-rounded weights that will be archived, in float32.
    tensors={k:(v.detach().cpu().contiguous() if k in ('numeric_mean','numeric_std')
                else v.detach().cpu().half().contiguous()) for k,v in model.state_dict().items()}
    save_file(tensors,str(directory/'weights.safetensors'))
    recipe={**recipe,'numeric_width':len(model.numeric_mean),
            'parameters':sum(p.numel() for p in model.parameters()),
            'weights_sha256':hashlib.sha256((directory/'weights.safetensors').read_bytes()).hexdigest()}
    (directory/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    return recipe


def load_network(directory):
    directory=Path(directory);recipe=json.loads((directory/'recipe.json').read_bytes())
    if hashlib.sha256((directory/'weights.safetensors').read_bytes()).hexdigest()!=recipe['weights_sha256']:
        raise ValueError('Trained observation weights changed')
    config=AutoConfig.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    encoder=AutoModel.from_config(config,attn_implementation='eager',trust_remote_code=False)
    model=Verifier(encoder,recipe['numeric_width'])
    model.load_state_dict(load_file(str(directory/'weights.safetensors')))
    model.eval()
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    return model,tokenizer,recipe


def inputs(tokenizer,left,right,numeric,length):
    data=tokenizer(left,right,padding='max_length',truncation='longest_first',max_length=length,return_tensors='pt')
    return {**data,'numeric':torch.tensor(np.asarray(numeric),dtype=torch.float32)}


def predict(model,data,batch=64):
    results=[]
    with torch.inference_mode():
        for start in range(0,len(data['input_ids']),batch):
            logits=model(**{k:v[start:start+batch] for k,v in data.items()})
            results.append(logits.softmax(dim=-1).cpu().numpy())
    return np.concatenate(results) if results else np.empty((0,4))


@dataclass
class Observer:
    directory: str
    _network: object = field(default=None,repr=False)
    _tokenizer: object = field(default=None,repr=False)
    _recipe: dict = field(default_factory=dict,repr=False)
    _probability_cache: dict = field(default_factory=dict,repr=False)
    classes_: object = field(default_factory=lambda:np.arange(4))

    def __getstate__(self):
        return {**vars(self),'_network':None,'_tokenizer':None,'_recipe':{},'_probability_cache':{}}

    def set_params(self,**kwargs):
        if set(kwargs)-{'n_jobs'}:raise ValueError('Unknown observer runtime setting')
        return self

    def load(self):
        if self._network is None:
            torch.set_num_threads(4)
            self._network,self._tokenizer,self._recipe=load_network(self.directory)

    def row_probabilities(self,row,path,tree):
        node=tree[path]
        if not node['answer']:return np.array([.5,0.,.5,0.])
        key=row['id'],path,node['prompt_sha256'],node['text']
        if key not in self._probability_cache:
            self.load();left,right=text_pair(row,node)
            encoded=inputs(self._tokenizer,[left],[right],[visible(row,path,tree)],self._recipe['max_length'])
            self._probability_cache[key]=predict(self._network,encoded,1)[0]
        return self._probability_cache[key]

    def warm(self,model,rows,trees):
        self.load();examples=[(r,p) for r in rows for p in trees[r['id']]]
        pairs=[text_pair(r,trees[r['id']][p]) for r,p in examples]
        encoded=inputs(self._tokenizer,[a for a,b in pairs],[b for a,b in pairs],
                       [visible(r,p,trees[r['id']]) for r,p in examples],self._recipe['max_length'])
        probabilities=predict(self._network,encoded)
        for (r,p),prob in zip(examples,probabilities):
            node=trees[r['id']][p];key=r['id'],p,node['prompt_sha256'],node['text']
            self._probability_cache[key]=prob
            model._observed[key]=observation(prob,bool(node['answer']))

    def charges(self,row,tree,path):
        self.load();parts=path.split('/') if path else []
        visited=['']+['/'.join(parts[:i]) for i in range(1,len(parts)+1)]
        tokens=0;calls=0
        for p in visited:
            node=tree[p]
            if not node['answer']:continue
            left,right=text_pair(row,node)
            encoded=self._tokenizer(left,right,truncation='longest_first',max_length=self._recipe['max_length'])
            tokens+=len(encoded['input_ids']);calls+=1
        return tokens,calls,self._recipe['parameters']/1e9
