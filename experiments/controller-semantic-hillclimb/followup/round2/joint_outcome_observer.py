"""Locally adapted discriminative16-state observer; never generates answers."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import network_large as initial

ARCHITECTURE='DeBERTa-large-paid-prefix-joint16-v1'
PRIOR_WEIGHT=.05
BITS=torch.tensor([[(i>>j)&1 for j in (3,2,1,0)] for i in range(16)],dtype=torch.float32)


class Network(nn.Module):
    def __init__(self,encoder,width,mean=None,std=None):
        super().__init__();self.encoder=encoder
        self.register_buffer('numeric_mean',torch.zeros(width) if mean is None else torch.as_tensor(mean,dtype=torch.float32).clone())
        self.register_buffer('numeric_std',torch.ones(width) if std is None else torch.as_tensor(std,dtype=torch.float32).clone())
        self.norm=nn.LayerNorm(encoder.config.hidden_size)
        self.project=nn.Sequential(nn.Linear(encoder.config.hidden_size+width,256),nn.GELU(),nn.Dropout(.1))
        self.joint=nn.Linear(256,16)

    def forward(self,input_ids,attention_mask,numeric,token_type_ids=None):
        states=self.encoder(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).last_hidden_state
        mask=attention_mask.unsqueeze(-1).to(states.dtype)
        pooled=(states*mask).sum(1)/mask.sum(1).clamp(min=1)
        numbers=((numeric-self.numeric_mean)/self.numeric_std).clamp(-6,6)
        return self.joint(self.project(torch.cat((self.norm(pooled),numbers),dim=1)))


def from_initial(directory,prior):
    old,tokenizer,recipe=initial.load(directory)
    model=Network(old.encoder,len(old.numeric_mean),old.numeric_mean,old.numeric_std)
    model.norm=old.norm;model.project=old.project
    nn.init.normal_(model.joint.weight,std=.02)
    with torch.no_grad():model.joint.bias.copy_(torch.as_tensor(np.log(prior),dtype=torch.float32))
    for p in model.encoder.parameters():p.requires_grad_(False)
    if len(model.encoder.encoder.layer)<4:raise ValueError('Need at least four encoder layers')
    for layer in model.encoder.encoder.layer[-4:]:
        for p in layer.parameters():p.requires_grad_(True)
    return model,tokenizer,recipe


def masked(logits,valid):
    if logits.ndim!=2 or logits.shape[1]!=16 or valid.shape!=(len(logits),):raise ValueError('Aligned16-state observations required')
    impossible=(~valid.bool())[:,None]&(torch.arange(16,device=logits.device)[None,:]>=8)
    return logits.float().masked_fill(impossible,-1e4)


def loss(logits,labels,valid):
    if ((labels>=8)&~valid.bool()).any():raise ValueError('Invalid answer cannot be correct')
    value=masked(logits,valid);probability=value.softmax(1);bits=BITS.to(value.device)
    categorical=nn.functional.cross_entropy(value,labels)
    marginal=nn.functional.binary_cross_entropy((probability@bits).clamp(1e-6,1-1e-6),bits[labels])
    return categorical+.5*marginal


def probabilities(logits,valid,prior):
    value=np.asarray(logits,dtype=float).copy();value-=value.max(1,keepdims=True)
    p=np.exp(value);p/=p.sum(1,keepdims=True)
    p=(1-PRIOR_WEIGHT)*p+PRIOR_WEIGHT*np.asarray(prior)
    p[~np.asarray(valid,dtype=bool),8:]=0.
    return p/p.sum(1,keepdims=True)


def save(model,tokenizer,directory,recipe):
    return initial.save(model,tokenizer,directory,{**recipe,'architecture':ARCHITECTURE})


def load(directory):
    directory=Path(directory);recipe=json.loads((directory/'recipe.json').read_bytes())
    if recipe['architecture']!=ARCHITECTURE:raise ValueError('Unknown joint observer architecture')
    tensors={}
    for name,digest in recipe['weights'].items():
        if initial.hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed joint observer weights')
        tensors.update(initial.load_file(str(directory/name)))
    state={key:torch.cat([tensors[n] for n in parts],dim=0) if len(parts)>1 else tensors[parts[0]]
           for key,parts in recipe['tensor_parts'].items()}
    encoder=initial.AutoModel.from_config(initial.AutoConfig.from_pretrained(directory,local_files_only=True),trust_remote_code=False)
    model=Network(encoder,recipe['numeric_width']);model.load_state_dict(state);model.eval()
    tokenizer=initial.AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False,use_fast=False)
    return model,tokenizer,recipe


def predict(model,tensors,device,batch=4):
    model.eval();outputs=[]
    with torch.inference_mode():
        for start in range(0,len(tensors['input_ids']),batch):
            values={k:v[start:start+batch] for k,v in tensors.items()}
            length=max(1,int(values['attention_mask'].sum(1).max()))
            values={k:(v[:,:length] if k!='numeric' else v).to(device) for k,v in values.items()}
            outputs.append(model(**values).float().cpu().numpy())
    return np.concatenate(outputs)
