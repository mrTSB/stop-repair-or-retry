"""Registered local adaptation for paired science outcomes, with excluded folds.

Only the last four observer layers and its heads are trained. The compact
solver and all response banks stay fixed. This first stage fits three excluded
question models, not a deployable full model. Its screen evaluates conditional
last-step POMDP decisions; a passing result still needs full fitting, full-panel
development selection, complete cost auditing and prospective validation.
"""
import argparse
import gc
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from transformers import get_linear_schedule_with_warmup

import fit_expanded1280_completion as source
import joint_outcome_observer as network

io,fixed,nested=source.io,source.fixed,source.nested
core=source.runtime.core
SCHEMA='local-joint16-observer-science1280-TRAIN-v1'
EPOCHS,BATCH,SEED=2,4,2381128
PENALTY=.02


def inputs(args):
    spec=io.read(args.expanded_fit/'registration.json');done=io.read(args.expanded_fit/'complete.json')
    old=SimpleNamespace(**{k:Path(v) for k,v in spec['paths'].items()})
    rows,trees,observer,data,_,expected=source.inputs(old)
    if (expected!=spec or done['registration_sha256']!=io.sha(args.expanded_fit/'registration.json')
            or done['schema']!=source.SCHEMA or done['development_or_test_read']):raise ValueError('Require sealed identical1280 TRAIN data')
    io.verify(spec['source_files']);io.verify(spec['input_evidence'])
    io.verify({str(args.expanded_fit/n):h for n,h in done['digests'].items()})
    evidence={**spec['input_evidence'],**{str((args.expanded_fit/n).resolve()):h for n,h in done['digests'].items()},
        str((args.expanded_fit/'complete.json').resolve()):io.sha(args.expanded_fit/'complete.json')}
    second=io.read(old.second_features/'registration.json');original=Path(second['paths']['original_neural'])
    metadata={};slots={}
    for folder,name in ((original,'dense-neural.npz'),(old.first_features,'neural.npz'),(old.second_features,'neural.npz')):
        records=[io.json.loads(s) for s in (folder/'metadata.jsonl').read_bytes().splitlines()]
        with np.load(folder/name,allow_pickle=False) as archive:
            for i,r in enumerate(records):
                key=r['id'],r['path']
                if key in metadata:raise ValueError('Duplicate paid observation')
                metadata[key]=r;slots[key]=int(archive['token_slots'][i])
        for filename in ('metadata.jsonl',name):evidence[str((folder/filename).resolve())]=io.sha(folder/filename)
    entries=[]
    for row in rows:
        for path in ('','repair','retry'):
            paid={p:trees[row['id']][p] for p in ('',path)}
            entry=observer.entry({**row,'answer':''},path,paid)
            if fixed.fingerprint(entry)!=metadata[row['id'],path]['input_sha256']:raise ValueError('Paid input changed')
            entries.append({k:entry[k] for k in ('left','right','numeric')})
    if len(entries)!=3840 or len(metadata)!=3840:raise ValueError('All1280 questions and three prefixes required')
    with np.load(args.expanded_fit/'private-oof.npz',allow_pickle=False) as archive:
        control=archive[nested.FAMILIES[0]].copy()
    if control.shape!=(3840,16):raise ValueError('Require all prior excluded-question forecasts')
    if args.device=='mps' and not torch.backends.mps.is_available():raise ValueError('Local MPS unavailable')
    record={'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{'expanded_fit':str(args.expanded_fit.resolve()),'output':str(args.output.resolve())},
        'initial_directory':str(Path(observer.directory).resolve()),'initial_parameters':observer._recipe['parameters'],
        'initial_recipe_sha256':io.sha(Path(observer.directory)/'recipe.json'),'architecture':network.ARCHITECTURE,
        'device':args.device,'epochs':EPOCHS,'batch':BATCH,'threads':2,'max_length':512,'seed':SEED,
        'training_questions':1280,'nodes':3840,'outer_question_folds':3,'full_model_fitted':False,
        'trainable_last_encoder_layers':4,'encoder_lr':2e-6,'heads_lr':1e-4,'weight_decay':.01,
        'gradient_clip':1.,'warmup_fraction':.1,'loss':'joint16 CE plus0.5 mean four-marginal BCE; mask known-invalid current truth',
        'class_reweighting':False,'prior_weight':network.PRIOR_WEIGHT,'prior_fits_only_excluded_fold_training':True,
        'entry_sha256':fixed.fingerprint(entries),'records_sha256':fixed.fingerprint(data['records']),
        'labels_sha256':fixed.array_sha(data['labels']),'valid_sha256':fixed.array_sha(data['valid']),
        'solver_unchanged':True,'cloud_calls':0,'cloud_training':False,'development_or_test_read':False,
        'screen':{'conditional_child_accuracy_gain_at_least':.02,'conditional_compute_ratio_at_most':1.10,
            'penalty':PENALTY,'controls':'Same1280 question-excluded raw1280 forest predictions',
            'scope':'TRAIN conditional last-decision potential only; not a full controller or fresh claim'},
        'resume':'Completed fold receipts skip training; sealed training checkpoint allows prediction-only resume; interrupted unsealed training fails closed',
        'model_selection':'One fixed recipe and fixed two epochs; no selection on held labels',
        'cost_scope':'Full paid prefix solver tokens plus one observer and chosen terminal continuation; offline training excluded'}
    return rows,trees,data,entries,slots,control,record


def fold_masks(records,fold):
    held=np.asarray([r['fold']==fold for r in records]);train=~held
    receipt=nested.group_receipt(records,train,held)
    if not train.any() or not held.any():raise ValueError('Nonempty question folds required')
    return train,held,receipt


def train_fold(args,tensors,data,record,fold):
    train,held,receipt=fold_masks(data['records'],fold);folder=args.output/f'fold{fold}';folder.mkdir(exist_ok=True)
    target=folder/'complete.json'
    if target.exists():
        done=io.read(target)
        if done['registration_sha256']!=args.registration_sha256 or done['fold_receipt']!=receipt:raise ValueError('Changed completed fold')
        io.verify({str(folder/n):h for n,h in done['digests'].items()})
        with np.load(folder/'held.npz',allow_pickle=False) as archive:return held,archive['probability'].copy(),done['parameters']
    counts=np.bincount(data['labels'][train],minlength=16)+1.;prior=counts/counts.sum()
    training_receipt=folder/'training-receipt.json';checkpoint=folder/'model'
    if not training_receipt.exists():
        if (folder/'training-attempt.json').exists():raise ValueError('Unsealed interrupted training; preserve and inspect')
        io.write(folder/'training-attempt.json',{'registration_sha256':args.registration_sha256,'fold':fold,'receipt':receipt,'seed':SEED+fold})
        random.seed(SEED+fold);np.random.seed(SEED+fold);torch.manual_seed(SEED+fold)
        model,tokenizer,initial=network.from_initial(record['initial_directory'],prior)
        if initial['parameters']!=record['initial_parameters']:raise ValueError('Initial architecture changed')
        model=model.to(args.device)
        optimizer=torch.optim.AdamW([
            {'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':2e-6},
            {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad],'lr':1e-4}],
            weight_decay=.01,foreach=False)
        indices=np.flatnonzero(train);keys=list(tensors)
        dataset=TensorDataset(*(tensors[k][indices] for k in keys),torch.tensor(data['labels'][train]),torch.tensor(data['valid'][train]))
        loader=DataLoader(dataset,batch_size=BATCH,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED+fold))
        total_steps=len(loader)*EPOCHS;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,total_steps//10),total_steps)
        elapsed=[];steps=0
        for epoch in range(EPOCHS):
            started=time.monotonic();model.train()
            for step,values in enumerate(loader):
                batch={k:v for k,v in zip(keys,values[:-2])};length=max(1,int(batch['attention_mask'].sum(1).max()))
                batch={k:(v[:,:length] if k!='numeric' else v).to(args.device) for k,v in batch.items()}
                labels,valid=(v.to(args.device) for v in values[-2:]);optimizer.zero_grad(set_to_none=True)
                loss=network.loss(model(**batch),labels,valid)
                if not torch.isfinite(loss):raise ValueError('Nonfinite joint-observer training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();steps+=1
                if (step+1)%100==0:print({'fold':fold,'epoch':epoch+1,'step':step+1,'steps_per_epoch':len(loader),'loss':float(loss.detach().cpu())},flush=True)
            elapsed.append(time.monotonic()-started);print({'fold':fold,'epoch_complete':epoch+1,'seconds':elapsed[-1]},flush=True)
        recipe=network.save(model,tokenizer,checkpoint,{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
            'fold':fold,'fold_receipt':receipt,'prior':prior.tolist(),'max_length':512,'initial_directory':record['initial_directory'],
            'input_profile':'unchanged-full-options-openstax-v1','epochs':EPOCHS,'trainable_last_encoder_layers':4})
        io.write(training_receipt,{'registration_sha256':args.registration_sha256,'fold_receipt':receipt,
            'recipe_sha256':io.sha(checkpoint/'recipe.json'),'training_steps':steps,'epoch_seconds':elapsed})
        del model,optimizer,loader,dataset,scheduler,loss,batch,values
        gc.collect()
        if args.device=='mps':torch.mps.empty_cache()
    sealed=io.read(training_receipt)
    if (sealed['registration_sha256']!=args.registration_sha256 or sealed['fold_receipt']!=receipt
            or sealed['recipe_sha256']!=io.sha(checkpoint/'recipe.json')):raise ValueError('Changed training checkpoint')
    model,_,recipe=network.load(checkpoint)
    if recipe['fold_receipt']!=receipt or recipe['prior']!=prior.tolist():raise ValueError('Changed held prediction model')
    predicted=network.predict(model.to(args.device),{k:v[held] for k,v in tensors.items()},args.device,BATCH)
    probability=network.probabilities(predicted,data['valid'][held],prior)
    io.npz(folder/'held.npz',{'logits':predicted,'probability':probability,'row_indices':np.flatnonzero(held)})
    del model;gc.collect()
    if args.device=='mps':torch.mps.empty_cache()
    digests={str(p.relative_to(folder)):io.sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='complete.json'}
    io.write(target,{'registration_sha256':args.registration_sha256,'fold_receipt':receipt,'parameters':recipe['parameters'],'digests':digests})
    print({'fold':fold,'held_observations':int(held.sum()),'fold_complete':True},flush=True)
    return held,probability,recipe['parameters']


def conditional_costs(rows,trees,held_fold):
    sums={s:np.zeros((3,16)) for s in ('chemistry','physics')};counts={s:np.zeros((3,16)) for s in sums}
    for row in rows:
        if fixed.refined.fold_for(row['id'])==held_fold:continue
        tree=trees[row['id']]
        for path in ('repair','retry'):
            state=core.state(*[core.plain.correct(tree[p],row['answer']) for p in (path,*[path+'/'+a for a in ('repair','retry','complete')])])
            for action,name in enumerate(('repair','retry','complete')):
                node=tree[path+'/'+name];sums[row['subject']][action,state]+=4*(node['input_tokens']+node['output_tokens'])/3.09
                counts[row['subject']][action,state]+=1
    total=sum(sums.values());number=sum(counts.values());pooled=np.divide(total,number,out=np.full_like(total,650.),where=number>0)
    return {s:(sums[s]+4*pooled)/(counts[s]+4) for s in sums}


def conditional(rows,trees,data,slots,p,parameters):
    by_id={r['id']:r for r in rows};costs={fold:conditional_costs(rows,trees,fold) for fold in range(3)};items=[]
    for i,r in enumerate(data['records']):
        if not r['path']:continue
        qid,path,subject=r['id'],r['path'],r['subject'];tree=trees[qid];cost=costs[r['fold']][subject]
        values=[p[i]@core.CURRENT,*[(p[i]@(core.AFTER[a]-PENALTY*cost[a]/1000)) for a in range(3)]]
        action=int(np.argmax(values));final=path if action==0 else path+'/'+core.ACTIONS[action]
        visited=['',path]+([] if action==0 else [final]);tokens=sum(tree[k]['input_tokens']+tree[k]['output_tokens'] for k in visited)
        items.append({'id':qid,'path':path,'action':core.ACTIONS[action],'correct':core.plain.correct(tree[final],by_id[qid]['answer']),
            'compute':(4*tokens+parameters/1e9*slots[qid,path])/3.09})
    return items


def train(args,rows,trees,data,entries,slots,control,record):
    torch.set_num_threads(2)
    tokenizer=network.initial.AutoTokenizer.from_pretrained(record['initial_directory'],local_files_only=True,trust_remote_code=False,use_fast=False)
    tensors=network.initial.encode(tokenizer,entries,512)
    expected=np.array([slots[r['id'],r['path']] for r in data['records']])
    np.testing.assert_array_equal(tensors['attention_mask'].sum(1).numpy(),expected)
    predicted=np.full((3840,16),np.nan);parameters=set()
    for fold in range(3):
        held,p,count=train_fold(args,tensors,data,record,fold);predicted[held]=p;parameters.add(count)
    if len(parameters)!=1 or not np.isfinite(predicted).all():raise ValueError('Incomplete aligned observer predictions')
    parameters=parameters.pop();old=conditional(rows,trees,data,slots,control,record['initial_parameters']);new=conditional(rows,trees,data,slots,predicted,parameters)
    if [(r['id'],r['path']) for r in old]!=[(r['id'],r['path']) for r in new]:raise ValueError('Changed conditional evaluation cohort')
    metrics=lambda items:{'n':len(items),'correct':sum(int(v['correct']) for v in items),'accuracy':float(np.mean([v['correct'] for v in items])),
                          'mean_compute':float(np.mean([v['compute'] for v in items]))}
    before,after=metrics(old),metrics(new);delta=after['accuracy']-before['accuracy'];ratio=after['mean_compute']/before['mean_compute']
    screen={'conditional_accuracy_gain':delta,'conditional_compute_ratio':ratio,'passed':delta>=.02 and ratio<=1.10}
    groups={r['id']:[] for r in old}
    for i,r in enumerate(old):groups[r['id']].append(i)
    if len(groups)!=1280 or any(len(g)!=2 for g in groups.values()):raise ValueError('All1280 paired child contexts required')
    differences=np.array([np.mean([int(new[i]['correct'])-int(old[i]['correct']) for i in g]) for g in groups.values()])
    draws=np.random.default_rng(238031).integers(0,1280,(2000,1280));screen['paired_question_accuracy_CI95']=np.quantile(differences[draws].mean(1),[.025,.975]).tolist()
    io.npz(args.output/'oof.npz',{'probability':predicted});io.write(args.output/'private-conditional-items.json',{'control':old,'candidate':new})
    report={'schema':SCHEMA,'control':before,'candidate':after,'screen':screen,'parameters':parameters,
        'forecast_losses':{'control':{k:float(v.mean()) for k,v in fixed.losses(data['labels'],control).items()},
            'candidate':{k:float(v.mean()) for k,v in fixed.losses(data['labels'],predicted).items()}},
        'full_model_fitted':False,'full_controller_claim':False,'fresh_validation':False,'development_or_test_read':False,
        'cloud_calls':0,'local_training':True,'offline_training_excluded_from_inference_proxy':True}
    io.write(args.output/'summary.json',report);io.verify(record['source_files']);io.verify(record['input_evidence'])
    digests={str(p.relative_to(args.output)):io.sha(p) for p in args.output.rglob('*') if p.is_file() and p!=args.output/'complete.json' and not p.name.startswith('.')}
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'screen_passed':screen['passed'],'full_model_fitted':False,'digests':digests})
    print({'control':before,'candidate':after,'screen':screen},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','train'))
    for name in ('expanded-fit','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=('mps','cpu'),default='mps');parser.add_argument('--registration-sha256');args=parser.parse_args()
    with fixed.training_only_reads():
        with fixed.guarded.no_model_calls():bundle=inputs(args)
        record=bundle[-1]
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'training_started':False});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Changed registered local training')
        with io.lock(args.output/'.training.lock'):
            if (args.output/'complete.json').exists():raise ValueError('Preserve completed local experiment')
            train(args,*bundle)


if __name__=='__main__':main()
