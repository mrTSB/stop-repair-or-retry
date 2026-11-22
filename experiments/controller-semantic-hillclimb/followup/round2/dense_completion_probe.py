"""TRAIN-only same-forward project256 observation probe, not a controller result."""
import numpy as np
import torch

import completion_v4_diagnostic as fixed

PROFILE='old-V1-project256-same-forward-TRAIN-probe-v1'
VARIANTS=('raw20_control','raw20_plus_project256')
PARITY={'absolute_tolerance':.005,'relative_tolerance':.001,
    'rationale':'Conservative float32 CPU-versus-MPS and padded-batch reduction tolerance; at most0.005 absolute plus0.1percent relative per raw logit. This is a prespecified engineering bound, not an empirically established platform error bound.',
    'rule':'Every one of768×20 logits must pass; report all differences, never filter rows or retune tolerance after extraction.'}


def capture(model,tensors):
    """One encoder forward; project hook only copies an existing activation."""
    captured=[];model.eval()
    def hook(module,args,output):captured.append(output.detach().float().cpu().numpy().copy())
    handle=model.project.register_forward_hook(hook)
    try:
        with torch.inference_mode():outputs=model(**tensors)
    finally:handle.remove()
    if len(captured)!=1 or captured[0].shape!=(len(tensors['input_ids']),256):
        raise ValueError('Expected exactly one project256 activation per shared forward')
    if len(outputs)!=3:raise ValueError('Expected exactly the three frozen raw heads')
    values={name:value.detach().float().cpu().numpy().copy() for name,value in zip(('state','plain','guided'),outputs)}
    values['joint256']=captured[0]
    values['token_slots']=tensors['attention_mask'].sum(1).cpu().numpy().astype(np.int64)
    validate(values,len(captured[0]));return values


def validate(values,n):
    if set(values)!={'state','plain','guided','joint256','token_slots'}:raise ValueError('Unexpected same-forward output fields')
    for name,width in (('state',4),('plain',8),('guided',8),('joint256',256)):
        value=np.asarray(values[name])
        if value.shape!=(n,width) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid dense observation shape/dtype/finite values')
    slots=np.asarray(values['token_slots'])
    if slots.shape!=(n,) or slots.dtype!=np.int64 or np.any(slots<1) or np.any(slots>512):
        raise ValueError('Invalid actual unpadded observation token slots')


def parity(values,old):
    raw=np.concatenate([values[n] for n in ('state','plain','guided')],axis=1)
    reference=np.asarray(old,dtype=np.float64)
    if reference.shape!=raw.shape or not np.isfinite(reference).all():raise ValueError('Wrong old-cache alignment')
    differences=raw.astype(np.float64)-reference
    tolerance=PARITY['absolute_tolerance']+PARITY['relative_tolerance']*np.abs(reference)
    failure=np.abs(differences)>tolerance
    return {'all_logits_within_tolerance':not bool(failure.any()),'logits':raw.size,'failed_logits':int(failure.sum()),
        'failed_rows':int(failure.any(1).sum()),'maximum_absolute_difference':float(np.abs(differences).max()),
        'mean_absolute_difference':float(np.abs(differences).mean()),
        'absolute_difference_quantiles':np.quantile(np.abs(differences),[.5,.9,.99,1.]).tolist(),
        'difference_sha256':fixed.array_sha(differences),'tolerance':PARITY},differences


def table(rows,trees,metadata,values):
    validate(values,len(metadata))
    if len(metadata)!=768:raise ValueError('Use all768 registered TRAIN science prefixes')
    lookup={};joint={}
    for i,item in enumerate(metadata):
        key=item['id'],item['path']
        if key in lookup:raise ValueError('Duplicate paid-prefix feature')
        if item['prompt_sha256']!=trees[item['id']][item['path']]['prompt_sha256']:
            raise ValueError('Changed dense paid-prompt alignment')
        lookup[key]=(*[values[n][i] for n in ('state','plain','guided')],int(values['token_slots'][i]))
        joint[key]=values['joint256'][i]
    # Reuse exactly the fixed numeric/raw20 feature and sixteen-label definition;
    # supplying the same raw20 twice cannot introduce the other neural model.
    data=fixed.table(rows,trees,lookup,lookup)
    control=data['features'][fixed.VARIANTS[0]]
    dense=np.stack([joint[r['id'],r['path']] for r in data['records']])
    data['features']={VARIANTS[0]:control,VARIANTS[1]:np.concatenate((control,dense),axis=1)}
    if control.shape!=(768,130) or data['features'][VARIANTS[1]].shape!=(768,386):
        raise ValueError('Changed fixed130+256 feature profile')
    return data


def diagnostics(data,predictions):
    if set(predictions)!=set(VARIANTS):raise ValueError('Only the two prespecified feature variants are supported')
    rows=data['records'];ids=sorted({r['id'] for r in rows})
    groups=[[i for i,r in enumerate(rows) if r['id']==qid] for qid in ids]
    if len(ids)!=256 or any(len(g)!=3 for g in groups):raise ValueError('Require256 complete three-node question groups')
    scores={name:fixed.losses(data['labels'],p) for name,p in predictions.items()}
    draws=np.random.default_rng(fixed.SEED).integers(0,len(ids),size=(fixed.BOOTSTRAPS,len(ids)))
    comparisons={}
    for metric in scores[VARIANTS[0]]:
        delta=scores[VARIANTS[1]][metric]-scores[VARIANTS[0]][metric]
        per_question=np.asarray([delta[g].mean() for g in groups]);boot=per_question[draws].mean(1)
        comparisons[metric]={'dense_minus_control':float(per_question.mean()),
            'paired_question_CI95':np.quantile(boot,[.025,.975]).tolist()}
    panels={}
    for subject in ('all','chemistry','physics'):
        mask=np.asarray([subject=='all' or r['subject']==subject for r in rows])
        panels[subject]={name:{m:float(v[mask].mean()) for m,v in s.items()} for name,s in scores.items()}
    return {'panels':panels,'comparisons':comparisons,
        'screen_passed':all(comparisons[m]['dense_minus_control']<0 for m in ('joint_log_loss','mean_four_marginal_Brier')),
        'screen_scope':'Both point estimates must improve; paired question CIs are descriptive with fitted folds held fixed',
        'encoder_excluded_all_new512_questions':True,'second_stage_question_excluded':True,
        'neural_refitting_or_nested_CV_claim':False,'controller_accuracy_claim':False,'fresh_validation':False}
