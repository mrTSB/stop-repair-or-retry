"""Same-forward full normalized semantic representation, TRAIN-only probe."""
import numpy as np
import dense_completion_probe as previous

fixed = previous.fixed
PROFILE = 'old-V1-norm1024-same-forward-TRAIN-probe-v1'
VARIANTS = ('raw20_plus_project256', 'raw20_project256_full1024')
PARITY = previous.PARITY
parity = previous.parity


def capture(model, tensors):
    captured = []
    handle = model.norm.register_forward_hook(lambda module, args, output: captured.append(output.detach().float().cpu().numpy().copy()))
    try: values = previous.capture(model, tensors)
    finally: handle.remove()
    if len(captured) != 1: raise ValueError('Exactly one normalized representation per shared forward')
    values['text1024'] = captured[0]
    validate(values, len(tensors['input_ids']))
    return values


def validate(values, n):
    if 'text1024' not in values: raise ValueError('Missing full semantic representation')
    previous.validate({k: v for k, v in values.items() if k != 'text1024'}, n)
    v = np.asarray(values['text1024'])
    if v.shape != (n, 1024) or v.dtype != np.float32 or not np.isfinite(v).all():
        raise ValueError('Full normalized representation must be finite float32 Nx1024')


def table(rows, trees, metadata, values):
    validate(values, len(metadata))
    data = previous.table(rows, trees, metadata, {k: v for k, v in values.items() if k != 'text1024'})
    full = {(m['id'], m['path']): values['text1024'][i] for i, m in enumerate(metadata)}
    semantic = np.stack([full[r['id'], r['path']] for r in data['records']])
    compressed = data['features'][VARIANTS[0]]
    data['features'] = {VARIANTS[0]: compressed, VARIANTS[1]: np.concatenate((compressed, semantic), axis=1)}
    if data['features'][VARIANTS[1]].shape != (768, 1410): raise ValueError('Feature dimensions changed')
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
