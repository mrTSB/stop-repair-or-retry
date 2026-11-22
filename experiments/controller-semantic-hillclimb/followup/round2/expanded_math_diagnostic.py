"""Nested TRAIN-only outcome forecasts after tripling paired math data.

Compare on the same original256 held math questions. Every original and new
question is excluded from its own forecast, including inner model selection.
"""
import argparse
import hashlib
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

import math_auxiliary_features as features
import reward_math_diagnostic as conditional

math = features.math
io, fixed = features.io, features.fixed
SCHEMA = 'expanded-math768-nested-outcome-TRAIN-v1'


def reference_join(public, source):
    import pyarrow as pa
    if len({r['id'] for r in public}) != len(public) or any(r['answer'] != '' for r in public):
        raise ValueError('Unique blank-reference selected TRAIN rows required')
    indices = []
    for row in public:
        index = int(row['id'].removeprefix('gsm8k-train-'))
        if row['id'] != f'gsm8k-train-{index:05d}' or index < 0: raise ValueError('Changed pinned math row identity')
        indices.append(index)
    with pa.memory_map(str(source), 'r') as stream:
        # Only the selected TRAIN rows become Python answer objects.
        selected = pa.ipc.open_stream(stream).read_all().take(pa.array(indices, type=pa.int64())).select(['question','answer']).to_pylist()
    result = []
    for row, original in zip(public, selected):
        if row['question'] != original['question'] or '####' not in original['answer']:
            raise ValueError('Changed selected question or malformed official reference; no replacement')
        answer = original['answer'].rsplit('####',1)[1].strip()
        if not answer: raise ValueError('Missing selected TRAIN answer')
        result.append({**row,'answer':answer})
    return result


def table(rows, trees, metadata, values):
    features.original.probe.validate(values,len(metadata)); records,x,y,valid = [],[],[],[]
    expected = [(r['id'],p) for r in rows for p in ('','repair','retry')]
    if [(r['id'],r['path']) for r in metadata] != expected: raise ValueError('Exact paid-prefix alignment required')
    i = 0
    for row in rows:
        for path in ('','repair','retry'):
            paid = {p:trees[row['id']][p] for p in ('',path)}
            if metadata[i]['prompt_sha256'] != paid[path]['prompt_sha256']: raise ValueError('Changed paid prompt')
            neural = tuple(values[k][i] for k in ('state','plain','guided'))
            x.append(math.core.observables({**row,'answer':''},path,paid,neural))
            prefix = path+'/' if path else ''
            y.append(math.core.state(*[math.core.correct(trees[row['id']][p],row['answer'])
                for p in (path,prefix+'repair',prefix+'retry')]))
            valid.append(bool(paid[path]['answer']))
            records.append({'id':row['id'],'path':path,'subject':'math',
                'fold':int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)%3})
            i += 1
    return {'records':records,'x':np.asarray(x),'y':np.asarray(y),'valid':np.asarray(valid)}


def outer(data, fold):
    splits = np.asarray([r['fold'] for r in data['records']]); fit,held = splits != fold,splits == fold
    receipt = math.nested.group_receipt(data['records'],fit,held)
    setting,search = math.select(data,fit)
    model = math.fit(data['x'][fit],data['y'][fit],setting)
    return held,math.predict(model,data['x'][held],data['valid'][held]),model,{**receipt,'setting':setting,'inner_search':search}


def inputs(args):
    spec = io.read(args.features/'registration.json')
    source = SimpleNamespace(**{k:Path(v) for k,v in spec['paths'].items()},registration_sha256=io.sha(args.features/'registration.json'))
    bundle,expected = features.inputs(source)
    if expected != spec: raise ValueError('Changed sealed math feature contract')
    io.verify(spec['source_files']);io.verify(spec['input_evidence']);values,done = features.verify(source,bundle,spec)
    old_spec = io.read(source.math_diagnostic/'registration.json')
    old_args = SimpleNamespace(**{k:Path(v) for k,v in old_spec['paths'].items()})
    old_data,_ = math.inputs(old_args); mask = np.asarray([r['subject']=='math' for r in old_data['records']])
    old = {'records':[r for r,keep in zip(old_data['records'],mask) if keep],
        **{k:old_data[k][mask].copy() for k in ('x','y','valid')}}
    base_spec = io.read(old_args.base_fit/'registration.json')
    rows,trees = math.source.plain_data(Path(base_spec['paths']['data']),'train')
    rows = [r for r in rows if r['subject']=='math']; trees = {r['id']:trees[r['id']] for r in rows}
    new_rows = reference_join(bundle['rows'],features.pairs.contract.reserved.reserve.previous.MATH)
    new = table(new_rows,bundle['trees'],bundle['metadata'],values)
    data = {'records':old['records']+new['records'], **{k:np.concatenate([old[k],new[k]]) for k in ('x','y','valid')}}
    if data['x'].shape != (2304,130) or len({r['id'] for r in data['records']}) != 768: raise ValueError('Require all768 math TRAIN questions')
    with np.load(source.math_diagnostic/'private-oof.npz',allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['indices'],np.flatnonzero(mask));control = archive['specialized'].copy()
    observer = bundle['observer']; by_id = {r['id']:r for r in rows}
    entries = [observer.entry({**by_id[r['id']],'answer':''},r['path'],{p:trees[r['id']][p] for p in ('',r['path'])}) for r in old['records']]
    tokenizer = AutoTokenizer.from_pretrained(observer.directory,local_files_only=True,trust_remote_code=False,use_fast=False)
    encoded = features.original.previous.network.encode(tokenizer,entries,512)
    slots = {(r['id'],r['path']):int(n) for r,n in zip(old['records'],encoded['attention_mask'].sum(1))}
    slots.update({(r['id'],r['path']):int(n) for r,n in zip(new['records'],values['token_slots'])})
    evidence = {**spec['input_evidence'], **{str((args.features/k).resolve()):v for k,v in done['digests'].items()},
        str((args.features/'complete.json').resolve()):io.sha(args.features/'complete.json')}
    record = {'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{k:str(getattr(args,k).resolve()) for k in ('features','output')},
        'math_training_questions':768,'original_common_held_questions':256,'outer_folds':3,'inner_folds':3,
        'settings':list(math.SETTINGS),'prior_weight':.01,'state_count':8,
        'records_sha256':fixed.fingerprint(data['records']),'features_sha256':fixed.array_sha(data['x']),
        'labels_sha256':fixed.array_sha(data['y']),'valid_sha256':fixed.array_sha(data['valid']),
        'control':'Retained math-specialized nested OOF procedure fitted on original256 math TRAIN',
        'screen':'Lower joint loss and mean-three marginal Brier, at least2pp conditional-decision gain, mean compute at most1.10x control',
        'conditional_policy_penalty':.02,'conditional_costs':'Fit excluding outer-held questions for each data regime',
        'compute':'Same frozen observer; actual token slots and all selected solver tokens charged',
        'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,'fresh_validation':False,
        'reference_scope':'Only original256 TRAIN and all512 preselected completed auxiliary TRAIN rows'}
    return rows+new_rows,{**trees,**bundle['trees']},data,old,control,slots,observer._recipe['parameters'],record


def diagnose(args,rows,trees,data,old,control,slots,parameters,record):
    p = np.full((2304,8),np.nan); receipts = []
    for fold in range(3):
        held,prob,model,receipt = outer(data,fold); p[held] = prob; receipts.append(receipt)
        io.atomic_bytes(args.output/f'outer{fold}.pkl',pickle.dumps(model,protocol=5))
        print({'fold':fold,'setting':receipt['setting']},flush=True)
    if not np.isfinite(p).all(): raise ValueError('Incomplete excluded-question forecasts')
    before = conditional.conditional(rows[:256],trees,old,control,slots,parameters)
    after = conditional.conditional(rows,trees,data,p,slots,parameters)[:512]
    if [(r['id'],r['path']) for r in before] != [(r['id'],r['path']) for r in after]: raise ValueError('Changed common held decisions')
    metrics = lambda items:{'n':len(items),'correct':sum(int(r['correct']) for r in items),
        'accuracy':float(np.mean([r['correct'] for r in items])),'mean_compute':float(np.mean([r['compute'] for r in items]))}
    a,b = metrics(before),metrics(after)
    losses = {name:{k:float(v.mean()) for k,v in math.losses(old['y'],prob).items()} for name,prob in (('control',control),('expanded',p[:768]))}
    checks = {k:losses['expanded'][k]<losses['control'][k] for k in ('joint_log_loss','mean_three_marginal_Brier')}
    checks.update(conditional_gain_at_least2pp=b['accuracy']-a['accuracy']>=.02,compute_at_most1p10=b['mean_compute']<=1.10*a['mean_compute'])
    delta = np.asarray([int(y['correct'])-int(x['correct']) for x,y in zip(before,after)]).reshape(256,2).mean(1)
    draws = np.random.default_rng(238031).integers(0,256,(2000,256))
    result = {'schema':SCHEMA,'control':a,'candidate':b,'forecast_losses':losses,'screen_checks':checks,'screen_passed':all(checks.values()),
        'paired_question_accuracy_CI95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
        'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,'fresh_validation':False,'full_controller_claim':False}
    io.npz(args.output/'private-oof.npz',{'probability':p});io.write(args.output/'fold-receipts.json',receipts)
    io.write(args.output/'private-conditional-items.json',{'control':before,'candidate':after});io.write(args.output/'summary.json',result)
    io.verify(record['source_files']);io.verify(record['input_evidence'])
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,'screen_passed':result['screen_passed'],
        'digests':{v.name:io.sha(v) for v in args.output.iterdir() if v.is_file() and not v.name.startswith('.')}})
    print(result,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','diagnose'))
    for key in ('features','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--registration-sha256');args = p.parse_args()
    with fixed.training_only_reads(),fixed.guarded.no_model_calls(),threadpool_limits(limits=1):
        bundle = inputs(args);record = bundle[-1]
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json')},flush=True);return
        if not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256 or io.read(args.output/'registration.json')!=record:
            raise ValueError('Changed expanded math diagnostic')
        with io.lock(args.output/'.run.lock'):
            if any(v.name not in ('registration.json','.run.lock') for v in args.output.iterdir()):raise ValueError('Preserve prior fitting outputs')
            diagnose(args,*bundle)


if __name__=='__main__':main()
