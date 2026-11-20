"""Tune POMDP A and selective-judge C on development, then score a sealed study holdout."""

import argparse
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from acquire import CHECKPOINTS, HERE, JUDGE, JUDGE_CAP, JUDGE_DIGEST, MODEL, MODEL_DIGEST, PARENT, PATHS, ROOT, SEED, SOLVER_CAP, digest, judge_prompt, paired_archive, parse_answer, read, solver_prompt
from belief import BeliefPlanner, evaluate, fit
from hillclimb import new_run, paired, save
from scoring import correct

PENALTIES = (.0, .015, .035, .07, .12, .2)
STRENGTHS = (16., 64., 160.)
SMOOTHING = (3., 12., 32.)
THRESHOLDS = (-.02, -.005, .0, .005, .02)


def records(source, split):
    questions = read(source / 'splits' / f'{split}-questions.jsonl')
    path = source / f'{split}-trees.jsonl'
    if not path.exists():
        path = path.with_suffix('.jsonl.gz')
    trees = {r['id']: r['tree'] for r in read(path)}
    probes = source / f'{split}-probes.jsonl'
    if not probes.exists():
        probes = probes.with_suffix('.jsonl.gz')
    if probes.exists():
        for observation in read(probes):
            compact = observation['compact']
            previous = trees[observation['id']]['']
            fields = ('text','answer','finish','input_tokens','output_tokens','prompt_sha256')
            matches = all(compact[key]==previous[key] for key in fields)
            if observation['matches_archive'] != matches:
                raise ValueError('Incorrect initial-probe archive match flag')
            if matches:
                previous['confidence'] = compact['confidence']
    path = source / f'{split}-judges.jsonl'
    if not path.exists():
        path = path.with_suffix('.jsonl.gz')
    judges = {(r['id'], r['path']):r['judge'] for r in read(path)}
    if set(trees) != {r['id'] for r in questions} or len(trees)!=len(questions):
        raise ValueError('Question and paired-tree coverage differ')
    allowed = {(r['id'],p) for r in questions for p in CHECKPOINTS}
    if (split!='test' and set(judges)!=allowed) or not set(judges)<=allowed:
        raise ValueError('Judge coverage differs from the decision checkpoints')
    for row in questions:
        tree = trees[row['id']]
        if (split!='test' and set(tree)!=set(PATHS)) or '' not in tree or not set(tree)<=set(PATHS):
            raise ValueError('Incomplete paired tree')
        for path, node in tree.items():
            expected = hashlib.sha256(solver_prompt(row,path,tree).encode()).hexdigest()
            sampled_seed = int(hashlib.sha256(f'{SEED}:{row["id"]}:{path}'.encode()).hexdigest()[:8],16)%(2**31)
            if (node['model']!=MODEL or node['model_digest']!=MODEL_DIGEST or node['prompt_sha256']!=expected
                    or parse_answer(row['domain'],node['text'])!=node['answer']
                    or node['input_tokens']<=0 or not 0<=node['output_tokens']<=SOLVER_CAP):
                raise ValueError(f'Changed solver protocol for {row["id"]}/{path}')
            if path and (node.get('seed')!=sampled_seed or node.get('temperature')!=.65):
                raise ValueError('Changed continuation sampling parameters')
        for path in CHECKPOINTS:
            if (row['id'],path) not in judges:
                continue
            node = judges[row['id'],path]
            expected = hashlib.sha256(judge_prompt(row,path,tree).encode()).hexdigest()
            sampled_seed = int(hashlib.sha256(f'{SEED}:{row["id"]}:{path}'.encode()).hexdigest()[:8],16)%(2**31)
            if (node['model']!=JUDGE or node['model_digest']!=JUDGE_DIGEST or node['prompt_sha256']!=expected
                    or node['input_tokens']<=0 or not 0<=node['output_tokens']<=JUDGE_CAP
                    or node.get('seed')!=sampled_seed or node.get('temperature')!=0):
                raise ValueError(f'Changed judge protocol for {row["id"]}/{path}')
    return questions, trees, judges


def fixed(row, tree, action, depth):
    keys = ['']
    if action != 'stop':
        keys += ['/'.join([action]*i) for i in range(1, depth+1)]
    path = keys[-1]
    tokens = sum(tree[key]['input_tokens']+tree[key]['output_tokens'] for key in keys)
    return {'id':row['id'], 'domain':row['domain'], 'subject':row['subject'],
            'correct':int(correct(tree[path],row['answer'])), 'path':path, 'solver_tokens':tokens,
            'judge_tokens':0, 'compute':tokens*4/3.09, 'solver_calls':len(keys), 'judge_calls':0}


def describe(results):
    summary = {}
    for domain in ('all','math','science','biology','chemistry','physics'):
        items = [r for r in results if domain == 'all' or r['domain']==domain or r['subject']==domain]
        if not items:
            continue
        summary[domain] = {'n':len(items),'correct':sum(r['correct'] for r in items),
                           'accuracy':float(np.mean([r['correct'] for r in items])),
                           'mean_compute':float(np.mean([r['compute'] for r in items])),
                           'mean_solver_tokens':float(np.mean([r['solver_tokens'] for r in items])),
                           'mean_judge_tokens':float(np.mean([r['judge_tokens'] for r in items])),
                           'mean_judge_calls':float(np.mean([r['judge_calls'] for r in items])),
                           'mean_solver_calls':float(np.mean([r['solver_calls'] for r in items]))}
    return summary


def statistics(results):
    return describe(results)['all']


def normalized_hash(path):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix('.jsonl.gz')
    return hashlib.sha256(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()).hexdigest()


def evidence(source):
    files = [*HERE.glob('*.py'),*(source/'splits').glob('*.json*')]
    ordered = sorted(files,key=lambda path:(not path.is_relative_to(HERE),path.name))
    result = {('code/' if path.is_relative_to(HERE) else 'split/')+path.name:digest(path) for path in ordered}
    dependencies = [PARENT/name for name in ('compact.py','collect.py','hillclimb.py','planner.py','observations.py')]
    dependencies += [PARENT/'retry-gate/paired-uplift'/name for name in ('data.py','features.py','gate.py')]
    dependencies += [PARENT/'retry-gate'/name for name in ('evaluate.py','collect_pairs.py','signals.py','token-confidence/probe.py')]
    dependencies += [ROOT/'results/cached-tree-dataset/scoring.py']
    result.update({str(path.relative_to(ROOT)):digest(path) for path in dependencies})
    for split in ('train','dev'):
        for kind in ('trees','judges'):
            result[f'{split}-{kind}'] = normalized_hash(source/f'{split}-{kind}.jsonl')
    result['train-probes'] = normalized_hash(source/'train-probes.jsonl')
    archive = paired_archive()
    result['archive_pairs'] = hashlib.sha256(json.dumps(archive,sort_keys=True).encode()).hexdigest()
    return result


def batch(rows, trees, judges, model, settings, method):
    return [evaluate(row, trees[row['id']], judges, model, settings, method) for row in rows]


def tune(source, output):
    rows, trees, judges = records(source, 'train')
    dev, dev_trees, dev_judges = records(source, 'dev')
    if {r['id'] for r in rows} & {r['id'] for r in dev}:
        raise ValueError('Fit and development question IDs overlap')
    fixed_options = {}
    for name, action in [('Stop','stop'),('Repair','repair'),('Retry','retry')]:
        variants = []
        for depth in ((0,) if action=='stop' else (1,2)):
            result = [fixed(r,dev_trees[r['id']],action,depth) for r in dev]
            metrics = statistics(result)
            variants.append((metrics['accuracy']-.05*metrics['mean_compute']/1000, depth, metrics))
        _, depth, metrics = max(variants, key=lambda item:(item[2]['accuracy'],-item[2]['mean_compute']))
        fixed_options[name] = {'action':action,'depth':depth,'dev':metrics}
    best_fixed = max(fixed_options, key=lambda name:(fixed_options[name]['dev']['accuracy'],-fixed_options[name]['dev']['mean_compute']))
    reference = fixed_options[best_fixed]['dev']
    candidates, selected, fits = [], {}, {}

    def consider(method,key,settings,results):
        metrics = statistics(results)
        utility = metrics['accuracy']-.05*metrics['mean_compute']/1000
        rank = utility,metrics['accuracy'],-metrics['mean_compute']
        candidate = {'family':method,'fit':key,'settings':settings,'dev':metrics,
                     'accuracy_gain_vs_best_fixed':metrics['accuracy']-reference['accuracy'],
                     'compute_saved_vs_best_fixed':reference['mean_compute']-metrics['mean_compute']}
        candidates.append(candidate)
        if method not in selected or rank>selected[method][0]:
            selected[method] = rank,candidate

    for strength, smoothing, richness, initial_mode in product(STRENGTHS, SMOOTHING, ('linear','context'),('binned','calibrated')):
        key = f'{strength}:{smoothing}:{richness}:{initial_mode}'
        model = fit(rows, trees, judges, strength, smoothing,richness,initial_mode)
        fits[key] = model
        for penalty, science_weight in product(PENALTIES, (.6,1.,1.6)):
            settings = {'penalty':penalty,'science_weight':science_weight}
            for method in ('A','C'):
                thresholds = THRESHOLDS if method=='C' else (0.,)
                for threshold in thresholds:
                    current = {**settings,'threshold':threshold}
                    results = batch(dev,dev_trees,dev_judges,model,current,method)
                    consider(method,key,current,results)

    for method in ('A','C'):
        by_fit = {}
        for item in candidates:
            if item['family']==method:
                utility = item['dev']['accuracy']-.05*item['dev']['mean_compute']/1000
                by_fit[item['fit']] = max(utility,by_fit.get(item['fit'],-float('inf')))
        finalists = sorted(by_fit,key=lambda key:(by_fit[key],key),reverse=True)[:4]
        thresholds = (-.04,-.02,-.01,-.005,0.,.005,.01,.02) if method=='C' else (0.,)
        for key,penalty,weight,threshold in product(finalists,(.02,.025,.035,.05,.07,.1,.12,.15,.2,.25),(.6,1.,1.3,1.6),thresholds):
            settings = {'penalty':penalty,'science_weight':weight,'threshold':threshold}
            results = batch(dev,dev_trees,dev_judges,fits[key],settings,method)
            consider(method,key,settings,results)
    first_key = next(iter(fits))
    for judge_stop in ('action',.5,.75,.9,.95,.98,.99,1.):
        settings = {'penalty':.05, 'threshold':0., 'judge_stop':judge_stop}
        results = batch(dev,dev_trees,dev_judges,fits[first_key],settings,'B')
        consider('B',first_key,settings,results)
    chosen = {family:selected[family][1] for family in ('A','B','C')}
    models = {key:fits[key].as_dict() for key in sorted({item['fit'] for item in chosen.values()})}
    c = chosen['C']
    c_model = fits[c['fit']]
    c_results = batch(dev,dev_trees,dev_judges,c_model,c['settings'],'C')
    checkpoints = sum(min(r['solver_calls'],2) for r in c_results)
    rate = sum(r['judge_calls'] for r in c_results)/checkpoints
    random_settings = {**c['settings'],'rate':rate}
    random_results = batch(dev,dev_trees,dev_judges,c_model,random_settings,'random')
    payload = {'evidence':evidence(source),'fit_questions':len(rows),'dev_questions':len(dev),
               'selection_objective':'Development accuracy minus 0.05 times mean solver-equivalent tokens / 1000.',
               'archive_initial_pairs':len(paired_archive()),'selected':chosen,'models':models,
               'baselines':fixed_options,'primary_baseline':best_fixed,
               'random_judge_control':{'fit':c['fit'],'settings':random_settings,'dev':statistics(random_results)}}
    with new_run(output) as stage:
        save(stage/'frozen.json',payload)
        save(stage/'development-search.json',candidates)
    for family, item in chosen.items():
        print(family, json.dumps({k:v for k,v in item.items() if k!='family'}))
    print('Baselines',json.dumps(fixed_options))
    print('Random judge comparison', json.dumps(payload['random_judge_control']))


def score(source, output):
    freeze = json.loads((output/'frozen.json').read_bytes())
    if freeze['evidence'] != evidence(source):
        raise ValueError('Code or development evidence changed after freeze')
    rows, trees, judges = records(source,'test')
    all_questions = [*read(source/'splits/train-questions.jsonl'),*read(source/'splits/dev-questions.jsonl')]
    previous = paired_archive()
    forbidden_ids = {row['id'] for row in all_questions+previous}
    if forbidden_ids & {row['id'] for row in rows}:
        raise ValueError('Training questions entered the holdout')
    results = {}
    for name, config in freeze['baselines'].items():
        results[name] = [fixed(row,trees[row['id']],config['action'],config['depth']) for row in rows]
    for name, config in freeze['selected'].items():
        model = BeliefPlanner.restore(freeze['models'][config['fit']])
        results[name] = batch(rows,trees,judges,model,config['settings'],name)
    control = freeze['random_judge_control']
    control_model = BeliefPlanner.restore(freeze['models'][control['fit']])
    results['Random judge'] = batch(rows,trees,judges,control_model,control['settings'],'random')
    for name,method in (('C without judge','never'),('C always judge','always')):
        results[name] = batch(rows,trees,judges,control_model,freeze['selected']['C']['settings'],method)
    summaries = {name:describe(items) for name,items in results.items()}
    measurements = {name:{'correct':np.array([r['correct'] for r in items]),'work':np.array([r['compute'] for r in items])} for name,items in results.items()}
    rng = np.random.default_rng(2026111906)
    comparisons = {name:{reference:paired(measurements[name],measurements[reference],rng) for reference in ('Stop','Repair','Retry','A','B','Random judge','C without judge','C always judge') if reference!=name}
                   for name in ('A','B','C')}
    with new_run(output/'test') as stage:
        save(stage/'summary.json',{'methods':summaries,'paired':comparisons,'primary_baseline':freeze['primary_baseline']})
        with (stage/'per_question.jsonl').open('w') as stream:
            for name,items in results.items():
                for row in items:
                    stream.write(json.dumps({'policy':name,**row})+'\n')
    for name, info in summaries.items():
        s=info['all']
        print(f'{name:<14} {s["correct"]}/{s["n"]} {s["accuracy"]:.2%} {s["mean_compute"]:.2f} equivalent tokens')
    print(json.dumps(comparisons['C'],indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('tune','score'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    {'tune':tune,'score':score}[args.phase](args.data,args.output)


if __name__=='__main__':
    main()
