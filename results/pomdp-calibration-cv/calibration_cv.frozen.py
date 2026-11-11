"""Question-grouped training cross-validation for unseen-observation calibration.

No test loading, no paid inference, no new observation feature. Dev is evaluated
only for the single variant/penalty selected from out-of-fold training results.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from stop_repair_retry import pomdp_experiment as p
import context_diagnostic as c

VARIANTS=['pooled','context','context_calibrated']
LAMBDAS=[.01,.248]
STRENGTH=10

def counts(rows,parents):
    t={a:[[0,0],[0,0]] for a in p.ACTIONS};z={a:[[0]*8 for _ in range(2)] for a in p.ACTIONS};cost={a:[[],[]] for a in p.ACTIONS}
    for r in rows:
        for parent in parents:
            s=int(r['tree'][parent]['correct'])
            for a in p.ACTIONS:
                child=r['tree'][parent+('/' if parent else '')+a];sp=int(child['correct'])
                t[a][s][sp]+=1;z[a][sp][p.observation(child)]+=1;cost[a][s].append(child['input_tokens']+child['output_tokens'])
    return t,z,cost

def unknown_equal(rows,observed_counts):
    """Unseen symbols have equal likelihood in both states: no belief evidence."""
    unseen=[j for j in range(len(rows[0])) if sum(r[j] for r in observed_counts)==0]
    if not unseen:return [list(r) for r in rows]
    total=sum(sum(r) for r in observed_counts);mass=1/(total+len(rows[0]));seen=[j for j in range(len(rows[0])) if j not in unseen]
    if not seen:return [[1/len(row)]*len(row) for row in rows]
    result=[]
    for row in rows:
        denominator=sum(row[j] for j in seen);remaining=1-mass*len(unseen)
        result.append([mass if j in unseen else remaining*row[j]/denominator for j in range(len(row))])
    return result

def fit(rows,variant):
    if variant not in VARIANTS or not rows or any(r['split']!='train' for r in rows) or len({r['id'] for r in rows})!=len(rows):raise ValueError('Fit only unique training questions')
    t,z,cost=counts(rows,('','repair','retry'));roots=[0,0];rootobs=[[0]*8 for _ in range(2)]
    for r in rows:
        node=r['tree'][''];s=int(node['correct']);roots[s]+=1;rootobs[s][p.observation(node)]+=1
    pooled={'states':['incorrect','correct'],'observations':p.OBSERVATIONS,'terminal_reward':[0,1],
      'prior':p.smooth(roots,1),'initial_observation':[p.smooth(row,1) for row in rootobs],
      'transition':{a:[p.smooth(row,1) for row in t[a]] for a in p.ACTIONS},
      'observation':{a:[p.smooth(row,1) for row in z[a]] for a in p.ACTIONS},
      'cost_tokens':{a:[statistics.mean(v) if v else statistics.mean(cost[a][0]+cost[a][1]) for v in cost[a]] for a in p.ACTIONS}}
    if variant=='context_calibrated':
        pooled['initial_observation']=unknown_equal(pooled['initial_observation'],rootobs)
        for a in p.ACTIONS:pooled['observation'][a]=unknown_equal(pooled['observation'][a],z[a])
    models={}
    for context,parent in c.CONTEXTS.items():
        if variant=='pooled':models[context]=pooled;continue
        ct,cz,ccost=counts(rows,(parent,))
        m={k:pooled[k] for k in ['states','observations','terminal_reward','prior','initial_observation']}
        m['transition']={a:[[(n+STRENGTH*prior)/(sum(row)+STRENGTH) for n,prior in zip(row,pooled['transition'][a][s])] for s,row in enumerate(ct[a])] for a in p.ACTIONS}
        m['observation']={a:[[(n+STRENGTH*prior)/(sum(row)+STRENGTH) for n,prior in zip(row,pooled['observation'][a][s])] for s,row in enumerate(cz[a])] for a in p.ACTIONS}
        m['cost_tokens']={a:[(sum(v)+STRENGTH*pooled['cost_tokens'][a][s])/(len(v)+STRENGTH) for s,v in enumerate(ccost[a])] for a in p.ACTIONS}
        if variant=='context_calibrated':
            for a in p.ACTIONS:m['observation'][a]=unknown_equal(m['observation'][a],z[a])
        p.validate_model(m);models[context]=m
    p.validate_model(pooled)
    return {'contexts':models,'pooled_prior_model':pooled,'variant':variant,'fit_ids':[r['id'] for r in rows],
      'training_root_counts':roots,'training_root_observation_counts':rootobs,'training_observation_counts':z}

def folds(rows):
    assignment={}
    for domain in ('math','science'):
        ordered=sorted([r for r in rows if r['domain']==domain],key=lambda r:hashlib.sha256(('cs238-calibration-cv-v1:'+r['id']).encode()).hexdigest())
        for i,r in enumerate(ordered):assignment[r['id']]=i%5
    return assignment

def summary(records):
    return {'n':len(records),'correct':sum(r['correct'] for r in records),'accuracy':statistics.mean(r['correct'] for r in records),
      'mean_tokens':statistics.mean(r['tokens'] for r in records),
      'action_sequences':dict(Counter(' -> '.join(s['action'] for s in r['steps']) for r in records))}

def run(output):
    if output.exists():raise ValueError('Refuse overwrite')
    output.mkdir(parents=True)
    p.io.save(output/'protocol.json',{'created_utc':p.io.now(),'variants':VARIANTS,'lambdas':LAMBDAS,'shrinkage_strength':STRENGTH,
      'folds':5,'fold_assignment':'Deterministic hashed question IDs, round-robin within domain. Every branch stays with its question.',
      'selection':'Maximize out-of-fold training accuracy, then minimize mean tokens, then prefer earlier variant in declared order, then smaller lambda. Do not select using dev or test.',
      'calibration':'Symbols absent from all fit questions at the relevant root/action distribution receive identical positive likelihood in every hidden state. Preserve the prior/predicted belief on an unseen symbol; renormalize supported symbols. Context emissions use globally unsupported action symbols for this safeguard.',
      'scope':'Training-only cross-validation, then evaluate only selected candidate on dev. No new observation feature, no test evaluation, no paid calls. Prior dev/test results have already informed diagnostic direction.',
      'script_sha256':p.io.sha(__file__),'context_script_sha256':p.io.sha(c.__file__),'core_sha256':p.io.sha(p.__file__),'dataset_manifest_sha256':p.io.sha(p.io.DEFAULT_DATA/'FILE_MANIFEST.json')})
    train=p.io.load_dataset(p.io.DEFAULT_DATA,'train');assignment=folds(train);p.io.save(output/'folds.json',assignment)
    all_records={(v,lam):[] for v in VARIANTS for lam in LAMBDAS}
    for fold in range(5):
        fitting=[r for r in train if assignment[r['id']]!=fold];held=[r for r in train if assignment[r['id']]==fold]
        assert len(fitting)==80 and len(held)==20 and not {r['id'] for r in fitting}&{r['id'] for r in held}
        for variant in VARIANTS:
            model=fit(fitting,variant);p.io.save(output/f'fold-{fold}-{variant}-model.json',model)
            for lam in LAMBDAS:
                rr=c.replay(model,held,lam);all_records[(variant,lam)]+=rr
                p.io.save(output/f'fold-{fold}-{variant}-{lam}-paths.json',rr)
    summaries=[]
    for (variant,lam),records in all_records.items():summaries.append({'variant':variant,'lambda':lam,**summary(records)})
    candidate=min(summaries,key=lambda r:(-r['accuracy'],r['mean_tokens'],VARIANTS.index(r['variant']),r['lambda']))
    p.io.save(output/'oof-summary.json',summaries);p.io.save(output/'selection.json',{'selected_utc':p.io.now(),'candidate':candidate,'note':'Selection completed before dev loading.'})
    # Only the selected variant is evaluated on dev.
    model=fit(train,candidate['variant']);p.io.save(output/'selected-model.json',model)
    dev=p.io.load_dataset(p.io.DEFAULT_DATA,'dev');records=c.replay(model,dev,candidate['lambda'])
    p.io.save(output/'selected-dev-paths.json',records);p.io.save(output/'selected-dev-summary.json',summary(records))
    baselines={name:{'train_accuracy':statistics.mean(r['correct'] for r in p.fixed_records(train,paths)),'train_tokens':statistics.mean(r['tokens'] for r in p.fixed_records(train,paths)),
      'dev_accuracy':statistics.mean(r['correct'] for r in p.fixed_records(dev,paths)),'dev_tokens':statistics.mean(r['tokens'] for r in p.fixed_records(dev,paths))} for name,paths in p.io.FIXED.items()}
    p.io.save(output/'fixed-baselines.json',baselines)
    p.io.save(output/'status.json',{'status':'complete','finished_utc':p.io.now(),'test_evaluated':False,'new_paid_calls':0})
    print({'oof':summaries,'selected':candidate,'dev':summary(records)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();run(args.output)
