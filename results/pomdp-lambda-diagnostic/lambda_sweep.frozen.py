"""Exploratory fine lambda sweep: TRAIN/DEV only, frozen two-state model, no inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from stop_repair_retry import pomdp_experiment as p

LAMBDAS=sorted(set([round(i/500,6) for i in range(601)]+[.001,.01,.1,1]))

def evaluate(model,rows,penalty):
    records=p.replay(model,rows,penalty)
    accuracy=statistics.mean(r['correct'] for r in records)
    tokens=statistics.mean(r['solver_tokens'] for r in records)
    paths=[r['path'] for r in records]
    signature=hashlib.sha256(json.dumps(paths).encode()).hexdigest()
    return {'lambda':penalty,'n':len(rows),'accuracy':accuracy,'mean_tokens':tokens,
      'utility_at_this_lambda':accuracy-penalty*tokens/1000,'utility_at_reference_lambda_0.01':accuracy-.01*tokens/1000,
      'policy_signature':signature,'action_sequences':dict(Counter(' -> '.join(s['action'] for s in r['steps']) for r in records))},records

def run(output):
    if output.exists():raise ValueError('Refuse overwriting diagnostic run')
    original=ROOT/'results/pomdp-two-state';model=p.io.read_json(original/'model.json')
    rows={s:p.io.load_dataset(p.io.DEFAULT_DATA,s) for s in ('train','dev')}
    assert model==p.fit_model(rows['train'])
    output.mkdir(parents=True)
    p.io.save(output/'protocol.json',{'created_utc':p.io.now(),'purpose':'Exploratory diagnostic, not replacement of frozen primary result','splits':['train','dev'],'test_evaluation':False,
      'lambdas':LAMBDAS,'range':[0,1.2],'regular_step':.002,'extra_points':[.001],
      'model_sha256':p.io.sha(original/'model.json'),'core_sha256':p.io.sha(p.__file__),'script_sha256':p.io.sha(__file__),
      'dataset_manifest_sha256':p.io.sha(p.io.DEFAULT_DATA/'FILE_MANIFEST.json'),
      'candidate_rule':'Highlight highest development accuracy, then lowest mean tokens, then lowest lambda. This is exploratory dev selection, not a new primary cost preference or a test claim.',
      'scope':'No refit, no model/observation changes, no test re-evaluation, no paid calls. Previously viewed test results motivate investigation but are not used to select lambda.'})
    summaries={s:[] for s in rows};distinct={s:{} for s in rows};candidate=None;candidate_records=None
    for i,lam in enumerate(LAMBDAS):
        for split in rows:
            result,records=evaluate(model,rows[split],lam);summaries[split].append(result)
            signature=result['policy_signature']
            if signature not in distinct[split]:
                distinct[split][signature]={'first_grid_lambda':lam,'question_ids':[r['id'] for r in records],
                    'paths':[r['path'] for r in records],'correct':[r['correct'] for r in records],'solver_tokens':[r['solver_tokens'] for r in records]}
            key=(-result['accuracy'],result['mean_tokens'],lam)
            if split=='dev' and (candidate is None or key<(-candidate['accuracy'],candidate['mean_tokens'],candidate['lambda'])):
                candidate=result;candidate_records=records
        if i%100==0:print(f'Grid points evaluated: {i+1}/{len(LAMBDAS)}',flush=True)
    for split in rows:
        p.io.save(output/f'{split}-sweep.json',summaries[split]);p.io.save(output/f'{split}-policies.json',distinct[split])
    p.io.save(output/'dev-candidate.json',candidate);p.io.save(output/'dev-candidate-paths.json',candidate_records)
    diagnostics=[]
    for obs in range(len(p.OBSERVATIONS)):
        selected=[r for r in rows['train'] if p.observation(r['tree'][''])==obs]
        diagnostics.append({'observation':p.OBSERVATIONS[obs],'train_roots':len(selected),'empirical_root_accuracy':statistics.mean(r['tree']['']['correct'] for r in selected) if selected else None,
          'initial_belief_correct':p.initial_belief(model,obs)[1]})
    p.io.save(output/'training-observations.json',diagnostics)
    boundaries=[]
    for lam in [0,.01,.1,candidate['lambda'],.2,.3,.4,.6,1,1.2]:
        for remaining in (1,2):
            boundaries.append({'lambda':lam,'remaining':remaining,'belief_correct':[i/100 for i in range(101)],
                'actions':[p.plan(model,[1-i/100,i/100],remaining,lam)['action'] for i in range(101)]})
    p.io.save(output/'belief-action-grid.json',boundaries)
    baselines=[]
    for name,paths in p.io.FIXED.items():
        rr=p.fixed_records(rows['dev'],paths);baselines.append({'policy':name,'accuracy':statistics.mean(r['correct'] for r in rr),'mean_tokens':statistics.mean(r['tokens'] for r in rr)})
    p.io.save(output/'dev-baselines.json',baselines)
    p.io.save(output/'status.json',{'status':'complete','grid_points':len(LAMBDAS),'unique_train_policies':len(distinct['train']),
      'unique_dev_policies':len(distinct['dev']),'candidate':candidate,'finished_utc':p.io.now(),'new_paid_calls':0})
    print(json.dumps({'candidate':candidate,'unique_dev_policies':len(distinct['dev'])}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.output)
