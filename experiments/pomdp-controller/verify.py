"""Offline verification of the frozen two-state study; no inference or cloud calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from stop_repair_retry import pomdp_experiment as p


def interval(values):
    rng=random.Random(20261110)
    means=sorted(sum(values[rng.randrange(len(values))] for _ in range(len(values)))/len(values) for _ in range(2000))
    return [means[50],means[1949]]


def verify(run):
    model=p.check_run(run);freeze=p.check_freeze(run)
    rows={s:p.io.load_dataset(p.io.DEFAULT_DATA,s) for s in ('train','dev','test')}
    ids={s:{r['id'] for r in rr} for s,rr in rows.items()}
    assert [len(ids[s]) for s in rows]==[100,60,400]
    assert not(ids['train']&ids['dev'] or ids['train']&ids['test'] or ids['dev']&ids['test'])
    assert p.io.read_json(run/'manifest.json')['training_ids']==[r['id'] for r in rows['train']]
    # Count train labels and features directly, independently of fit_model.
    roots=Counter(int(r['tree']['']['correct']) for r in rows['train'])
    assert model['training_counts']['roots']==[roots[i] for i in range(2)]
    assert model['prior']==[(roots[i]+1)/102 for i in range(2)]
    def obs(n):return (('low','medium','high','missing').index(n['confidence'] if n['confidence'] in ('low','medium','high') else 'missing')*2+int(n['answer_parse_valid']))
    for state in range(2):
        nodes=[r['tree'][''] for r in rows['train'] if int(r['tree']['']['correct'])==state]
        counts=Counter(obs(n) for n in nodes)
        assert model['initial_observation'][state]==[(counts[o]+1)/(len(nodes)+8) for o in range(8)]
    for action in p.ACTIONS:
        edges=[(r['tree'][path],r['tree'][path+('/' if path else '')+action]) for r in rows['train'] for path in ('','repair','retry')]
        assert len(edges)==300
        for state in range(2):
            targets=[v for u,v in edges if int(u['correct'])==state]
            counts=Counter(int(v['correct']) for v in targets)
            assert model['transition'][action][state]==[(counts[j]+1)/(len(targets)+2) for j in range(2)]
            costs=[v['input_tokens']+v['output_tokens'] for v in targets] or [v['input_tokens']+v['output_tokens'] for u,v in edges]
            assert model['cost_tokens'][action][state]==statistics.mean(costs)
            targets=[v for u,v in edges if int(v['correct'])==state];counts=Counter(obs(v) for v in targets)
            assert model['observation'][action][state]==[(counts[o]+1)/(len(targets)+8) for o in range(8)]
    output={'status':'PASS','scope':'Offline verification; no inference or cloud calls','checks':['Published split IDs disjoint and complete','Train-only independently recounted prior, transitions, joint observations and conditional costs','Frozen module/helper/design/data/A hashes and model reconstruction','Complete dev before freeze before test','All five penalties replay identically','Actual visited tokens and terminal correctness independently recomputed','All summary means and primary bootstrap endpoints independently reproduced'],'splits':{}}
    for split in ('dev','test'):
        status=p.validate_phase(run,split);summaries={}
        if split=='dev':assert status['finished_utc']<=freeze['frozen_utc']
        else:assert freeze['frozen_utc']<=status['started_utc']<=status['finished_utc']
        for penalty in p.CONFIG['lambda_grid']:
            records=p.io.read_json(run/split/f'paths-{penalty}.json');summary=p.io.read_json(run/split/f'summary-{penalty}.json')
            methods={'pomdp_two_state':[]}
            for row,record in zip(rows[split],records):
                path=record['path'];parts=path.split('/') if path else []
                visited=['']+['/'.join(parts[:i]) for i in range(1,len(parts)+1)]
                tokens=sum(row['tree'][q]['input_tokens']+row['tree'][q]['output_tokens'] for q in visited)
                assert tokens==record['solver_tokens'] and record['controller_tokens']==0
                assert record['correct']==int(row['tree'][path]['correct'])
                methods['pomdp_two_state'].append({'id':row['id'],'domain':row['domain'],'correct':record['correct'],'tokens':tokens})
            for name,paths in p.io.FIXED.items():
                methods[name]=[{'id':r['id'],'domain':r['domain'],'correct':int(r['tree'][paths[-1]]['correct']),'tokens':sum(r['tree'][q]['input_tokens']+r['tree'][q]['output_tokens'] for q in paths)} for r in rows[split]]
            if penalty==.01:methods['llm_controller_A']=p.io.read_json(p.A_RESULTS/split/'controller-results.json')
            for item in summary['summaries']:
                rr=[r for r in methods[item['policy']] if item['domain']=='all' or r['domain']==item['domain']]
                scores=[r['correct'] for r in rr]
                assert item['n']==len(rr) and item['accuracy']==statistics.mean(scores)
                assert item['mean_tokens']==statistics.mean(r['tokens'] for r in rr)
                assert item['mean_utility']==statistics.mean(r['correct']-penalty*r['tokens']/1000 for r in rr)
                if penalty==.01:assert item['accuracy_ci95']==interval(scores)
            for item in summary['paired_utility']:
                diff=[a['correct']-b['correct']-penalty*(a['tokens']-b['tokens'])/1000 for a,b in zip(methods['pomdp_two_state'],methods[item['comparator']])]
                assert item['mean_utility_difference']==statistics.mean(diff) and item['ci95']==interval(diff)
            summaries[str(penalty)]={'overall':next(s for s in summary['summaries'] if s['policy']=='pomdp_two_state' and s['domain']=='all'),'action_sequences':dict(Counter(' -> '.join(s['action'] for s in r['steps']) for r in records))}
        output['splits'][split]=summaries
    output['limitations']=['Bootstrap endpoints reproduce declared arithmetic; this does not establish interval coverage.','Correctness-only dynamics, pooled depth/domain and within-question correlated branches remain modeling approximations.','Test outcomes were previously inspected; not an untouched confirmation set.','Observed primary policy selects repair twice for all 60 dev and 400 test questions, exactly matching that fixed baseline; no adaptive benefit is demonstrated.','Four-state planner interface is unit-tested; no four-state estimator/model is fitted.']
    output['reproduce']='python3 -B experiments/pomdp-controller/verify.py --run results/pomdp-two-state --output results/pomdp-two-state/validation.json'
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=ROOT/'results/pomdp-two-state');parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=verify(args.run)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
