"""Independent offline evidence audit for the separate lambda=.248 follow-up."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import frozen_lambda as f
p=f.p


def interval(values):
    rng=random.Random(20261110);n=len(values)
    means=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(2000))
    return [means[50],means[1949]]


def verify(run):
    f.verify(run)
    freeze=p.io.read_json(run/'freeze.json');summary=p.io.read_json(run/'summary.json');records=p.io.read_json(run/'test-paths.json')
    assert p.io.sha(run/'frozen_lambda.frozen.py')==p.io.sha(f.__file__)
    assert freeze['planning_lambda']==.248 and freeze['prior_test_exposure'] is True
    assert freeze['comparison_utilities']==[.248,.01] and freeze['llm_policy_lambda']==.01
    # Original study integrity (no reevaluation or modifications).
    p.check_run(f.ORIGINAL);p.check_freeze(f.ORIGINAL)
    rows=p.io.load_dataset(p.io.DEFAULT_DATA,'test')
    assert len(rows)==len(records)==400
    assert [r['id'] for r in rows]==[r['id'] for r in records]
    methods={'pomdp_lambda_0.248':[]}
    for row,record in zip(rows,records):
        parts=record['path'].split('/') if record['path'] else []
        visited=['']+['/'.join(parts[:i]) for i in range(1,len(parts)+1)]
        assert len(visited)<=3
        tokens=sum(row['tree'][path]['input_tokens']+row['tree'][path]['output_tokens'] for path in visited)
        correct=int(row['tree'][record['path']]['correct'])
        assert record['solver_tokens']==tokens and record['controller_tokens']==0 and record['correct']==correct
        methods['pomdp_lambda_0.248'].append({'id':row['id'],'domain':row['domain'],'correct':correct,'tokens':tokens})
    for name,paths in p.io.FIXED.items():
        methods[name]=[{'id':r['id'],'domain':r['domain'],'correct':int(r['tree'][paths[-1]]['correct']),
            'tokens':sum(r['tree'][path]['input_tokens']+r['tree'][path]['output_tokens'] for path in paths)} for r in rows]
    methods['llm_controller_A']=p.io.read_json(p.A_RESULTS/'test/controller-results.json')
    assert [r['id'] for r in methods['llm_controller_A']]==[r['id'] for r in rows]
    assert len(summary['summaries'])==21 and len(summary['paired_comparisons'])==6
    assert {(s['method'],s['domain']) for s in summary['summaries']}=={(m,d) for m in methods for d in ('all','math','science')}
    for item in summary['summaries']:
        selected=[r for r in methods[item['method']] if item['domain']=='all' or r['domain']==item['domain']]
        scores=[r['correct'] for r in selected]
        assert item['n']==len(selected) and item['correct']==sum(scores)
        assert item['accuracy']==statistics.mean(scores) and item['accuracy_ci95']==interval(scores)
        assert item['mean_tokens']==statistics.mean(r['tokens'] for r in selected)
        for lam,key in ((.248,'mean_utility_lambda_0.248'),(.01,'mean_utility_reference_0.01')):
            assert item[key]==statistics.mean(r['correct']-lam*r['tokens']/1000 for r in selected)
    assert {s['comparator'] for s in summary['paired_comparisons']}==set(methods)-{'pomdp_lambda_0.248'}
    for item in summary['paired_comparisons']:
        pairs=list(zip(methods['pomdp_lambda_0.248'],methods[item['comparator']]))
        differences=[a['correct']-b['correct'] for a,b in pairs]
        assert item['accuracy_difference']==statistics.mean(differences) and item['accuracy_difference_ci95']==interval(differences)
        for lam in (.248,.01):
            differences=[a['correct']-b['correct']-lam*(a['tokens']-b['tokens'])/1000 for a,b in pairs]
            assert item[f'utility_difference_lambda_{lam}']=={'mean':statistics.mean(differences),'ci95':interval(differences)}
    assert summary['action_sequences']==dict(Counter(' -> '.join(s['action'] for s in r['steps']) for r in records))
    return {'status':'PASS','questions':400,'planning_lambda':.248,'overall':next(s for s in summary['summaries'] if s['method']=='pomdp_lambda_0.248' and s['domain']=='all'),'action_sequences':summary['action_sequences'],
        'checks':['Source review confirms durable freeze and dev verification before loading test','Frozen source/model/dev selection hashes and chronological ordering verified','Original model/design/core and original dev freeze integrity verified','All400 question IDs and replayed paths/beliefs verified','Terminal correctness and visited solver-token accounting independently recomputed','All21 method/domain summaries and six paired comparisons at both utility penalties independently recomputed','Accuracy and paired bootstrap endpoints reproduced with declared2000 resamples and seed'],
        'limitations':['Exploratory lambda selected on60dev after earlier test exposure; no fresh confirmatory set.','Frozen LLM decisions retain original .01 prompt; .248 utility is reweighting, not retuning.','Bootstrap arithmetic reproduced; selection uncertainty, model-fit uncertainty and multiple comparisons are not accounted for.'],
        'reproduce':'python3 -B experiments/pomdp-controller/verify_frozen_lambda.py --output results/pomdp-lambda-0248/validation.json'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=f.ROOT/'results/pomdp-lambda-0248');parser.add_argument('--output',type=Path);args=parser.parse_args();result=verify(args.run)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
