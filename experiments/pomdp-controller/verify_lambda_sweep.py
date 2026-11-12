"""Offline train/dev-only audit of the exploratory lambda diagnostic."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from stop_repair_retry import pomdp_experiment as p


def verify(directory):
    protocol=p.io.read_json(directory/'protocol.json');status=p.io.read_json(directory/'status.json')
    assert status['status']=='complete' and status['new_paid_calls']==0
    assert protocol['splits']==['train','dev'] and protocol['test_evaluation'] is False
    assert protocol['script_sha256']==p.io.sha(Path(__file__).with_name('lambda_sweep.py'))
    assert protocol['core_sha256']==p.io.sha(p.__file__)
    assert protocol['dataset_manifest_sha256']==p.io.sha(p.io.DEFAULT_DATA/'FILE_MANIFEST.json')
    model=p.io.read_json(ROOT/'results/pomdp-two-state/model.json')
    assert protocol['model_sha256']==p.io.sha(ROOT/'results/pomdp-two-state/model.json')
    assert model==p.fit_model(p.io.load_dataset(p.io.DEFAULT_DATA,'train'))
    grid=sorted(set([round(i/500,6) for i in range(601)]+[.001]))
    assert protocol['lambdas']==grid and status['grid_points']==602
    original=p.plan
    @lru_cache(None)
    def cached(belief,remaining,penalty):return original(model,list(belief),remaining,penalty)
    def fast_plan(m,b,remaining,penalty):
        assert m is model
        return cached(tuple(b),remaining,penalty)
    p.plan=fast_plan
    try:
        for split in ('train','dev'):
            rows=p.io.load_dataset(p.io.DEFAULT_DATA,split)
            sweep=p.io.read_json(directory/f'{split}-sweep.json');policies=p.io.read_json(directory/f'{split}-policies.json')
            assert [r['lambda'] for r in sweep]==grid
            seen=set()
            for result in sweep:
                lam=result['lambda'];records=p.replay(model,rows,lam)
                paths=[r['path'] for r in records];sig=hashlib.sha256(json.dumps(paths).encode()).hexdigest()
                accuracy=statistics.mean(r['correct'] for r in records);tokens=statistics.mean(r['solver_tokens'] for r in records)
                assert result=={'lambda':lam,'n':len(rows),'accuracy':accuracy,'mean_tokens':tokens,
                    'utility_at_this_lambda':accuracy-lam*tokens/1000,'utility_at_reference_lambda_0.01':accuracy-.01*tokens/1000,
                    'policy_signature':sig,'action_sequences':dict(Counter(' -> '.join(s['action'] for s in r['steps']) for r in records))}
                if sig not in seen:
                    assert policies[sig]=={'first_grid_lambda':lam,'question_ids':[r['id'] for r in records],'paths':paths,
                        'correct':[r['correct'] for r in records],'solver_tokens':[r['solver_tokens'] for r in records]}
                    seen.add(sig)
            assert seen==set(policies) and len(seen)==status[f'unique_{split}_policies']
            if split=='dev':
                winner=min(sweep,key=lambda r:(-r['accuracy'],r['mean_tokens'],r['lambda']))
                assert winner==p.io.read_json(directory/'dev-candidate.json')==status['candidate']
                replay=p.replay(model,rows,winner['lambda']);saved=p.io.read_json(directory/'dev-candidate-paths.json')
                def untimed(rr):return [{k:v for k,v in r.items() if k!='controller_cpu_seconds'} for r in rr]
                assert untimed(saved)==untimed(replay)
                baseline=p.io.read_json(directory/'dev-baselines.json')
                for item in baseline:
                    records=p.fixed_records(rows,p.io.FIXED[item['policy']])
                    assert item['accuracy']==statistics.mean(r['correct'] for r in records)
                    assert item['mean_tokens']==statistics.mean(r['tokens'] for r in records)
    finally:p.plan=original
    return {'status':'PASS','checked_splits':['train','dev'],'grid_points':602,'candidate':winner,
        'checks':['No test-data or test-outcome reads in reviewed sweep code','Original frozen model/code hashes match; independently confirmed train fit','All 602 penalties on both splits replayed with exact paths, scores, costs and summaries','All distinct-policy records and candidate paths verified','Candidate exactly follows highest dev accuracy, lowest tokens, lowest lambda rule','Dev fixed baselines recomputed'],
        'limitations':['Exploratory selection over 602 correlated grid points on only 60 dev questions can overfit.','No held-out test re-evaluation; cannot claim a test accuracy improvement or compare dev accuracy directly to 54.75% test accuracy.','Original frozen primary results remain unchanged.'],
        'reproduce':'python3 -B experiments/pomdp-controller/verify_lambda_sweep.py --output results/pomdp-lambda-diagnostic/validation.json'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=ROOT/'results/pomdp-lambda-diagnostic');parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=verify(args.run)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
