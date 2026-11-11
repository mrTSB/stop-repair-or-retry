"""Offline analytic checks of dev-only lambda-region construction."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import policy_regions as r
p=r.p


def verify(directory):
    protocol=p.io.read_json(directory/'protocol.json');status=p.io.read_json(directory/'status.json')
    model=p.io.read_json(r.ROOT/'results/pomdp-two-state/model.json')
    assert protocol['script_sha256']==p.io.sha(r.__file__)
    assert protocol['model_sha256']==p.io.sha(r.ROOT/'results/pomdp-two-state/model.json')
    assert protocol['splits']==['dev'] and protocol['test_evaluation'] is False
    boundaries,samples=r.regions(model)
    assert protocol['boundaries']==boundaries and protocol['sample_lambdas']==samples
    assert status['boundary_count']==len(boundaries) and status['probes']==len(samples)
    rng=random.Random(238)
    probes=samples+[rng.uniform(a,b) for a,b in zip(boundaries,boundaries[1:])]
    roots=[p.initial_belief(model,z) for z in range(len(model['observations']))]
    comparisons=0
    for belief in roots:
        for lam in probes:
            decision=p.plan(model,belief,2,lam)
            for action,(intercept,cost) in r.root_lines(model,belief,lam).items():
                assert math.isclose(intercept-lam*cost,decision['action_values'][action],rel_tol=1e-12,abs_tol=1e-12)
                comparisons+=1
        for action in p.ACTIONS:
            for obs in range(len(model['observations'])):
                child=p.update(model,belief,action,obs);lines=r.one_call_lines(model,child)
                for lam in probes:
                    decision=p.plan(model,child,1,lam)
                    for a,(intercept,cost) in lines.items():
                        assert math.isclose(intercept-lam*cost,decision['action_values'][a],rel_tol=1e-12,abs_tol=1e-12)
                        comparisons+=1
    rows=p.io.load_dataset(p.io.DEFAULT_DATA,'dev');summaries=p.io.read_json(directory/'dev-regions.json');policies=p.io.read_json(directory/'dev-policies.json')
    assert [s['lambda'] for s in summaries]==samples
    seen=set()
    for saved in summaries:
        recomputed,records=r.evaluate(model,rows,saved['lambda']);assert saved==recomputed
        sig=saved['policy_signature'];paths=[x['path'] for x in records]
        assert sig==hashlib.sha256(json.dumps(paths).encode()).hexdigest()
        if sig not in seen:
            assert policies[sig]=={'representative_lambda':saved['lambda'],'paths':paths,'question_ids':[x['id'] for x in records],
                'correct':[x['correct'] for x in records],'solver_tokens':[x['solver_tokens'] for x in records],
                'accuracy':saved['accuracy'],'mean_tokens':saved['mean_tokens']}
            seen.add(sig)
    assert set(policies)==seen and len(seen)==status['unique_dev_policies']
    assert status['best_observed_dev']==min(summaries,key=lambda s:(-s['accuracy'],s['mean_tokens'],s['lambda']))
    grid=p.io.read_json(r.ROOT/'results/pomdp-lambda-diagnostic/dev-policies.json')
    assert set(grid)==set(policies)
    return {'status':'PASS','boundaries':len(boundaries),'probes':len(samples),'action_value_checks':comparisons,
        'unique_dev_policies':len(seen),'candidate':status['best_observed_dev'],
        'checks':['Reviewed piecewise linear intercept/cost construction: posterior probabilities independent of lambda; child optimum is envelope of three affine values; root action values affine within child-policy intervals.',
        'Every root/action line matches frozen horizon-two planner at boundaries, midpoints, +/-1e-9 and deterministic interior probes.',
        'Every reachable child/action line matches frozen horizon-one planner at same probes.',
        'All dev paths, costs, scores and candidate selection recomputed; eight signatures equal fine-grid policies.',
        'Reviewed code loads only frozen model and dev; no test evaluation or paid calls.'],
        'limitations':['Floating-point intersections and 1e-12 tie tolerance mean these probes are not a formal exact-arithmetic certificate of every microscopic tolerance region.','Observed best dev accuracy does not exceed original dev accuracy; no claim of improving held-out test accuracy.','Exploratory post-result diagnostic, not a replacement primary experiment.'],
        'reproduce':'python3 -B experiments/pomdp-controller/verify_policy_regions.py --output results/pomdp-lambda-regions/validation.json'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=r.ROOT/'results/pomdp-lambda-regions');parser.add_argument('--output',type=Path);args=parser.parse_args();result=verify(args.run)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
