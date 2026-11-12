"""Offline audit of latent-state grouped CV, deterministic refits and replay."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import latent_states as l
p=l.p


def untimed(records):return [{k:v for k,v in r.items() if k!='controller_cpu_seconds'} for r in records]


def verify(directory):
    protocol=p.io.read_json(directory/'protocol.json');status=p.io.read_json(directory/'status.json')
    assert protocol['state_counts']==[2,4] and protocol['lambdas']==[.01,.248] and protocol['iterations']==20
    assert status['status']=='complete' and status['test_evaluated'] is False and status['new_paid_calls']==0
    for key,path in [('script_sha256',l.__file__),('context_sha256',l.c.__file__),('calibration_sha256',l.cv.__file__),('core_sha256',p.__file__),('dataset_manifest_sha256',p.io.DEFAULT_DATA/'FILE_MANIFEST.json')]:assert protocol[key]==p.io.sha(path)
    rows=p.io.load_dataset(p.io.DEFAULT_DATA,'train');assignment=p.io.read_json(directory/'folds.json');assert assignment==l.cv.folds(rows)
    assert len(rows)==len({r['id'] for r in rows})==100
    aggregate={(n,lam):[] for n in [2,4] for lam in [.01,.248]};last_steps=[];control_errors=[]
    for fold in range(5):
        training=[r for r in rows if assignment[r['id']]!=fold];held=[r for r in rows if assignment[r['id']]==fold]
        assert len(training)==80 and len(held)==20 and not ({r['id'] for r in training}&{r['id'] for r in held})
        assert sum(r['domain']=='math' for r in held)==10
        for n in [2,4]:
            saved=p.io.read_json(directory/f'fold-{fold}-states-{n}-model.json');refit=l.fit(training,n)
            assert saved==refit and saved['fit_ids']==[r['id'] for r in training]
            counts=saved['expected_training_counts'];assert abs(sum(counts['root'])-80)<1e-9
            for action in p.ACTIONS:assert abs(sum(map(sum,counts['transitions']['pooled'][action]))-240)<1e-9
            if n==4:last_steps.append(saved['training_log_likelihood'][-1]-saved['training_log_likelihood'][-2])
            if n==2:
                control=l.cv.fit(training,'context_calibrated');errors=[]
                for ctx in ['pooled_prior_model',*l.c.CONTEXTS]:
                    a=saved[ctx] if ctx=='pooled_prior_model' else saved['contexts'][ctx]
                    b=control[ctx] if ctx=='pooled_prior_model' else control['contexts'][ctx]
                    for s in range(2):
                        errors.append(abs(a['prior'][s]-b['prior'][1-s]))
                        for z in range(8):errors.append(abs(a['initial_observation'][s][z]-b['initial_observation'][1-s][z]))
                        for action in p.ACTIONS:
                            errors.append(abs(a['cost_tokens'][action][s]-b['cost_tokens'][action][1-s]))
                            for sp in range(2):errors.append(abs(a['transition'][action][s][sp]-b['transition'][action][1-s][1-sp]))
                            for z in range(8):errors.append(abs(a['observation'][action][s][z]-b['observation'][action][1-s][z]))
                control_errors.append(max(errors));assert max(errors)<1e-9
            for lam in [.01,.248]:
                paths=p.io.read_json(directory/f'fold-{fold}-states-{n}-{lam}-paths.json')
                assert untimed(paths)==untimed(l.c.replay(saved,held,lam))
                assert [r['id'] for r in paths]==[r['id'] for r in held]
                aggregate[(n,lam)]+=paths
    summaries=p.io.read_json(directory/'oof-summary.json');assert len(summaries)==4
    for summary in summaries:
        records=aggregate[(summary['states'],summary['lambda'])];assert len({r['id'] for r in records})==100
        assert summary=={'states':summary['states'],'lambda':summary['lambda'],**l.cv.summary(records)}
    selection=p.io.read_json(directory/'selection.json');winner=min(summaries,key=lambda s:(-s['accuracy'],s['mean_tokens'],s['states'],s['lambda']))
    assert selection['candidate']==winner and selection['dev_not_loaded_yet'] is True
    assert protocol['created_utc']<=selection['selected_utc']<=status['finished_utc']
    selected=p.io.read_json(directory/'selected-model.json');assert selected==l.fit(rows,winner['states'])
    dev=p.io.load_dataset(p.io.DEFAULT_DATA,'dev');paths=p.io.read_json(directory/'selected-dev-paths.json')
    assert len(paths)==60 and untimed(paths)==untimed(l.c.replay(selected,dev,winner['lambda']))
    assert not ({r['id'] for r in rows}&{r['id'] for r in dev})
    dev_summary=p.io.read_json(directory/'selected-dev-summary.json');assert dev_summary==l.cv.summary(paths)
    return {'status':'PASS','folds':5,'fold_models_refitted':10,'heldout_path_sets_replayed':20,'selected':winner,'selected_dev':dev_summary,
        'maximum_two_state_control_parameter_error':max(control_errors),'four_state_last_likelihood_increases':last_steps,
        'checks':['Exact enumeration unit test verifies three-node four-state likelihood, node marginals and edge marginals','Held-out correctness-label changes leave decisions and beliefs invariant','Two-state fits agree with calibrated context control after reversing state order','All deterministic fold fits and selected100question fit exactly reproduced','Each training question held out once/configuration; fit80/held20 domain-balanced and branch-grouped','All four OOF summaries and declared selection rule reproduced; only selected candidate evaluated on dev','Source review confirms no test reads, unvisited-policy inputs or inference calls'],
        'limitations':['One deterministic initialization and20iterations; four-state last-iteration likelihood increments show convergence is not established.','Empirical shrinkage and calibration are regularized updates, not guaranteed monotone maximum-likelihood EM.','Latent incorrect types have no established repairability semantics.','Prior observed outcomes informed diagnostic direction; not untouched confirmation; no four-state accuracy gain demonstrated.'],
        'reproduce':'python3 -B experiments/pomdp-controller/verify_latent_states.py --output results/pomdp-latent-state-cv/validation.json'}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,default=l.ROOT/'results/pomdp-latent-state-cv');ap.add_argument('--output',type=Path);args=ap.parse_args();result=verify(args.run)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
