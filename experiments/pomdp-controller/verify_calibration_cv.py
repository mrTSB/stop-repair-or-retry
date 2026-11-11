"""Offline verification of grouped train-only CV and its single dev follow-up."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import calibration_cv as v
p=v.p


def verify(directory):
    protocol=p.io.read_json(directory/'protocol.json');status=p.io.read_json(directory/'status.json')
    assert protocol['variants']==['pooled','context','context_calibrated'] and protocol['lambdas']==[.01,.248]
    assert protocol['folds']==5 and protocol['shrinkage_strength']==10
    assert status['status']=='complete' and status['test_evaluated'] is False and status['new_paid_calls']==0
    for key,path in [('script_sha256',v.__file__),('context_script_sha256',v.c.__file__),('core_sha256',p.__file__),('dataset_manifest_sha256',p.io.DEFAULT_DATA/'FILE_MANIFEST.json')]:assert protocol[key]==p.io.sha(path)
    train=p.io.load_dataset(p.io.DEFAULT_DATA,'train');assignment=p.io.read_json(directory/'folds.json');assert assignment==v.folds(train)
    assert len(train)==100 and len(set(r['id'] for r in train))==100
    assert len({' '.join(r['question'].split()).casefold() for r in train})==100
    aggregate={(variant,lam):[] for variant in v.VARIANTS for lam in v.LAMBDAS}
    for fold in range(5):
        fitting=[r for r in train if assignment[r['id']]!=fold];held=[r for r in train if assignment[r['id']]==fold]
        assert len(fitting)==80 and len(held)==20 and not ({r['id'] for r in fitting}&{r['id'] for r in held})
        assert Counter(r['domain'] for r in held)=={'math':10,'science':10}
        for variant in v.VARIANTS:
            model=p.io.read_json(directory/f'fold-{fold}-{variant}-model.json');assert model==v.fit(fitting,variant)
            assert model['fit_ids']==[r['id'] for r in fitting]
            root=Counter(int(r['tree']['']['correct']) for r in fitting)
            assert model['pooled_prior_model']['prior']==[(root[s]+1)/82 for s in range(2)]
            # Independently verify pooled transition estimates use exactly the
            # 240 within-fit-question edges/action, never held-out branches.
            for action in p.ACTIONS:
                edges=[(r['tree'][q],r['tree'][q+('/' if q else '')+action]) for r in fitting for q in ('','repair','retry')]
                assert len(edges)==240
                for s in range(2):
                    successors=[b for a,b in edges if int(a['correct'])==s];counts=Counter(int(b['correct']) for b in successors)
                    assert model['pooled_prior_model']['transition'][action][s]==[(counts[j]+1)/(len(successors)+2) for j in range(2)]
            for lam in v.LAMBDAS:
                saved=p.io.read_json(directory/f'fold-{fold}-{variant}-{lam}-paths.json');expected=v.c.replay(model,held,lam)
                def untimed(records):return [{k:x for k,x in r.items() if k!='controller_cpu_seconds'} for r in records]
                assert untimed(saved)==untimed(expected)
                assert [r['id'] for r in saved]==[r['id'] for r in held]
                aggregate[(variant,lam)]+=saved
    summaries=p.io.read_json(directory/'oof-summary.json')
    assert len(summaries)==6
    for row in summaries:
        records=aggregate[(row['variant'],row['lambda'])]
        assert len(records)==100 and len({r['id'] for r in records})==100
        assert row=={'variant':row['variant'],'lambda':row['lambda'],**v.summary(records)}
    winner=min(summaries,key=lambda r:(-r['accuracy'],r['mean_tokens'],v.VARIANTS.index(r['variant']),r['lambda']))
    selection=p.io.read_json(directory/'selection.json');assert selection['candidate']==winner
    assert protocol['created_utc']<=selection['selected_utc']<=status['finished_utc']
    selected=p.io.read_json(directory/'selected-model.json');assert selected==v.fit(train,winner['variant'])
    # At100 rows, unsafeguarded implementation matches original estimator.
    full=v.fit(train,'pooled')['pooled_prior_model'];old=p.io.read_json(v.ROOT/'results/pomdp-two-state/model.json')
    assert full=={k:x for k,x in old.items() if k!='training_counts'}
    dev=p.io.load_dataset(p.io.DEFAULT_DATA,'dev');saved=p.io.read_json(directory/'selected-dev-paths.json')
    assert untimed(saved)==untimed(v.c.replay(selected,dev,winner['lambda']))
    dev_summary=p.io.read_json(directory/'selected-dev-summary.json');assert dev_summary==v.summary(saved)
    for name,table in p.io.read_json(directory/'fixed-baselines.json').items():
        for split,rows in [('train',train),('dev',dev)]:
            records=p.fixed_records(rows,p.io.FIXED[name]);assert table[f'{split}_accuracy']==statistics.mean(r['correct'] for r in records)
            assert table[f'{split}_tokens']==statistics.mean(r['tokens'] for r in records)
    return {'status':'PASS','folds':5,'fit_questions_per_fold':80,'held_questions_per_fold':20,'selected':winner,'selected_dev':dev_summary,
        'checks':['Reviewed code loads train only before OOF selection; dev loaded only for selected candidate; no test reads/inference','All question IDs and normalized training stems unique; folds disjoint, balanced, and all branches stay with question','All15 fold models refitted on80questions, pooled counts independently verified,30 held-out path sets reproduced','All six OOF summaries cover each training question exactly once; selection rule reproduced','100-question pooled estimator matches original model; selected full fit and dev paths reproduced','Four analytic tests cover unsupported-likelihood prior/prediction invariance, normalization, grouped folds and invalid fitting rows'],
        'limitations':['Exploratory diagnostics were motivated by previously viewed dev/test outcomes; OOF estimates are not fresh confirmation.','The selected candidate shows no improvement over repair twice on dev.','Equal-likelihood safeguard covers globally unsupported root/action symbols; it is not a general confidence-calibration guarantee.'],
        'reproduce':'python3 -B experiments/pomdp-controller/verify_calibration_cv.py --output results/pomdp-calibration-cv/validation.json'}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,default=v.ROOT/'results/pomdp-calibration-cv');ap.add_argument('--output',type=Path);args=ap.parse_args();result=verify(args.run)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
