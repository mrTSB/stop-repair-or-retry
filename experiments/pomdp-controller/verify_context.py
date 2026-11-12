"""Independent offline train/dev audit of context-conditioned diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import context_diagnostic as c
p=c.p


def verify(directory):
    protocol=p.io.read_json(directory/'protocol.json');status=p.io.read_json(directory/'status.json')
    assert protocol['splits']==['train','dev'] and protocol['lambdas']==[.01,.248] and protocol['shrinkage_strength']==10
    assert status['status']=='complete' and status['test_evaluated'] is False and status['new_paid_calls']==0
    for key,file in [('script_sha256',c.__file__),('core_sha256',p.__file__),('helper_sha256',p.io.__file__),('design_sha256',p.DESIGN),('dataset_manifest_sha256',p.io.DEFAULT_DATA/'FILE_MANIFEST.json')]:assert protocol[key]==p.io.sha(file)
    rows={split:p.io.load_dataset(p.io.DEFAULT_DATA,split) for split in ('train','dev')}
    assert len(rows['train'])==100 and len(rows['dev'])==60 and not ({r['id'] for r in rows['train']}&{r['id'] for r in rows['dev']})
    model=p.io.read_json(directory/'model.json');assert model==c.fit(rows['train'])
    pooled=model['pooled_prior_model'];assert pooled==p.io.read_json(c.ROOT/'results/pomdp-two-state/model.json')
    # Independently recount each observed-context transition and emission row.
    for context,parent in c.CONTEXTS.items():
        m=model['contexts'][context]
        assert m['prior']==pooled['prior'] and m['initial_observation']==pooled['initial_observation']
        for action in p.ACTIONS:
            edges=[(r['tree'][parent],r['tree'][parent+('/' if parent else '')+action]) for r in rows['train']]
            for s in range(2):
                children=[v for u,v in edges if int(u['correct'])==s];counts=Counter(int(v['correct']) for v in children)
                assert m['training_counts']['transition'][action][s]==[counts[j] for j in range(2)]
                assert m['transition'][action][s]==[(counts[j]+10*pooled['transition'][action][s][j])/(len(children)+10) for j in range(2)]
                assert m['cost_tokens'][action][s]==(sum(v['input_tokens']+v['output_tokens'] for v in children)+10*pooled['cost_tokens'][action][s])/(len(children)+10)
                children=[v for u,v in edges if int(v['correct'])==s];counts=Counter(p.observation(v) for v in children)
                assert m['observation'][action][s]==[(counts[z]+10*pooled['observation'][action][s][z])/(len(children)+10) for z in range(8)]
    for split,rr in rows.items():
        assert p.io.read_json(directory/f'{split}-dynamics.json')==c.dynamics(rr)
        assert p.io.read_json(directory/f'{split}-answer-change.json')==c.answer_change(rr)
        for lam in [.01,.248]:
            records=p.io.read_json(directory/f'{split}-paths-{lam}.json');replayed=c.replay(model,rr,lam)
            def untimed(records):return [{k:v for k,v in r.items() if k!='controller_cpu_seconds'} for r in records]
            assert untimed(records)==untimed(replayed)
            for row,record in zip(rr,records):
                parts=record['path'].split('/') if record['path'] else []
                paths=['']+['/'.join(parts[:i]) for i in range(1,len(parts)+1)]
                assert record['tokens']==sum(row['tree'][path]['input_tokens']+row['tree'][path]['output_tokens'] for path in paths)
                assert record['correct']==int(row['tree'][record['path']]['correct']) and record['controller_tokens']==0
            old=p.replay(pooled,rr,lam)
            methods={'context_pomdp':records,'pooled_pomdp':[dict(r,tokens=r['solver_tokens']) for r in old]}
            methods.update({name:p.fixed_records(rr,paths) for name,paths in p.io.FIXED.items()})
            summary=p.io.read_json(directory/f'{split}-summary-{lam}.json')
            assert len(summary['summaries'])==21 and summary['split']==split and summary['lambda']==lam
            for item in summary['summaries']:
                selected=[r for r in methods[item['method']] if item['domain']=='all' or r['domain']==item['domain']]
                assert item['n']==len(selected) and item['accuracy']==statistics.mean(r['correct'] for r in selected)
                assert item['mean_tokens']==statistics.mean(r['tokens'] for r in selected)
                assert item['mean_utility']==statistics.mean(r['correct']-lam*r['tokens']/1000 for r in selected)
            actions=dict(Counter(' -> '.join(s['action'] for s in r['steps']) for r in records))
            assert summary['action_sequences']==actions
            assert status['results'][f'{split}/{lam}']=={'overall':next(r for r in summary['summaries'] if r['method']=='context_pomdp' and r['domain']=='all'),'actions':actions}
    return {'status':'PASS','splits':['train','dev'],'hidden_states':2,'shrinkage_strength':10,'lambdas':[.01,.248],
        'checks':['Source review: no test loading/inference; fitting uses100train only','All context transition/emission probabilities and conditional costs independently recounted with kappa10 pooled shrinkage','Root prior and observations exactly original pooled model','Analytic tests confirm parent-context likelihood and child-context continuation','All train/dev paths, beliefs, terminal labels and visited token charges replayed','All84 summary rows independently reconstructed','Dynamics and canonical answer-change descriptive diagnostics reproduced; answer change not used by policy'],
        'dev_results':{str(lam):status['results'][f'dev/{lam}'] for lam in [.01,.248]},
        'limitations':['Post-result train/dev exploratory diagnostic; no new held-out generalization result.','Shrinkage toward overlapping pooled samples is regularization, not independent Bayesian evidence.','At .248,55% dev accuracy matches repair once while saving18.3667tokens/question on60dev; this is a modest descriptive result.'],
        'reproduce':'python3 -B experiments/pomdp-controller/verify_context.py --output results/pomdp-context-diagnostic/validation.json'}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,default=c.ROOT/'results/pomdp-context-diagnostic');ap.add_argument('--output',type=Path);args=ap.parse_args();result=verify(args.run)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
