"""Two-vs-four hidden states with correctness-constrained latent tree fitting.

Gold correctness is used only in training inference. Held-out controller replay
uses the unchanged confidence/validity observation API, never gold or siblings.
"""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from stop_repair_retry import pomdp_experiment as p
import context_diagnostic as c
import calibration_cv as cv

ITERATIONS=20
STRENGTH=10.0
COUNTS=[2,4]
LAMBDAS=[.01,.248]

def context(path):return 'initial' if not path else 'after_'+path.split('/')[-1]
def children(tree,path):return [(a,path+('/' if path else '')+a) for a in p.ACTIONS if path+('/' if path else '')+a in tree]
def parent(path):return tuple(path.rsplit('/',1)) if '/' in path else ('',path)

def seed(row,n):
    gamma={}
    for path,node in row['tree'].items():
        g=[0.0]*n
        if node['correct']:g[0]=1
        elif n==2:g[1]=1
        elif len(children(row['tree'],path))==2:
            prefix=path+'/' if path else ''
            label=1 if row['tree'][prefix+'repair']['correct'] else (2 if row['tree'][prefix+'retry']['correct'] else 3)
            g[label]=1
        else:g[1:]=[1/(n-1)]*(n-1)
        gamma[path]=g
    xi={}
    for path in row['tree']:
        if path:
            par,a=parent(path);xi[path]=[[x*y for y in gamma[path]] for x in gamma[par]]
    return {'gamma':gamma,'xi':xi}

def posterior(model,row):
    """Exact sum-product on the complete training tree with correctness masks."""
    tree=row['tree'];n=len(model['pooled_prior_model']['states']);up={};scale=0.0
    for path in sorted(tree,key=lambda x:x.count('/')+(1 if x else 0),reverse=True):
        node=tree[path];obs=p.observation(node)
        if not path:emission=model['pooled_prior_model']['initial_observation']
        else:
            par,a=parent(path);emission=model['contexts'][context(par)]['observation'][a]
        weights=[emission[s][obs] if (s==0)==bool(node['correct']) else 0.0 for s in range(n)]
        for a,child in children(tree,path):
            t=model['contexts'][context(path)]['transition'][a]
            weights=[w*sum(t[s][sp]*up[child][sp] for sp in range(n)) for s,w in enumerate(weights)]
        mass=sum(weights)
        if mass<=0:raise ValueError('Impossible training evidence')
        up[path]=[w/mass for w in weights];scale+=math.log(mass)
    root=[x*y for x,y in zip(model['pooled_prior_model']['prior'],up[''])]
    likelihood=math.log(sum(root))+scale;gamma={'':p.normalize(root)};xi={}
    for path in sorted(tree,key=lambda x:x.count('/')+(1 if x else 0)):
        for a,child in children(tree,path):
            t=model['contexts'][context(path)]['transition'][a];edge=[]
            for s in range(n):
                denominator=sum(t[s][sp]*up[child][sp] for sp in range(n))
                edge.append([gamma[path][s]*t[s][sp]*up[child][sp]/denominator for sp in range(n)])
            xi[child]=edge;gamma[child]=[sum(edge[s][sp] for s in range(n)) for sp in range(n)]
    return {'gamma':gamma,'xi':xi,'log_likelihood':likelihood}

def maximize(rows,posteriors,n):
    def zeros(a,b):return [[0.0]*b for _ in range(a)]
    prior=[0.0]*n;rootobs=zeros(n,8)
    contexts=['pooled',*c.CONTEXTS]
    trans={ctx:{a:zeros(n,n) for a in p.ACTIONS} for ctx in contexts}
    obs={ctx:{a:zeros(n,8) for a in p.ACTIONS} for ctx in contexts}
    cost={ctx:{a:[0.0]*n for a in p.ACTIONS} for ctx in contexts}
    weight={ctx:{a:[0.0]*n for a in p.ACTIONS} for ctx in contexts}
    for row,post in zip(rows,posteriors):
        for s,g in enumerate(post['gamma']['']):prior[s]+=g;rootobs[s][p.observation(row['tree'][''])]+=g
        for path,node in row['tree'].items():
            if not path:continue
            par,a=parent(path);z=p.observation(node);tokens=node['input_tokens']+node['output_tokens']
            for ctx in ('pooled',context(par)):
                for s in range(n):
                    for sp in range(n):trans[ctx][a][s][sp]+=post['xi'][path][s][sp]
                    obs[ctx][a][s][z]+=post['gamma'][path][s]
                    g=post['gamma'][par][s];cost[ctx][a][s]+=g*tokens;weight[ctx][a][s]+=g
    # Keep total prior pseudomass on correct/incorrect identical for 2 and 4 states.
    alpha=[1.0]+[1/(n-1)]*(n-1)
    states=['correct']+(['incorrect'] if n==2 else ['incorrect_type_1','incorrect_type_2','incorrect_type_3'])
    pooled={'states':states,'observations':p.OBSERVATIONS,'terminal_reward':[1]+[0]*(n-1),
      'prior':p.normalize([x+y for x,y in zip(prior,alpha)]),
      'initial_observation':cv.unknown_equal([p.smooth(row,1) for row in rootobs],rootobs),
      'transition':{a:[p.normalize([v+prior for v,prior in zip(row,alpha)]) for row in trans['pooled'][a]] for a in p.ACTIONS},
      'observation':{a:cv.unknown_equal([p.smooth(row,1) for row in obs['pooled'][a]],obs['pooled'][a]) for a in p.ACTIONS},
      'cost_tokens':{a:[cost['pooled'][a][s]/weight['pooled'][a][s] if weight['pooled'][a][s]>1e-15 else sum(cost['pooled'][a])/sum(weight['pooled'][a]) for s in range(n)] for a in p.ACTIONS}}
    models={}
    for ctx in c.CONTEXTS:
        m={k:pooled[k] for k in ('states','observations','terminal_reward','prior','initial_observation')}
        m['transition']={a:[p.normalize([v+STRENGTH*pr for v,pr in zip(row,pooled['transition'][a][s])]) for s,row in enumerate(trans[ctx][a])] for a in p.ACTIONS}
        m['observation']={a:cv.unknown_equal([p.normalize([v+STRENGTH*pr for v,pr in zip(row,pooled['observation'][a][s])]) for s,row in enumerate(obs[ctx][a])],obs['pooled'][a]) for a in p.ACTIONS}
        m['cost_tokens']={a:[(cost[ctx][a][s]+STRENGTH*pooled['cost_tokens'][a][s])/(weight[ctx][a][s]+STRENGTH) for s in range(n)] for a in p.ACTIONS}
        p.validate_model(m);models[ctx]=m
    p.validate_model(pooled)
    return {'contexts':models,'pooled_prior_model':pooled,'states':states,
      'expected_training_counts':{'root':prior,'root_observation':rootobs,'transitions':trans,'observations':obs,'cost_weights':weight}}

def fit(rows,n):
    if n not in COUNTS or not rows or any(r['split']!='train' for r in rows) or len({r['id'] for r in rows})!=len(rows):raise ValueError('Unique training questions and 2/4 states required')
    model=maximize(rows,[seed(row,n) for row in rows],n);trace=[]
    for iteration in range(ITERATIONS):
        post=[posterior(model,row) for row in rows];trace.append(sum(x['log_likelihood'] for x in post));model=maximize(rows,post,n)
    trace.append(sum(posterior(model,row)['log_likelihood'] for row in rows))
    model.update(fit_ids=[r['id'] for r in rows],iterations=ITERATIONS,training_log_likelihood=trace,
      note='Correctness clamps only during training. Incorrect types remain latent; seed semantics are not guaranteed. Empirical shrinkage/calibration may break unregularized likelihood monotonicity.')
    return model

def run(output):
    if output.exists():raise ValueError('Refuse overwrite')
    output.mkdir(parents=True)
    p.io.save(output/'protocol.json',{'created_utc':p.io.now(),'state_counts':COUNTS,'lambdas':LAMBDAS,'iterations':ITERATIONS,'context_shrinkage':STRENGTH,
      'scope':'Five-fold training comparison, then only one training-selected candidate evaluated on dev. No test evaluation, no new observations, no paid calls.',
      'state_semantics':'Correct plus one or three latent incorrect types. Nonleaf cached repair/retry outcomes initialize incorrect types only. Leaves start uniform over incorrect types. All incorrect types can change during fitting; no true repairability label is claimed.',
      'training':'Correctness-constrained tree sum-product expected counts across all seven nodes and six edges. Gold masks used only on fitting questions. Conditioned on parent state, child branches are assumed independent; this is an approximation for same-question responses.',
      'selection':'Maximize OOF training accuracy, then minimize tokens, then fewer states, then smaller lambda. Declared before results; only selected candidate evaluated on dev.',
      'regularization':'Same total correct/incorrect Dirichlet pseudomass in priors and transition rows across state counts. Observation Laplace1 per symbol, context shrinkage10, globally unseen emissions equal across states. Empirical priors are re-estimated each iteration; no monotonicity/global-optimum guarantee.',
      'fold_rule':'Reuse deterministic question-grouped domain-balanced fold assignment; every branch stays with its question.',
      'script_sha256':p.io.sha(__file__),'context_sha256':p.io.sha(c.__file__),'calibration_sha256':p.io.sha(cv.__file__),'core_sha256':p.io.sha(p.__file__),
      'dataset_manifest_sha256':p.io.sha(p.io.DEFAULT_DATA/'FILE_MANIFEST.json')})
    rows=p.io.load_dataset(p.io.DEFAULT_DATA,'train');assignment=cv.folds(rows);p.io.save(output/'folds.json',assignment)
    records={(n,lam):[] for n in COUNTS for lam in LAMBDAS}
    for fold in range(5):
        training=[r for r in rows if assignment[r['id']]!=fold];held=[r for r in rows if assignment[r['id']]==fold]
        assert len(training)==80 and len(held)==20
        for n in COUNTS:
            model=fit(training,n);p.io.save(output/f'fold-{fold}-states-{n}-model.json',model)
            for lam in LAMBDAS:
                rr=c.replay(model,held,lam);records[(n,lam)]+=rr;p.io.save(output/f'fold-{fold}-states-{n}-{lam}-paths.json',rr)
            print(f'Completed fold {fold+1}/5, states={n}',flush=True)
    summaries=[{'states':n,'lambda':lam,**cv.summary(rr)} for (n,lam),rr in records.items()]
    selected=min(summaries,key=lambda s:(-s['accuracy'],s['mean_tokens'],s['states'],s['lambda']))
    p.io.save(output/'oof-summary.json',summaries);p.io.save(output/'selection.json',{'selected_utc':p.io.now(),'candidate':selected,'dev_not_loaded_yet':True})
    model=fit(rows,selected['states']);p.io.save(output/'selected-model.json',model)
    dev=p.io.load_dataset(p.io.DEFAULT_DATA,'dev');rr=c.replay(model,dev,selected['lambda']);p.io.save(output/'selected-dev-paths.json',rr);p.io.save(output/'selected-dev-summary.json',cv.summary(rr))
    p.io.save(output/'status.json',{'status':'complete','finished_utc':p.io.now(),'new_paid_calls':0,'test_evaluated':False})
    print({'oof':summaries,'selected':selected,'dev':cv.summary(rr)},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();run(args.output)
