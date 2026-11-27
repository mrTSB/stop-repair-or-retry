"""Accuracy-first refinement with label-independent exact gate breakpoints."""
import copy, hashlib, itertools, json, math, pickle, sys, time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/controller-semantic-hillclimb/followup/round2'))
sys.path.insert(0,str(ROOT/'scripts/reproduction'))
import development_bayesian_hybrid_v3 as runner
import audit_bayesian_hybrid as audit
import replay_latest as replay
from threadpoolctl import threadpool_limits
original=runner.original; policy=original.policy; io=runner.io
OUT=Path(sys.argv[1]); OUT.mkdir(parents=True,exist_ok=False)
PENALTIES=[0.,.0001,.00025,.0005,.00075,.001,.0015,.002,.0025,.003,.0035,.004,.0045,.005,.006,.007,.008,.009,.01,.0125,.015,.0175,.02,.025,.03,.04,.05,.075,.1,.2,.5,1.]
THRESHOLDS=[0.,.0025,.005,.01,.015,.02,.03,.04,.05,.06,.07,.08,.09,.1,.12,.14,.16,.2,.25,.3]
WEIGHTS=[.01,.025,.05,.075,.1,.125,.15,.175,.2,.225,.25,.3,.35,.4,.45,.5,.6,.65,.7,.75,.8,.9,.95,.975,.99]
SUBJECTS=['math','chemistry','physics']
def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def metrics(items):
    return {'n':len(items),'correct':sum(int(x['correct']) for x in items),'accuracy':sum(int(x['correct']) for x in items)/len(items),'mean_compute':math.fsum(x['compute'] for x in items)/len(items)}
def execute(row,trees,judges,model,subject,settings,weight):
    expert=model.experts[subject]; child=model.models[expert['model']]
    if isinstance(child,policy.Planner):
        child.weights=[weight,1-weight]
        return policy.evaluate(row,trees[row['id']],judges,child,settings,'C')
    return policy.core.evaluate(row,trees[row['id']],judges,child,settings,'C')
def choose(candidates,indices,ceiling):
    # Independent subject-wise DP: cheapest cost for each integer correct count.
    fronts={}
    for s in SUBJECTS:
        best={}
        for k,c in enumerate(candidates[s]):
            selected=[c['items'][i] for i in indices[s]]
            correct=sum(x['correct'] for x in selected); cost=math.fsum(x['compute'] for x in selected)
            if correct not in best or (cost,k)<best[correct]:best[correct]=(cost,k)
        fronts[s]=best
    dp={0:(0.,{})}
    for s in SUBJECTS:
        new={}
        for n,(cost,choices) in dp.items():
            for m,(extra,k) in fronts[s].items():
                if n+m not in new or cost+extra<new[n+m][0]:new[n+m]=(cost+extra,{**choices,s:k})
        dp=new
    size=sum(map(len,indices.values()))
    eligible=[(n,cost,choice) for n,(cost,choice) in dp.items() if cost/size<=ceiling+1e-10]
    if not eligible:raise ValueError('No feasible candidate')
    n,cost,choice=min(eligible,key=lambda x:(-x[0],x[1]))
    return choice,{'n':size,'correct':n,'accuracy':n/size,'mean_compute':cost/size}
with original.prior.guard.no_test_reads(),original.prior.guarded.no_model_calls(),threadpool_limits(limits=1):
    folder=ROOT/'runs/controller-semantic-followup/round2/development-bayesian-model-hybrid-v3'
    spec=io.read(folder/'registration.json'); summary=io.read(folder/'summary.json')
    write('registration.json',{'source_sha256':io.sha(__file__),'parent_sha256':summary['C_candidate']['model_sha256'],
        'penalties':PENALTIES,'thresholds':THRESHOLDS,'physics_prior_weights':WEIGHTS,
        'selection':'Maximum development accuracy without a compute ceiling; ties by lower compute; secondary comparisons under original C and B ceilings',
        'architecture_changes':False,'trained_weights_changed':False,'tokenizer_changed':False,
        'reused_development':True,'fresh_validation':False,'test_used':False,'new_model_calls':0,
        'threshold_candidates':'Fixed thresholds plus one representative in every gate interval derived from information values only; no labels used to generate thresholds',
        'threshold_optimization':'Evaluate original runtime at zero and 1e6 thresholds; select by its exact information-value comparison; directly replay and blind-audit selected configuration.'})
    with runner.configured():bundle,_,_,record=original.inputs(SimpleNamespace(previous=Path(spec['previous']),output=folder))
    assert record==spec
    rows,trees,judges,_,caches,*_=bundle
    assert len(rows)==256
    chosen=summary['C_candidate']; assert io.sha(chosen['model_path'])==chosen['model_sha256']
    original_bytes=Path(chosen['model_path']).read_bytes(); model=pickle.loads(original_bytes)
    by_subject={s:[r for r in rows if r['subject']==s] for s in SUBJECTS}
    candidates={s:[] for s in SUBJECTS}
    original_items=io.read(folder/'private-selected-items.json')
    with ExitStack() as stack:
        for child in model.models.values():
            for parent in child.models if isinstance(child,policy.Planner) else [child]:
                stack.enter_context(original.canonical.cached_model(parent,caches))
        # Reconfirm the exact baseline before searching.
        for row,expected in zip(rows,original_items,strict=True):
            replay.same(policy.routed(row,trees[row['id']],judges,model,{},'C'),expected)
        for s in SUBJECTS:
            baseline=[x for x in original_items if x['id'] in {r['id'] for r in by_subject[s]}]
            byid={x['id']:x for x in baseline}; baseline=[byid[r['id']] for r in by_subject[s]]
            candidates[s].append({'settings':copy.deepcopy(model.experts[s]['settings']),'weight':.25 if s=='physics' else None,'items':baseline,'baseline':True})
            for weight,penalty in itertools.product(WEIGHTS if s=='physics' else [None],PENALTIES):
                low=[]; high=[]
                for row in by_subject[s]:
                    lo=execute(row,trees,judges,model,s,{'penalty':penalty,'threshold':0.},weight)
                    hi=execute(row,trees,judges,model,s,{'penalty':penalty,'threshold':1e6},weight)
                    assert not hi['decisions'][0]['queried']
                    assert math.isclose(lo['decisions'][0]['information_value'],hi['decisions'][0]['information_value'],abs_tol=1e-12)
                    low.append(lo);high.append(hi)
                boundaries=sorted({float(lo['decisions'][0]['information_value'])-1e-9 for lo in low if lo['decisions'][0]['information_value']>1e-9})
                extra=[(a+b)/2 for a,b in zip(boundaries,boundaries[1:])]
                if boundaries:extra.append(boundaries[-1]+1e-6)
                for threshold in sorted(set(THRESHOLDS+extra)):
                    items=[lo if lo['decisions'][0]['information_value']>threshold+1e-9 else hi for lo,hi in zip(low,high)]
                    candidates[s].append({'settings':{'penalty':penalty,'threshold':threshold},'weight':weight,'items':items})
                print(json.dumps({'subject':s,'weight':weight,'penalty':penalty,'subject_candidates':len(candidates[s])}),flush=True)
        indices={s:list(range(len(by_subject[s]))) for s in SUBJECTS}
        selected,value=choose(candidates,indices,float('inf'))
        under_c,value_c=choose(candidates,indices,chosen['dev']['mean_compute'])
        under_b,value_b=choose(candidates,indices,summary['B']['dev']['mean_compute']-1e-8)
        selected_params={s:{k:v for k,v in candidates[s][selected[s]].items() if k!='items'} for s in SUBJECTS}
        # Save an unloaded model with only the existing settings/weights changed.
        clean=pickle.loads(original_bytes)
        for s,param in selected_params.items():
            clean.experts[s]['settings']=param['settings']
            child=clean.models[clean.experts[s]['model']]
            if isinstance(child,policy.Planner):child.weights=[param['weight'],1-param['weight']]
        (OUT/'hybrid.pkl').write_bytes(pickle.dumps(clean,protocol=5))
        # Replay from serialized tuned model with frozen caches, and only paid observations.
        selected_items=[]
        with ExitStack() as selected_stack:
            for child in clean.models.values():
                for parent in child.models if isinstance(child,policy.Planner) else [child]:
                    selected_stack.enter_context(original.canonical.cached_model(parent,caches))
            expected={item['id']:item for s in SUBJECTS for item in candidates[s][selected[s]]['items']}
            for row in rows:
                item=policy.routed(row,trees[row['id']],judges,clean,{},'C')
                replay.same(item,expected[row['id']])
                observed,acquired=replay.paid(item,trees[row['id']],judges)
                blind=policy.routed({**row,'answer':''},observed,acquired,clean,{},'C')
                replay.same(blind,item,grade=False)
                assert blind['decisions']==item['decisions']
                selected_items.append(item)
        checked=audit.check_items(selected_items,rows,trees,judges)
        audit.close(checked['mean_compute'],value['mean_compute']);assert checked['correct']==value['correct']
        # Exhaustively check subject frontier combinations against the separate DP.
        fronts={}
        for s in SUBJECTS:
            best={}
            for k,c in enumerate(candidates[s]):
                m=metrics(c['items']); n=m['correct'];cost=math.fsum(x['compute'] for x in c['items'])
                if n not in best or cost<best[n][0]:best[n]=(cost,k)
            fronts[s]=[(n,cost,k) for n,(cost,k) in best.items()]
        feasible=[]
        for combo in itertools.product(*(fronts[s] for s in SUBJECTS)):
            cost=math.fsum(c[1] for c in combo)/256
            if math.isfinite(cost):feasible.append((sum(c[0] for c in combo),cost))
        independent_best=min(feasible,key=lambda x:(-x[0],x[1]))
        assert independent_best[0]==value['correct'];audit.close(independent_best[1],value['mean_compute'])
        # Selection sensitivity only: reused DEV folds are not fresh validation.
        rng=np.random.default_rng(20261127);folds={s:np.array_split(rng.permutation(len(by_subject[s])),4) for s in SUBJECTS}
        cv=[]
        for fold in range(4):
            training={s:[int(i) for f,part in enumerate(folds[s]) if f!=fold for i in part] for s in SUBJECTS}
            holdout={s:[int(i) for i in folds[s][fold]] for s in SUBJECTS}
            baseline_train=[candidates[s][0]['items'][i] for s in SUBJECTS for i in training[s]]
            choice,_=choose(candidates,training,float('inf'))
            items=[candidates[s][choice[s]]['items'][i] for s in SUBJECTS for i in holdout[s]]
            base=[candidates[s][0]['items'][i] for s in SUBJECTS for i in holdout[s]]
            cv.append({'fold':fold,'tuned':metrics(items),'baseline':metrics(base),'selected_indices':choice})
        write('candidate-metrics.json',{s:[{'index':k,**{a:b for a,b in c.items() if a!='items'},'metrics':metrics(c['items'])} for k,c in enumerate(cs)] for s,cs in candidates.items()})
        # Lossless compressed complete per-candidate outcomes; no candidates discarded.
        import gzip
        with gzip.GzipFile(filename=str(OUT/'candidate-outcomes.json.gz'),mode='wb',mtime=0) as raw:
            import io as stdio
            with stdio.TextIOWrapper(raw,encoding='utf-8') as stream:
                encoder=json.JSONEncoder(separators=(',',':'),allow_nan=False)
                stream.write('{')
                for si,(subject,cs) in enumerate(candidates.items()):
                    if si:stream.write(',')
                    stream.write(json.dumps(subject)+':[')
                    for k,c in enumerate(cs):
                        if k:stream.write(',')
                        entry={'index':k,'rows':[{'id':x['id'],'path':x['path'],'correct':x['correct'],'compute':x['compute']} for x in c['items']]}
                        for chunk in encoder.iterencode(entry):stream.write(chunk)
                    stream.write(']')
                stream.write('}')
        write('per-question.json',selected_items)
        base_byid={x['id']:x for x in original_items}
        changed=[{'id':x['id'],'subject':next(r['subject'] for r in rows if r['id']==x['id']),'old_path':base_byid[x['id']]['path'],'new_path':x['path'],'old_correct':base_byid[x['id']]['correct'],'new_correct':x['correct'],'compute_difference':x['compute']-base_byid[x['id']]['compute']} for x in selected_items if x['path']!=base_byid[x['id']]['path'] or x['compute']!=base_byid[x['id']]['compute']]
        write('changed-questions.json',changed)
        result={'passed':True,'baseline_C':chosen['dev'],'tuned_C':value,'B':summary['B']['dev'],
            'selected':selected_params,'subject_candidate_counts':{s:len(c) for s,c in candidates.items()},
            'alternative_under_original_C_ceiling':{'metrics':value_c,'settings':{s:{k:v for k,v in candidates[s][under_c[s]].items() if k!='items'} for s in SUBJECTS}},
            'alternative_under_B_ceiling':{'metrics':value_b,'settings':{s:{k:v for k,v in candidates[s][under_b[s]].items() if k!='items'} for s in SUBJECTS}},
            'selection_sensitivity_four_dev_folds':cv,'fresh_validation':False,'test_used':False,'architecture_changed':False,
            'new_model_calls':0,'cloud_cost':0,'checks':{'baseline_reproduced':True,'direct_selected_replays':256,'blinded_paid_replays':256,'independent_billing_and_correctness':True,'independent_selection_check':True},
            'model_sha256':io.sha(OUT/'hybrid.pkl'),'source_sha256':io.sha(__file__)}
        write('summary.json',result)
        write('complete.json',{'passed':True,'digests':{p.name:io.sha(p) for p in OUT.iterdir() if p.is_file()},'test_used':False})
        print(json.dumps(result),flush=True)
