"""Train/dev-only context dynamics diagnostic. Two hidden states, unchanged observations."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from stop_repair_retry import pomdp_experiment as p

CONTEXTS={'initial':'','after_repair':'repair','after_retry':'retry'}
STRENGTH=10.0
LAMBDAS=[.01,.248]

def fit(rows):
    pooled=p.fit_model(rows);models={}
    for context,parent in CONTEXTS.items():
        transitions={a:[[0,0],[0,0]] for a in p.ACTIONS};observations={a:[[0]*8 for _ in range(2)] for a in p.ACTIONS};costs={a:[[],[]] for a in p.ACTIONS}
        for r in rows:
            s=int(r['tree'][parent]['correct'])
            for a in p.ACTIONS:
                child=r['tree'][parent+('/' if parent else '')+a];sp=int(child['correct'])
                transitions[a][s][sp]+=1;observations[a][sp][p.observation(child)]+=1
                costs[a][s].append(child['input_tokens']+child['output_tokens'])
        m={'states':pooled['states'],'observations':pooled['observations'],'prior':pooled['prior'],
           'initial_observation':pooled['initial_observation'],'terminal_reward':pooled['terminal_reward'],
           'transition':{a:[[(v+STRENGTH*prior)/(sum(counts)+STRENGTH) for v,prior in zip(counts,pooled['transition'][a][s])] for s,counts in enumerate(transitions[a])] for a in p.ACTIONS},
           'observation':{a:[[(v+STRENGTH*prior)/(sum(counts)+STRENGTH) for v,prior in zip(counts,pooled['observation'][a][s])] for s,counts in enumerate(observations[a])] for a in p.ACTIONS},
           'cost_tokens':{a:[(sum(v)+STRENGTH*pooled['cost_tokens'][a][s])/(len(v)+STRENGTH) for s,v in enumerate(costs[a])] for a in p.ACTIONS},
           'training_counts':{'transition':transitions,'observations':observations,'cost_samples':{a:[len(x) for x in costs[a]] for a in p.ACTIONS}}}
        p.validate_model(m);models[context]=m
    return {'contexts':models,'pooled_prior_model':pooled,'shrinkage_strength':STRENGTH,'hidden_states':['incorrect','correct']}

def plan(model,b,remaining,penalty,context):
    m=model['contexts'][context];values={'stop':p.dot(b,m['terminal_reward'])}
    if remaining:
        for a in p.ACTIONS:
            pred=p.predict(m,b,a);value=-penalty*p.dot(b,m['cost_tokens'][a])/1000
            for z in range(len(m['observations'])):
                w=[v*o[z] for v,o in zip(pred,m['observation'][a])];prob=sum(w)
                if prob:value+=prob*plan(model,[v/prob for v in w],remaining-1,penalty,'after_'+a)['value']
            values[a]=value
    best='stop'
    for a in p.ACTIONS:
        if a in values and values[a]>values[best]+p.CONFIG['tie_tolerance']:best=a
    return {'action':best,'value':values[best],'action_values':values}

def replay(model,rows,penalty):
    out=[]
    for row in rows:
        path='';visited=[''];context='initial';b=p.initial_belief(model['pooled_prior_model'],p.observation(row['tree']['']));steps=[];started=time.perf_counter()
        for remaining in (2,1):
            decision=plan(model,b,remaining,penalty,context);steps.append({'path':path,'context':context,'belief':b,'observation':p.OBSERVATIONS[p.observation(row['tree'][path])],'remaining':remaining,**decision})
            if decision['action']=='stop':break
            a=decision['action'];path+=('/' if path else '')+a;visited.append(path)
            b=p.update(model['contexts'][context],b,a,p.observation(row['tree'][path]));context='after_'+a
        elapsed=time.perf_counter()-started
        out.append({'id':row['id'],'domain':row['domain'],'path':path,'steps':steps,'final_belief':b,'correct':int(row['tree'][path]['correct']),
            'tokens':sum(row['tree'][q]['input_tokens']+row['tree'][q]['output_tokens'] for q in visited),'controller_tokens':0,'controller_cpu_seconds':elapsed})
    return out

def dynamics(rows):
    output=[]
    for context,parent in CONTEXTS.items():
        for a in p.ACTIONS:
            pairs=[(r['tree'][parent],r['tree'][parent+('/' if parent else '')+a]) for r in rows]
            wrong=[b['correct'] for s,b in pairs if not s['correct']];right=[not b['correct'] for s,b in pairs if s['correct']]
            output.append({'context':context,'action':a,'wrong_parents':len(wrong),'fixed':sum(wrong),'fix_rate':statistics.mean(wrong) if wrong else None,
              'correct_parents':len(right),'broken':sum(right),'harm_rate':statistics.mean(right) if right else None,'net_correct_gain':sum(wrong)-sum(right)})
    return output

def answer_change(rows):
    spec=importlib.util.spec_from_file_location('scoring',p.io.DEFAULT_DATA/'scoring.py');scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    result=[]
    for parent in p.ACTIONS:
        groups={g:[] for g in ['same','changed','unavailable']}
        for r in rows:
            a,b=r['tree'][''],r['tree'][parent]
            group=('same' if scorer.canonical_answer(a['answer'])==scorer.canonical_answer(b['answer']) else 'changed') if a['answer_parse_valid'] and b['answer_parse_valid'] else 'unavailable'
            groups[group].append(r)
        for group,rr in groups.items():
            result.append({'previous_action':parent,'answer_change':group,'n':len(rr),'current_correct':sum(r['tree'][parent]['correct'] for r in rr),
              'next_repair_correct':sum(r['tree'][parent+'/repair']['correct'] for r in rr),'next_retry_correct':sum(r['tree'][parent+'/retry']['correct'] for r in rr)})
    return result

def run(directory):
    if directory.exists():raise ValueError('Refuse overwrite')
    directory.mkdir(parents=True)
    p.io.save(directory/'protocol.json',{'created_utc':p.io.now(),'splits':['train','dev'],'lambdas':LAMBDAS,'shrinkage_strength':STRENGTH,
      'changes':'Condition transitions, successor observation distributions, and expected costs on known context: initial, after repair, after retry. Keep two hidden states, confidence/validity observations, pooled domains, root prior, and two-call planning.',
      'shrinkage':'Each probability row adds10 pseudo-observations distributed according to the pooled training estimate; each cost mean adds10 pooled-mean pseudo-samples. Pooled estimate overlaps context samples; this is regularization, not independent Bayesian evidence.',
      'selection':'One fixed model variant and two existing lambda values, no hyperparameter search. Dev-only exploration; no test evaluation or new primary result.',
      'answer_change':'Descriptive train/dev diagnostic only; NOT supplied to this policy or fitted into observations.',
      'script_sha256':p.io.sha(__file__),'core_sha256':p.io.sha(p.__file__),'helper_sha256':p.io.sha(p.io.__file__),'design_sha256':p.io.sha(p.DESIGN),
      'dataset_manifest_sha256':p.io.sha(p.io.DEFAULT_DATA/'FILE_MANIFEST.json'),'paid_calls':0})
    rows={s:p.io.load_dataset(p.io.DEFAULT_DATA,s) for s in ('train','dev')};model=fit(rows['train']);p.io.save(directory/'model.json',model)
    summaries={}
    for split in rows:
        p.io.save(directory/f'{split}-dynamics.json',dynamics(rows[split]));p.io.save(directory/f'{split}-answer-change.json',answer_change(rows[split]))
        for lam in LAMBDAS:
            rr=replay(model,rows[split],lam);p.io.save(directory/f'{split}-paths-{lam}.json',rr)
            old=p.replay(model['pooled_prior_model'],rows[split],lam)
            methods={'context_pomdp':rr,'pooled_pomdp':[{'id':r['id'],'domain':r['domain'],'correct':r['correct'],'tokens':r['solver_tokens']} for r in old]}
            methods.update({name:p.fixed_records(rows[split],paths) for name,paths in p.io.FIXED.items()})
            table=[]
            for name,records in methods.items():
                for domain in ['all','math','science']:
                    selected=[r for r in records if domain=='all' or r['domain']==domain]
                    table.append({'method':name,'domain':domain,'n':len(selected),'accuracy':statistics.mean(r['correct'] for r in selected),
                      'mean_tokens':statistics.mean(r['tokens'] for r in selected),'mean_utility':statistics.mean(r['correct']-lam*r['tokens']/1000 for r in selected)})
            summary={'split':split,'lambda':lam,'summaries':table,'action_sequences':dict(Counter(' -> '.join(s['action'] for s in r['steps']) for r in rr)),
                     'controller_cpu_seconds':sum(r['controller_cpu_seconds'] for r in rr)}
            p.io.save(directory/f'{split}-summary-{lam}.json',summary)
            summaries[f'{split}/{lam}']={'overall':next(r for r in table if r['method']=='context_pomdp' and r['domain']=='all'),'actions':summary['action_sequences']}
    p.io.save(directory/'status.json',{'status':'complete','finished_utc':p.io.now(),'new_paid_calls':0,'test_evaluated':False,'results':summaries})
    print(summaries)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();run(args.output)
