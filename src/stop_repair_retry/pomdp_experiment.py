"""Training-only two-state fit and generic finite-horizon belief planning; no inference."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time
from . import llm_experiment as io

ROOT=Path(__file__).resolve().parents[2]
DESIGN=ROOT/'experiments/pomdp-controller/DESIGN.md'
A_RESULTS=ROOT/'results/llm-controller-full-evaluation'
ACTIONS=('repair','retry')
OBSERVATIONS=[f'{c}:{v}' for c in ('low','medium','high','missing') for v in ('invalid','valid')]
CONFIG={'states':['incorrect','correct'],'observations':OBSERVATIONS,'smoothing':1.0,'primary_lambda':0.01,
        'lambda_grid':[0,0.001,0.01,0.1,1],'horizon':2,'tie_order':['stop','repair','retry'],'tie_tolerance':1e-12,
        'pooling':'all domains and depths','observation_features':['confidence','answer_parse_valid']}

def observation(node):
    """Only these two parser fields enter the policy; never correctness or gold."""
    c=node['confidence'] if node['confidence'] in ('low','medium','high') else 'missing'
    return OBSERVATIONS.index(c+(':'+('valid' if node['answer_parse_valid'] else 'invalid')))

def normalize(values):
    total=sum(values)
    if total<=0:raise ValueError('Impossible observation: zero probability')
    return [x/total for x in values]

def smooth(counts,alpha):return normalize([v+alpha for v in counts])
def dot(a,b):return sum(x*y for x,y in zip(a,b))

def validate_model(model):
    n=len(model['states']);k=len(model['observations'])
    def vector(v,size,distribution=False):
        if len(v)!=size or any(not math.isfinite(x) or x<0 for x in v):raise ValueError('Invalid model vector')
        if distribution and abs(sum(v)-1)>1e-9:raise ValueError('Probabilities must sum to one')
    if n<2 or k<1:raise ValueError('Invalid model dimensions')
    vector(model['prior'],n,True);vector(model['terminal_reward'],n)
    for row in model['initial_observation']:vector(row,k,True)
    if len(model['initial_observation'])!=n:raise ValueError('Initial observation shape')
    for a in ACTIONS:
        if len(model['transition'][a])!=n or len(model['observation'][a])!=n:raise ValueError('Action model shape')
        for row in model['transition'][a]:vector(row,n,True)
        for row in model['observation'][a]:vector(row,k,True)
        vector(model['cost_tokens'][a],n)

def initial_belief(model,obs):
    return normalize([p*o[obs] for p,o in zip(model['prior'],model['initial_observation'])])

def predict(model,belief,action):
    return [sum(belief[i]*model['transition'][action][i][j] for i in range(len(belief))) for j in range(len(belief))]

def update(model,belief,action,obs):
    pred=predict(model,belief,action)
    return normalize([p*o[obs] for p,o in zip(pred,model['observation'][action])])

def plan(model,belief,remaining,penalty):
    """Exact observation-contingent planning; arbitrary number of hidden states."""
    if remaining<0 or not math.isfinite(penalty) or penalty<0:raise ValueError('Invalid horizon/penalty')
    if len(belief)!=len(model['states']) or any(b<0 or not math.isfinite(b) for b in belief) or abs(sum(belief)-1)>1e-9:raise ValueError('Invalid belief')
    values={'stop':dot(belief,model['terminal_reward'])}
    if remaining:
        for action in ACTIONS:
            pred=predict(model,belief,action);value=-penalty*dot(belief,model['cost_tokens'][action])/1000
            for obs in range(len(model['observations'])):
                weights=[p*o[obs] for p,o in zip(pred,model['observation'][action])];prob=sum(weights)
                if prob>0:
                    posterior=[w/prob for w in weights]
                    value+=prob*plan(model,posterior,remaining-1,penalty)['value']
            values[action]=value
    best='stop'
    for action in ACTIONS:
        if action in values and values[action]>values[best]+CONFIG['tie_tolerance']:best=action
    return {'action':best,'value':values[best],'action_values':values}

def fit_model(rows):
    if len(rows)!=100 or any(r['split']!='train' for r in rows):raise ValueError('Fit requires exactly published training split')
    k=len(OBSERVATIONS);root_counts=[0,0];root_obs=[[0]*k for _ in range(2)]
    transitions={a:[[0,0],[0,0]] for a in ACTIONS};obs_counts={a:[[0]*k for _ in range(2)] for a in ACTIONS}
    costs={a:[[],[]] for a in ACTIONS}
    for row in rows:
        root=row['tree'][''];s=int(root['correct']);root_counts[s]+=1;root_obs[s][observation(root)]+=1
        for parent in ('','repair','retry'):
            s=int(row['tree'][parent]['correct'])
            for a in ACTIONS:
                child=row['tree'][parent+('/' if parent else '')+a];sp=int(child['correct'])
                transitions[a][s][sp]+=1;obs_counts[a][sp][observation(child)]+=1
                costs[a][s].append(child['input_tokens']+child['output_tokens'])
    alpha=CONFIG['smoothing']
    model={'states':CONFIG['states'],'observations':OBSERVATIONS,'terminal_reward':[0,1],
      'prior':smooth(root_counts,alpha),'initial_observation':[smooth(v,alpha) for v in root_obs],
      'transition':{a:[smooth(v,alpha) for v in transitions[a]] for a in ACTIONS},
      'observation':{a:[smooth(v,alpha) for v in obs_counts[a]] for a in ACTIONS},
      'cost_tokens':{a:[statistics.mean(v) if v else statistics.mean(costs[a][0]+costs[a][1]) for v in costs[a]] for a in ACTIONS},
      'training_counts':{'roots':root_counts,'root_observations':root_obs,'transitions':transitions,'successor_observations':obs_counts,'cost_samples':{a:[len(v) for v in costs[a]] for a in ACTIONS}}}
    validate_model(model);return model

def replay(model,rows,penalty):
    records=[]
    for row in rows:
        path='';visited=[''];belief=initial_belief(model,observation(row['tree']['']));steps=[];started=time.perf_counter()
        for remaining in (2,1):
            decision=plan(model,belief,remaining,penalty)
            steps.append({'path':path,'observation':model['observations'][observation(row['tree'][path])],
                          'belief':belief,'remaining':remaining,**decision})
            if decision['action']=='stop':break
            action=decision['action'];path+=('/' if path else '')+action;visited.append(path)
            belief=update(model,belief,action,observation(row['tree'][path]))
        elapsed=time.perf_counter()-started;nodes=[row['tree'][p] for p in visited]
        records.append({'id':row['id'],'domain':row['domain'],'steps':steps,'path':path,'final_belief':belief,
          'correct':int(nodes[-1]['correct']),'initial_correct':int(nodes[0]['correct']),
          'solver_tokens':sum(n['input_tokens']+n['output_tokens'] for n in nodes),'controller_tokens':0,
          'controller_cpu_seconds':elapsed,'forced_submit':steps[-1]['action']!='stop'})
    return records

def fixed_records(rows,paths):
    return [{'id':r['id'],'domain':r['domain'],'correct':int(r['tree'][paths[-1]]['correct']),
             'tokens':sum(r['tree'][p]['input_tokens']+r['tree'][p]['output_tokens'] for p in paths)} for r in rows]

def summarize(rows,records,penalty,split):
    methods={'pomdp_two_state':[{'id':r['id'],'domain':r['domain'],'correct':r['correct'],'tokens':r['solver_tokens']} for r in records]}
    methods.update({name:fixed_records(rows,paths) for name,paths in io.FIXED.items()})
    if penalty==CONFIG['primary_lambda']:
        a=io.read_json(A_RESULTS/split/'controller-results.json')
        if [r['id'] for r in a]!=[r['id'] for r in rows]:raise ValueError('Controller A split/ID mismatch')
        methods['llm_controller_A']=a
    summaries=[];paired=[]
    for name,rr in methods.items():
        for domain in ('all','math','science'):
            selected=[r for r in rr if domain=='all' or r['domain']==domain]
            scores=[r['correct'] for r in selected];utilities=[r['correct']-penalty*r['tokens']/1000 for r in selected]
            s={'policy':name,'domain':domain,'n':len(selected),'accuracy':statistics.mean(scores),'mean_tokens':statistics.mean(r['tokens'] for r in selected),'mean_utility':statistics.mean(utilities)}
            if penalty==CONFIG['primary_lambda']:s['accuracy_ci95']=io.interval(scores)
            summaries.append(s)
        if name!='pomdp_two_state' and penalty==CONFIG['primary_lambda']:
            diffs=[a['correct']-b['correct']-penalty*(a['tokens']-b['tokens'])/1000 for a,b in zip(methods['pomdp_two_state'],rr)]
            paired.append({'comparator':name,'mean_utility_difference':statistics.mean(diffs),'ci95':io.interval(diffs)})
    return {'split':split,'penalty':penalty,'summaries':summaries,'paired_utility':paired,
      'controller_cpu_seconds':sum(r['controller_cpu_seconds'] for r in records),'action_sequences':dict(Counter(' -> '.join(s['action'] for s in r['steps']) for r in records))}

def a_digest():
    manifest=io.read_json(A_RESULTS/'artifact-sha256.json')
    for f,h in manifest.items():
        if io.sha(A_RESULTS/f)!=h:raise ValueError('Controller A archive changed')
    return io.sha(A_RESULTS/'artifact-sha256.json')

def evidence():
    return {'module_sha256':io.sha(__file__),'helper_module_sha256':io.sha(io.__file__),'design_sha256':io.sha(DESIGN),'dataset_manifest_sha256':io.sha(io.DEFAULT_DATA/'FILE_MANIFEST.json'),
            'controller_A_archive_sha256':a_digest(),'config':CONFIG}

def check_run(run):
    if io.read_json(run/'manifest.json')['evidence']!=evidence():raise ValueError('Frozen experiment inputs changed; use a new run')
    model=io.read_json(run/'model.json');validate_model(model)
    if io.sha(run/'model.json')!=io.read_json(run/'manifest.json')['model_sha256']:raise ValueError('Model changed')
    if model!=fit_model(io.load_dataset(io.DEFAULT_DATA,'train')):raise ValueError('Model does not match training-only fit')
    return model

def phase(run,split):
    model=check_run(run);directory=run/split
    if directory.exists():raise ValueError('Refuse overwriting an evaluated split')
    if split=='test':check_freeze(run)
    rows=io.load_dataset(io.DEFAULT_DATA,split);directory.mkdir();started_utc=io.now()
    for penalty in CONFIG['lambda_grid']:
        records=replay(model,rows,penalty);key=str(penalty)
        io.save(directory/f'paths-{key}.json',records)
        io.save(directory/f'summary-{key}.json',summarize(rows,records,penalty,split))
    io.save(directory/'status.json',{'status':'complete','split':split,'questions':len(rows),'started_utc':started_utc,'finished_utc':io.now()})

def file_hashes(directory):return {p.name:io.sha(p) for p in sorted(directory.iterdir()) if p.is_file()}

def validate_phase(run,split):
    model=check_run(run);rows=io.load_dataset(io.DEFAULT_DATA,split);directory=run/split
    expected={'status.json'}|{f'{kind}-{str(lam)}.json' for lam in CONFIG['lambda_grid'] for kind in ('paths','summary')}
    if set(file_hashes(directory))!=expected:raise ValueError('Incomplete or unexpected split files')
    status=io.read_json(directory/'status.json')
    if status['status']!='complete' or status['split']!=split or status['questions']!=len(rows):raise ValueError('Invalid split status')
    for penalty in CONFIG['lambda_grid']:
        saved=io.read_json(directory/f'paths-{str(penalty)}.json');recomputed=replay(model,rows,penalty)
        for record in saved:
            if not math.isfinite(record['controller_cpu_seconds']) or record['controller_cpu_seconds']<0:raise ValueError('Invalid CPU timing')
        def untimed(records):return [{k:v for k,v in r.items() if k!='controller_cpu_seconds'} for r in records]
        if untimed(saved)!=untimed(recomputed):raise ValueError('Paths/beliefs/scores differ from frozen model replay')
        summary=io.read_json(directory/f'summary-{str(penalty)}.json')
        if summary!=summarize(rows,saved,penalty,split):raise ValueError('Summary differs from verified paths')
    return status

def check_freeze(run):
    f=io.read_json(run/'frozen.json')
    if f['manifest_sha256']!=io.sha(run/'manifest.json') or f['dev_hashes']!=file_hashes(run/'dev'):raise ValueError('Development evidence changed after freeze')
    status=validate_phase(run,'dev')
    if status['questions']!=60 or status['finished_utc']!=f['dev_finished_utc']:raise ValueError('Incomplete dev evidence')
    return f

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['fit','dev','freeze','test']);p.add_argument('--run',required=True);args=p.parse_args();run=Path(args.run)
    if args.command=='fit':
        if run.exists():raise ValueError('Refuse existing run directory')
        rows=io.load_dataset(io.DEFAULT_DATA,'train');model=fit_model(rows);run.mkdir(parents=True)
        io.save(run/'model.json',model);io.save(run/'manifest.json',{'created_utc':io.now(),'evidence':evidence(),'model_sha256':io.sha(run/'model.json'),'training_ids':[r['id'] for r in rows]})
    elif args.command=='freeze':
        check_run(run)
        if (run/'frozen.json').exists():raise ValueError('Refuse overwriting freeze')
        status=validate_phase(run,'dev')
        if status['status']!='complete' or status['questions']!=60:raise ValueError('Full dev required')
        io.save(run/'frozen.json',{'frozen_utc':io.now(),'manifest_sha256':io.sha(run/'manifest.json'),'dev_hashes':file_hashes(run/'dev'),'dev_finished_utc':status['finished_utc'],'note':'No dev-selected parameters; primary lambda and sweep predeclared.'})
    else:phase(run,'dev' if args.command=='dev' else 'test')
    print(json.dumps({'status':'complete','command':args.command,'run':str(run)}))

if __name__=='__main__':main()
