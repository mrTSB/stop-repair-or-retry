"""Standalone LLM-controller replay on the published solver cache. No POMDP imports."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / 'results/cached-tree-dataset'
DEFAULT_PROMPT = ROOT / 'experiments/llm-controller/prompt.txt'
REWARD_DESIGN = ROOT / 'experiments/llm-controller/reward_design.json'
CONFIG_KEYS = {'endpoint','model','api_key_env','temperature','max_output_tokens','output_limit_parameter','response_format','penalty','timeout_seconds','max_requests','max_run_seconds','billing_mode','input_usd_per_million','output_usd_per_million','spend_limit_usd','request_reserve_usd','compute_budget_confirmed'}
ACTIONS = ('stop', 'repair', 'retry')
FIXED = {'stop':[''], 'repair_once':['','repair'], 'repair_twice':['','repair','repair/repair'], 'retry_once':['','retry'], 'retry_twice':['','retry','retry/retry']}

class ExperimentError(RuntimeError): pass

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(condition, message):
    if not condition: raise ExperimentError(message)
def numeric(value, positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
def read_json(path): return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def append(path,value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

@contextmanager
def lock(directory):
    path=Path(directory)/'.lock'
    try: fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError: raise ExperimentError('Run is locked. If a process crashed, verify it has stopped before removing .lock.') from None
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd);yield
    finally: path.unlink()

def load_config(path):
    c=read_json(path);require(set(c)==CONFIG_KEYS,'Config fields must match config.example.json exactly; never put keys in config.')
    require(isinstance(c['endpoint'],str) and isinstance(c['model'],str),'Endpoint and model must be strings.')
    require(isinstance(c['api_key_env'],str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',c['api_key_env']) is not None,'api_key_env must name an environment variable.')
    require(c['temperature'] is None or numeric(c['temperature']),'Invalid temperature.')
    for k in ('max_output_tokens','max_requests','max_run_seconds','timeout_seconds'):
        require(type(c[k]) is int and c[k]>0,f'{k} must be a positive integer.')
    require(c['max_output_tokens']<=4096,'Controller output limit must be at most 4096.')
    require(numeric(c['penalty']),'Invalid token penalty.')
    require(c['output_limit_parameter'] in ('max_tokens','max_completion_tokens'),'Unsupported output-limit field.')
    require(c['response_format'] in ('none','json_object','json_schema'),'Unsupported response format.')
    require(c['billing_mode'] in ('unconfigured','token','compute'),'Unsupported billing mode.')
    require(type(c['compute_budget_confirmed']) is bool,'compute_budget_confirmed must be boolean.')
    return c

def check_live(c, need_key=True):
    u=urllib.parse.urlsplit(c['endpoint'])
    require(u.scheme=='https' and u.hostname and not u.username and not u.password and not u.query and not u.fragment,'Use a full HTTPS chat-completions URL without credentials/query/fragment.')
    require(bool(c['model'].strip()),'Set an explicit controller model before a live run.')
    if c['billing_mode']=='token':
        for k in ('input_usd_per_million','output_usd_per_million'):
            require(numeric(c[k]),f'Set verified {k}; do not guess prices.')
        for k in ('spend_limit_usd','request_reserve_usd'):
            require(numeric(c[k],True),f'Set positive {k}.')
        require(c['request_reserve_usd']<=c['spend_limit_usd'],'Request reserve exceeds run budget.')
    elif c['billing_mode']=='compute':
        require(c['compute_budget_confirmed'],'Configure provider compute budget/resource limits, then explicitly confirm them.')
        require(all(c[k] is None for k in ('input_usd_per_million','output_usd_per_million','spend_limit_usd','request_reserve_usd')),'Compute billing cannot be represented by token-price estimates; leave token billing fields null.')
    else: raise ExperimentError('Select token or compute billing before live inference.')
    if need_key: require(bool(os.environ.get(c['api_key_env'])),'API-key environment variable is unset. No request sent.')

def load_dataset(directory,split):
    d=Path(directory).resolve();manifest=read_json(d/'FILE_MANIFEST.json')
    for name,entry in manifest['files'].items():
        require(sha(d/name)==entry['sha256'],f'Dataset manifest mismatch: {name}')
    require(split in ('train','dev','test'),'Unknown split.')
    with gzip.open(d/'data'/f'{split}-trees.jsonl.gz','rt',encoding='utf-8') as f: rows=[json.loads(x) for x in f]
    require(len({r['id'] for r in rows})==len(rows),'Duplicate question IDs.')
    require(all(r['split']==split and r['max_depth']==2 for r in rows),'Unexpected split or horizon.')
    return rows

def observable(row,path,penalty):
    require(path in ('','repair','retry'),'No controller call is allowed at a terminal node.')
    paths=['']+([path] if path else [])
    # Explicit allowlist: never serialize row, tree, gold, labels, or an unvisited node.
    return {'problem':row['question'],'remaining_solver_calls':2 if not path else 1,'lambda_per_1000_tokens':penalty,
            'visited_responses':[{'action':'initial' if not p else p,'response':row['tree'][p]['text'],
                                  'input_tokens':row['tree'][p]['input_tokens'],'output_tokens':row['tree'][p]['output_tokens']} for p in paths]}

def request_body(row,path,c,prompt):
    body={'model':c['model'],'messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(observable(row,path,c['penalty']),ensure_ascii=False)}],
          'stream':False,c['output_limit_parameter']:c['max_output_tokens']}
    if c['temperature'] is not None: body['temperature']=c['temperature']
    if c['response_format']=='json_object': body['response_format']={'type':'json_object'}
    elif c['response_format']=='json_schema': body['response_format']={'type':'json_schema','json_schema':{'name':'controller_action','strict':True,'schema':{'type':'object','properties':{'action':{'type':'string','enum':list(ACTIONS)}},'required':['action'],'additionalProperties':False}}}
    return body

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

class HTTPClient:
    def __init__(self,c): self.c=c
    def __call__(self,body):
        c=self.c
        req=urllib.request.Request(c['endpoint'],data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ[c['api_key_env']]})
        # No redirects, SDK retries, fallback models, or silent request-parameter removal.
        try:
            with urllib.request.build_opener(NoRedirect()).open(req,timeout=c['timeout_seconds']) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            error=ExperimentError(f'HTTP {e.code}; request remains unresolved. No automatic retry.')
            error.http_status=e.code
            raise error from None
        except Exception as e: raise ExperimentError(f'Transport failure ({type(e).__name__}); request remains unresolved. No automatic retry.') from None

class MockClient:
    def __call__(self,body):
        observation=json.loads(body['messages'][1]['content'])
        action='repair' if observation['remaining_solver_calls']==2 else 'stop'
        return {'id':'offline-mock','model':'offline-mock','choices':[{'message':{'content':json.dumps({'action':action})},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':100,'completion_tokens':8,'total_tokens':108}}

def decode(response,c):
    require(isinstance(response,dict),'Response must be a JSON object.')
    require(isinstance(response.get('model'),str) and bool(response['model']),'Response must identify the actual model.')
    usage=response.get('usage',{});require(isinstance(usage,dict),'Missing token usage.')
    inp,out=usage.get('prompt_tokens'),usage.get('completion_tokens')
    require(type(inp) is int and inp>0 and type(out) is int and out>=0,'Missing/invalid prompt_tokens or completion_tokens: cannot report zero cost.')
    if 'total_tokens' in usage: require(type(usage['total_tokens']) is int and usage['total_tokens']==inp+out,'Usage totals disagree; reconcile before evaluation.')
    # Reasoning/cached token detail fields are subsets of these totals, not added again.
    choices=response.get('choices');require(isinstance(choices,list) and len(choices)==1,'Expected exactly one choice.')
    require(isinstance(choices[0],dict),'Malformed choice.')
    message=choices[0].get('message');text=message.get('content') if isinstance(message,dict) else None;valid=False;action='stop'
    if isinstance(text,str):
        try:
            obj=json.loads(text);valid=isinstance(obj,dict) and set(obj)=={'action'} and obj['action'] in ACTIONS
            if valid: action=obj['action']
        except (json.JSONDecodeError,TypeError): pass
    cost=(inp*c['input_usd_per_million']+out*c['output_usd_per_million'])/1e6 if c['billing_mode']=='token' else None
    return {'action':action,'valid_action':valid,'text':text,'input_tokens':inp,'output_tokens':out,'estimated_usd':cost,
            'reported_model':response.get('model'),'response_id':response.get('id'),'finish_reason':choices[0].get('finish_reason')}

def events(directory):
    p=Path(directory)/'events.jsonl'
    if not p.exists(): return []
    try: return [json.loads(line) for line in p.read_text().splitlines()]
    except json.JSONDecodeError: raise ExperimentError('Journal has an incomplete record; preserve it and reconcile before resuming.') from None

def journal(directory,c):
    requests={};responses={};total_seconds=0
    for e in events(directory):
        if e['type']=='request':
            require(e['request_id'] not in requests,'Duplicate request ID.');requests[e['request_id']]=e
        elif e['type']=='response':
            rid=e['request_id'];require(rid in requests and rid not in responses,'Duplicate or orphan response.')
            parsed=decode(e['response'],c);responses[rid]={**parsed,'latency_seconds':e['latency_seconds']};total_seconds+=e['latency_seconds']
    pending=set(requests)-set(responses)
    return requests,responses,pending,total_seconds

def fingerprint(c,prompt,dataset,mode):
    return {'config':c,'prompt':prompt,'mode':mode,'module_sha256':sha(__file__),
            'dataset_manifest_sha256':sha(Path(dataset)/'FILE_MANIFEST.json'),'reward_design_sha256':sha(REWARD_DESIGN)}

def validated_run(directory):
    require((Path(directory)/'manifest.json').exists(),'Run manifest does not exist.')
    m=read_json(Path(directory)/'manifest.json');c=m['experiment']['config'];d=Path(m['dataset'])
    rows=load_dataset(d,m['split'])[:len(m['question_ids'])]
    require([r['id'] for r in rows]==m['question_ids'],'Selected questions changed.')
    require(fingerprint(c,m['experiment']['prompt'],d,m['experiment']['mode'])==m['experiment'],'Code/dataset fingerprint changed.')
    req,res,pending,_=journal(directory,c)
    if m['split']=='test':
        frozen=m.get('test_freeze');require(isinstance(frozen,dict),'Test run lacks frozen development evidence.')
        require(sha(frozen['path'])==frozen['sha256'],'Test freeze changed.')
        evidence=verify_freeze(frozen['path'],m['experiment'])
        require(all(v['reported_model'] in evidence['reported_models'] for v in res.values()),'Test response model differs from development freeze.')
    return m,rows,req,res,pending

def completed_records(directory):
    m,rows,req,res,pending=validated_run(directory);require(not pending,'Unresolved requests block evaluation.')
    c=m['experiment']['config'];records=[];used=set()
    for row in rows:
        path='';paths=[''];actions=[];calls=[]
        for step in range(2):
            rid=digest([row['id'],path]);require(rid in res,'Run is incomplete: missing controller response.')
            require(req[rid]['body']==request_body(row,path,c,m['experiment']['prompt']),'Request does not match visited history/config.')
            used.add(rid);v=res[rid];calls.append(v);actions.append(v['action'])
            if v['action']=='stop': break
            path=path+('/' if path else '')+v['action'];paths.append(path)
        nodes=[row['tree'][p] for p in paths];si=sum(n['input_tokens'] for n in nodes);so=sum(n['output_tokens'] for n in nodes)
        ci=sum(v['input_tokens'] for v in calls);co=sum(v['output_tokens'] for v in calls)
        records.append({'id':row['id'],'domain':row['domain'],'policy':'llm_controller','actions':actions,'path':path,'answer':nodes[-1]['answer'],
                        'correct':int(nodes[-1]['correct']),'initial_correct':int(nodes[0]['correct']),'solver_input_tokens':si,'solver_output_tokens':so,
                        'controller_input_tokens':ci,'controller_output_tokens':co,'tokens':si+so+ci+co,'controller_calls':len(calls),
                        'invalid_actions':sum(not v['valid_action'] for v in calls),'controller_latency_seconds':sum(v['latency_seconds'] for v in calls)})
    require(set(req)==used,'Journal has extra or unvisited requests.')
    return m,rows,records

def verify_freeze(path,experiment):
    f=read_json(path);require(f['experiment']==experiment and f['development_questions']==60 and f['mode']=='live','Freeze does not match this experiment or completed live development run.')
    require(f['fingerprint_sha256']==digest(experiment),'Invalid freeze fingerprint.')
    dev=Path(f['development_run'])
    require(sha(dev/'manifest.json')==f['dev_manifest_sha256'] and sha(dev/'events.jsonl')==f['dev_events_sha256'],'Development evidence changed after freeze.')
    m,rows,records=completed_records(dev)
    require(m['split']=='dev' and len(rows)==60 and m['experiment']==experiment,'Development evidence does not match freeze.')
    return f

def run(c,prompt,dataset,split,limit,output,mode,freeze=None,client=None):
    require(mode in ('live','mock'),'Explicit live or mock mode required.')
    all_rows=load_dataset(dataset,split);require(limit is None or type(limit) is int and 0<limit<=len(all_rows),'Invalid limit.')
    rows=all_rows[:limit] if limit else all_rows;exp=fingerprint(c,prompt,dataset,mode);frozen=None
    if split=='test':
        require(mode=='live' and freeze is not None,'Test requires a frozen, completed live development experiment.')
        frozen=verify_freeze(freeze,exp)
    if mode=='live': check_live(c,need_key=client is None)
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    with lock(directory):
        mp=directory/'manifest.json'
        wanted={'experiment':exp,'dataset':str(Path(dataset).resolve()),'split':split,'question_ids':[r['id'] for r in rows],'test_freeze':{'path':str(Path(freeze).resolve()),'sha256':sha(freeze)} if split=='test' else None}
        if mp.exists():
            old=read_json(mp);require(all(old[k]==v for k,v in wanted.items()),'Existing run differs in model/config/prompt/data/mode/split/selection. Use a new output directory.')
        else:
            require(not (directory/'events.jsonl').exists(),'Journal without manifest; refuse overwrite.')
            save(mp,{**wanted,'created_utc':now()})
        req,res,pending,spent_seconds=journal(directory,c)
        require(not pending,'Unresolved request found. Recover its provider response; automatic resubmission could double-charge.')
        if mode=='live' and c['billing_mode']=='token':
            require(all(v['estimated_usd']<=c['request_reserve_usd'] for v in res.values()),'A saved request exceeded its reserve; do not resume with this pricing/budget configuration.')
        backend=client or (MockClient() if mode=='mock' else HTTPClient(c));started=time.monotonic()
        for row in rows:
            path=''
            for step in range(2):
                rid=digest([row['id'],path]);body=request_body(row,path,c,prompt)
                if rid in res:
                    require(req[rid]['body']==body,'Saved request differs from current visited history.');decision=res[rid]
                else:
                    cost=sum(v['estimated_usd'] or 0 for v in res.values())
                    require(len(req)<c['max_requests'],'Request cap reached; no further paid calls sent.')
                    require(spent_seconds+time.monotonic()-started<c['max_run_seconds'],'Run-time cap reached.')
                    if mode=='live' and c['billing_mode']=='token':
                        require(cost+c['request_reserve_usd']<=c['spend_limit_usd'],'Estimated spend plus reserve reaches run limit; stopped before request.')
                    request={'type':'request','request_id':rid,'question_id':row['id'],'path':path,'body':body,'created_utc':now(),'reserved_usd':c['request_reserve_usd'] if mode=='live' and c['billing_mode']=='token' else None}
                    append(directory/'events.jsonl',request);req[rid]=request;t=time.monotonic()
                    try:
                        response=backend(body);elapsed=time.monotonic()-t
                        # Save raw success before parsing so response/usage failures remain auditable.
                        append(directory/'events.jsonl',{'type':'raw_response','request_id':rid,'response':response,'latency_seconds':elapsed,'created_utc':now()})
                        decision=decode(response,c)
                    except Exception as e:
                        append(directory/'events.jsonl',{'type':'failure','request_id':rid,'error_class':type(e).__name__,'http_status':getattr(e,'http_status',None),'created_utc':now()})
                        raise ExperimentError(f"Request {rid} unresolved ({type(e).__name__}, HTTP status {getattr(e,'http_status',None)}); no automatic retry. See journal and recovery instructions.") from None
                    append(directory/'events.jsonl',{'type':'response','request_id':rid,'response':response,'latency_seconds':elapsed,'created_utc':now()})
                    decision['latency_seconds']=elapsed;res[rid]=decision
                    if mode=='live' and c['billing_mode']=='token':
                        require(decision['estimated_usd']<=c['request_reserve_usd'],'Actual estimated charge exceeded reserve. Halted; inspect prices/provider billing before continuing.')
                if frozen is not None:
                    require(decision['reported_model'] in frozen['reported_models'],'Reported model differs from development freeze; halt before using this decision.')
                if decision['action']=='stop': break
                path=path+('/' if path else '')+decision['action']
        # Completeness is derived from the journal, never merely a status flag.
        completed_records(directory)
        save(directory/'status.json',{'status':'complete','mode':mode,'questions':len(rows),'controller_requests':len(req),'finished_utc':now()})
    return directory

def recover(directory,rid,response_path,latency):
    require(numeric(latency),'Provide measured nonnegative latency.')
    with lock(directory):
        m,rows,req,res,pending=validated_run(directory);require(rid in pending,'Request is not unresolved.')
        response=read_json(response_path);decode(response,m['experiment']['config'])
        append(Path(directory)/'events.jsonl',{'type':'response','request_id':rid,'response':response,'latency_seconds':latency,'created_utc':now(),'recovered_from_provider_receipt':True})

def freeze_run(directory,output):
    m,rows,records=completed_records(directory)
    require(m['split']=='dev' and len(rows)==60 and m['experiment']['mode']=='live','Freeze requires all 60 development questions completed with live inference.')
    check_live(m['experiment']['config'],need_key=False);target=Path(output);require(not target.exists(),'Freeze already exists; refuse overwrite.')
    target.parent.mkdir(parents=True,exist_ok=True)
    save(target,{'experiment':m['experiment'],'fingerprint_sha256':digest(m['experiment']),'development_questions':60,'mode':'live','development_run':str(Path(directory).resolve()),'reported_models':sorted({str(v['reported_model']) for v in journal(directory,m['experiment']['config'])[1].values()}),'dev_manifest_sha256':sha(Path(directory)/'manifest.json'),'dev_events_sha256':sha(Path(directory)/'events.jsonl'),'frozen_utc':now(),'test_status':'Existing baseline test outcomes have been viewed; this is not a new unseen confirmation set.'})

def interval(values):
    rng=random.Random(20261110);n=len(values)
    samples=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(2000))
    return [samples[50],samples[1949]]

def evaluate(directory):
    m,rows,details=completed_records(directory);penalty=m['experiment']['config']['penalty'];all_results={'llm_controller':details}
    for policy,paths in FIXED.items():
        rr=[]
        for row in rows:
            nodes=[row['tree'][p] for p in paths]
            rr.append({'id':row['id'],'domain':row['domain'],'policy':policy,'correct':int(nodes[-1]['correct']),'initial_correct':int(nodes[0]['correct']),
                       'tokens':sum(n['input_tokens']+n['output_tokens'] for n in nodes),'controller_calls':0,'invalid_actions':0,'controller_latency_seconds':0})
        all_results[policy]=rr
    summaries=[];paired=[]
    for domain in ('all','math','science'):
        for policy,rr in all_results.items():
            rr=[x for x in rr if domain=='all' or x['domain']==domain]
            if not rr: continue
            accuracy=[x['correct'] for x in rr];utility=[x['correct']-penalty*x['tokens']/1000 for x in rr]
            summaries.append({'domain':domain,'policy':policy,'n':len(rr),'accuracy':statistics.mean(accuracy),'accuracy_ci95':interval(accuracy),
                              'mean_tokens':statistics.mean(x['tokens'] for x in rr),'mean_utility':statistics.mean(utility),'mean_controller_calls':statistics.mean(x['controller_calls'] for x in rr)})
    for policy in FIXED:
        rr=all_results[policy];require([x['id'] for x in rr]==[x['id'] for x in details],'Paired IDs differ.')
        diffs=[a['correct']-b['correct']-penalty*(a['tokens']-b['tokens'])/1000 for a,b in zip(details,rr)]
        paired.append({'baseline':policy,'mean_utility_difference':statistics.mean(diffs),'ci95':interval(diffs)})
    req,res,pending,_=journal(directory,m['experiment']['config'])
    report={'mode':m['experiment']['mode'],'split':m['split'],'n':len(rows),'penalty':penalty,'summaries':summaries,'paired_utility_differences':paired,
            'invalid_actions':sum(v['invalid_actions'] for v in details),'reported_models':sorted({str(v['reported_model']) for v in res.values()}),
            'estimated_controller_usd':sum(v['estimated_usd'] for v in res.values()) if m['experiment']['config']['billing_mode']=='token' and m['experiment']['mode']=='live' else None,
            'latency_note':'Only actual controller request latency is measured. Cached solver batch times are not serial serving latency.',
            'cost_note':'Token-price estimates are not invoices. Compute/idle/startup charges require the provider ledger. Budgets are per run, not account-wide.',
            'test_note':'Existing test baseline outcomes have been inspected; use fresh data if those outcomes informed redesign.'}
    d=Path(directory);save(d/'summary.json',report);save(d/'controller-results.json',details)
    title='OFFLINE MOCK SMOKE TEST — NOT LLM PERFORMANCE' if report['mode']=='mock' else 'LLM controller and five fixed baselines'
    lines=['# '+title,'',f"Split: {report['split']}; questions: {len(rows)}; token penalty: {penalty}.",'', '| Method | Accuracy | Mean total tokens | Mean utility |', '|---|---:|---:|---:|']
    for s in summaries:
        if s['domain']=='all':lines.append(f"| {s['policy']} | {100*s['accuracy']:.2f}% | {s['mean_tokens']:.1f} | {s['mean_utility']:.4f} |")
    lines+=['','The controller pays for every executed input/output token, including a decision to stop. Baselines use only the same visited solver cache. No fitted controller or POMDP is loaded. Invalid action outputs submit the current answer and remain counted.','','Intervals are paired question bootstrap estimates conditional on one fixed cache and controller run. Development results are exploratory; comparisons against five baselines are not adjusted for multiple testing. This report does not establish a confirmatory win.','',report['latency_note'],report['cost_note'],report['test_note']]
    (d/'report.md').write_text('\n'.join(lines)+'\n');return report

def landscape(dataset,output):
    """Describe TRAINING outcomes/costs; no fitted transition model or planner."""
    rows=load_dataset(dataset,'train');design=read_json(REWARD_DESIGN);transitions=[]
    for action in ('repair','retry'):
        for state in (False,True):
            children=[]
            for row in rows:
                for parent in ('','repair','retry'):
                    if bool(row['tree'][parent]['correct'])==state:
                        children.append(row['tree'][parent+('/' if parent else '')+action])
            transitions.append({'action':action,'parent_correct':state,'edges':len(children),'next_correct':sum(n['correct'] for n in children),
                                'empirical_next_correct_rate':statistics.mean(n['correct'] for n in children) if children else None,
                                'mean_action_tokens':statistics.mean(n['input_tokens']+n['output_tokens'] for n in children) if children else None})
    observations=[]
    for confidence in ('low','medium','high','missing'):
        nodes=[n for row in rows for n in row['tree'].values() if n['confidence']==confidence]
        observations.append({'confidence':confidence,'nodes':len(nodes),'correct':sum(n['correct'] for n in nodes)})
    utilities=[]
    for lam in design['candidate_lambda_grid']:
        for policy,paths in FIXED.items():
            vals=[(int(row['tree'][paths[-1]]['correct']),sum(row['tree'][p]['input_tokens']+row['tree'][p]['output_tokens'] for p in paths)) for row in rows]
            utilities.append({'lambda':lam,'baseline':policy,'accuracy':statistics.mean(v[0] for v in vals),'mean_tokens':statistics.mean(v[1] for v in vals),'mean_utility':statistics.mean(v[0]-lam*v[1]/1000 for v in vals)})
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    report={'split':'train','questions':len(rows),'scope':'Descriptive training-only action/observation counts and fixed-baseline reward tradeoffs; no fitted POMDP or controller.','reward_design':design,'reward_design_sha256':sha(REWARD_DESIGN),'training_cache_sha256':sha(Path(dataset)/'data/train-trees.jsonl.gz'),'transitions':transitions,'confidence_counts':observations,'fixed_baseline_reward_grid':utilities}
    save(out/'landscape.json',report)
    lines=['# Training action and reward landscape','','Training questions only. Counts are descriptive and correlated within question; no POMDP is fitted. The reward is a design choice; trajectories describe the action dynamics to be learned later.','', '| Action | Current correct | Edges | Next correct | Mean action tokens |','|---|---|---:|---:|---:|']
    for v in transitions:lines.append(f"| {v['action']} | {v['parent_correct']} | {v['edges']} | {v['next_correct']} | {v['mean_action_tokens'] or 0:.1f} |")
    lines+=['','The JSON includes confidence counts and a five-baseline reward grid. Do not choose lambda by maximizing utility across lambdas: reducing the penalty mechanically raises utility. Declare a cost preference or report the tradeoff. A future POMDP would learn transitions/observations and expected costs from training, then be evaluated separately.']
    (out/'landscape.md').write_text('\n'.join(lines)+'\n');return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preview','run'):
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--prompt',default=str(DEFAULT_PROMPT));p.add_argument('--dataset',default=str(DEFAULT_DATA));p.add_argument('--split',choices=('train','dev','test'),default='dev');p.add_argument('--limit',type=int)
        if name=='run':
            p.add_argument('--output',required=True);group=p.add_mutually_exclusive_group(required=True);group.add_argument('--live',action='store_true');group.add_argument('--mock',action='store_true');p.add_argument('--freeze')
    p=sub.add_parser('landscape');p.add_argument('--dataset',default=str(DEFAULT_DATA));p.add_argument('--output',required=True)
    p=sub.add_parser('evaluate');p.add_argument('--run',required=True)
    p=sub.add_parser('freeze');p.add_argument('--dev-run',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('recover-response');p.add_argument('--run',required=True);p.add_argument('--request-id',required=True);p.add_argument('--response-json',required=True);p.add_argument('--latency-seconds',required=True,type=float)
    a=parser.parse_args()
    try:
        if a.command in ('preview','run'):
            c=load_config(a.config);prompt=Path(a.prompt).read_text()
            if a.command=='preview':
                require(a.split!='test','Preview development first; test is behind the freeze gate.');rows=load_dataset(a.dataset,a.split)
                limit=a.limit if a.limit is not None else 1;require(0<limit<=len(rows),'Invalid preview limit.')
                print(json.dumps({'network_calls':0,'selected_questions':limit,'maximum_controller_calls':2*limit,'requests':[request_body(r,'',c,prompt) for r in rows[:limit]]},indent=2))
            else: print(run(c,prompt,a.dataset,a.split,a.limit,a.output,'live' if a.live else 'mock',a.freeze))
        elif a.command=='landscape': print(json.dumps(landscape(a.dataset,a.output),indent=2))
        elif a.command=='evaluate': print(json.dumps(evaluate(a.run),indent=2))
        elif a.command=='freeze': freeze_run(a.dev_run,a.output);print(a.output)
        else: recover(a.run,a.request_id,a.response_json,a.latency_seconds)
    except (ExperimentError,FileNotFoundError,json.JSONDecodeError) as e: parser.exit(2,str(e)+'\n')
if __name__=='__main__': main()
