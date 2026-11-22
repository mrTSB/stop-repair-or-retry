"""Register/collect all512 auxiliary TRAIN questions×10 fixed solver paths.

No correctness scoring, policy fitting, neural inference or reviewer call is
implemented. Every attempted physical request is durable before dispatch;
an unresolved attempt blocks automatic resume instead of silently repeating.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import math
from pathlib import Path
import sys
import threading

import auxiliary_paired_contract as contract
import completion_v4_diagnostic as guarded
import expanded_mentor_io as io
from prospective_parallel import RequestJournal,KeyLocks,durable_line

complete=contract.complete
pilot=complete.pilot
old=pilot.protocol.old
WORKERS=4
MAX_CALLS=5120
BACKEND='0.40.16'
SCHEMA=contract.SCHEMA


def paid_tree(path,tree):
    if path in complete.PATHS:
        prefix=path.removesuffix('/complete') if '/' in path else ''
        return {p:tree[p] for p in ('',prefix)}
    if path not in contract.PLAIN:raise ValueError('Unexpected auxiliary solver path')
    if not path or path.endswith('retry'):return {}
    parent=path.rsplit('/',1)[0] if '/' in path else ''
    return {p:tree[p] for p in ('',parent)}


def payload(row,path,tree):
    if row.get('answer')!='' or row['domain']!='science':raise ValueError('Require reference-blank public science row')
    paid=paid_tree(path,tree)
    if path in complete.PATHS:
        prefix=path.removesuffix('/complete') if '/' in path else ''
        return pilot.request_payload(row,prefix,paid)[0]
    prompt=old.solver_prompt(row,path,paid,{})
    seed=int(old.hashlib.sha256(f'{old.SEED}:{row["id"]}:{path}'.encode()).hexdigest()[:8],16)%2**31
    return {'model':old.MODEL,'prompt':prompt,'raw':True,'stream':False,'logprobs':True,'keep_alive':'30m',
        'options':{'temperature':0 if not path else .65,'seed':seed,'num_predict':128 if not path else 256,'num_ctx':4096}}


def logical(row,path,tree,raw,registration):
    expected=payload(row,path,tree);RequestJournal.validate(raw)
    if raw['payload']!=expected:raise ValueError('Physical request differs from exact fixed paid prefix')
    value=raw['response'];cap=expected['options']['num_predict'];elapsed=raw['returned_unix']-raw['started_unix']
    if (value.get('model')!=old.MODEL or value.get('done') is not True
        or value.get('srr_backend',{}).get('ollama_version')!=BACKEND
        or type(value.get('prompt_eval_count')) is not int or not 0<value['prompt_eval_count']<=4096
        or type(value.get('eval_count')) is not int or not 0<=value['eval_count']<=cap
        or not isinstance(value.get('response'),str) or not math.isfinite(elapsed) or elapsed<0):
        raise ValueError('Invalid returned compact generation; retain it without regeneration')
    if path in complete.PATHS:
        prefix=path.removesuffix('/complete') if '/' in path else ''
        node=complete.logical_record(row,prefix,paid_tree(path,tree),raw,False)['response']
    else:
        text=value['response']
        if text and not value.get('logprobs'):raise ValueError('Returned compact text lacks token receipts; preserve without regeneration')
        node={'model':old.MODEL,'model_digest':old.MODEL_DIGEST,'text':text,
            'prompt_sha256':old.hashlib.sha256(expected['prompt'].encode()).hexdigest(),
            'seed':expected['options']['seed'],'temperature':expected['options']['temperature'],
            'output_cap':cap,'context_length':4096,'finish':value.get('done_reason'),
            'input_tokens':value['prompt_eval_count'],'output_tokens':value['eval_count'],
            'latency_seconds':elapsed,'backend':value['srr_backend'],
            'answer':old.parse_answer('science',text),'confidence':old.confidence(text,value.get('logprobs',[]),'science')}
    return {'id':row['id'],'subject':row['subject'],'kind':'solver','path':path,'protocol':SCHEMA,
        'registration_sha256':registration,'request_sha256':raw['request_sha256'],
        'raw_response_sha256':raw['response_sha256'],'paid_ancestry_sha256':contract.fingerprint(paid_tree(path,tree)),
        'response':node}


def attempts_checked(records,raw):
    attempts={}
    for record in records:
        key=record['request_sha256']
        if (key in attempts or record['schema']!=SCHEMA or record['endpoint']!='/api/generate'
            or record['payload_sha256']!=contract.fingerprint(record['payload'])
            or key!=contract.fingerprint({'endpoint':record['endpoint'],'payload':record['payload']})
            or record['payload']['model']!=old.MODEL):raise ValueError('Changed/duplicate attempted physical request')
        attempts[key]=record
    raw_by_key={}
    for record in raw:
        RequestJournal.validate(record);key=record['request_sha256']
        if key in raw_by_key or key not in attempts or record['payload']!=attempts[key]['payload']:
            raise ValueError('Raw physical response lacks its unique durable attempt')
        raw_by_key[key]=record
    if len(attempts)>MAX_CALLS:raise ValueError('Physical attempt budget exceeded')
    if set(attempts)!=set(raw_by_key):raise ValueError('Unresolved physical attempt; inspect before any automatic resume')
    return attempts


class Journal(RequestJournal):
    def __init__(self,request,stream,attempt_stream,records=(),attempts=()):
        self.attempts=attempts_checked(attempts,records);self.attempt_stream=attempt_stream;self.attempt_lock=threading.Lock()
        def once(endpoint,payload=None):
            if endpoint=='/api/generate':
                key=contract.fingerprint({'endpoint':endpoint,'payload':payload})
                with self.attempt_lock:
                    if key in self.attempts:raise ValueError('Attempted request cannot be dispatched again')
                    if len(self.attempts)>=MAX_CALLS:raise ValueError('Maximum5120 physical requests reached')
                    record={'schema':SCHEMA,'endpoint':endpoint,'payload':payload,'request_sha256':key,
                        'payload_sha256':contract.fingerprint(payload)}
                    durable_line(self.attempt_stream,record);self.attempts[key]=record
            return request(endpoint,payload)
        super().__init__(once,stream,records,slots=WORKERS)


def restore(rows,records,raw,attempts,registration,require_complete=False):
    attempts_checked(attempts,raw);lookup={r['request_sha256']:r for r in raw}
    saved={};known={};used=set()
    for item in records:
        key=item['id'],item['path']
        if key in saved:raise ValueError('Duplicate durable logical node')
        saved[key]=item
    for row in rows:
        tree={};missing_logical=False
        for path in contract.PATHS:
            request=payload(row,path,tree);key=contract.fingerprint({'endpoint':'/api/generate','payload':request})
            if key not in lookup:break
            if missing_logical:raise ValueError('A later physical request preceded its durable parent logical receipt')
            item=logical(row,path,tree,lookup[key],registration);used.add(key);tree[path]=item['response']
            node_key=row['id'],path
            if node_key in saved:
                if saved[node_key]!=item:raise ValueError('Logical node differs from exact immutable raw replay')
                known[node_key]=item
            else:missing_logical=True
    if set(saved)!=set(known) or used!=set(lookup):raise ValueError('Unrecognized physical response or logical node outside paid prefix')
    expected=len(rows)*10
    coverage={'questions':len(rows),'expected_cells':expected,'logical_actions':len(known),'physical_calls':len(raw),
        'physical_attempts':len(attempts),'review_calls':0,'neural_calls':0}
    if require_complete and (len(known)!=expected or len(raw)!=expected):raise ValueError('All question/path cells must complete before any later fitting')
    return known,coverage


def specification(args):
    rows,chosen,evidence=contract.public_inputs(args)
    record={'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{n:str(getattr(args,n).resolve()) for n in ('auxiliary','cache','data','observer','legacy_examples','output','priority_completion','blind_first_completion')},
        'selection':chosen,'selection_seed':contract.SEED,'selection_rule':'Fixed question hash within observed subject,256each from registered1440TRAIN;160cal never eligible',
        'public_rows_sha256':contract.fingerprint(rows),'questions':512,'paths_per_question':list(contract.PATHS),
        'fixed_solver':{'model':old.MODEL,'digest':old.MODEL_DIGEST,'backend':BACKEND,'root_cap':128,'root_temperature':0,
            'plain_continuation_cap':256,'plain_temperature':.65,'Complete_cap':1,'Complete_seed_temperature':'Exactly its paid parent original node',
            'seed_rule':f'SHA256({old.SEED}:question_id:original_plain_path) first8hex modulo2**31','context_length':4096},
        'maximum_physical_calls':MAX_CALLS,'workers':WORKERS,'max_cloud_gpus':2,'app':pilot.protocol.APP,
        'raw_response_reuse':0,'reference_fields_decoded':False,'new_neural_calls':0,'review_calls':0,
        'training_only':True,'protected_split_records_read':False,'correctness_filtering':False,'new_reservation':False,
        'priority_completions':{'strong_late':contract.priority_receipt(args.priority_completion),
            'blind_first':contract.priority_receipt(args.blind_first_completion,True)},
        'resume_rule':'Every attempt precedes dispatch; every raw return precedes logical parse. Unresolved attempts fail closed; no quality resampling.',
        'future_work':'Collect all5120 cells before labels/fit selection. Combined768-question fit and1536 new encoder passes require separate authorization.'}
    return rows,record


def register(args):
    rows,record=specification(args);args.output.mkdir(parents=True,exist_ok=False)
    io.jsonl(args.output/'public-questions.jsonl',rows);io.write(args.output/'registration.json',record)
    print({'registered_questions':512,'maximum_new_calls':MAX_CALLS,'registration_sha256':io.sha(args.output/'registration.json'),'inference_started':False},flush=True)


def registered(args):
    rows,expected=specification(args)
    if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
        or io.read(args.output/'registration.json')!=expected
        or pilot.protocol.strict_records(args.output/'public-questions.jsonl')!=rows):raise ValueError('Changed explicit auxiliary paired registration')
    io.verify(expected['source_files']);io.verify(expected['input_evidence']['files']);return rows,expected


class Collector:
    def __init__(self,known,stream,journal,registration):
        self.known,self.stream,self.journal,self.registration=known,stream,journal,registration
        self.keys,self.lock,self.stopped=KeyLocks(),threading.Lock(),threading.Event();self.completed=0
    def one(self,row):
        try:
            with self.keys.get(row['id']):
                tree={}
                for path in contract.PATHS:
                    if self.stopped.is_set():return
                    key=row['id'],path
                    if key in self.known:tree[path]=self.known[key]['response'];continue
                    request=payload(row,path,tree);self.journal('/api/generate',request)
                    request_key=contract.fingerprint({'endpoint':'/api/generate','payload':request})
                    item=logical(row,path,tree,self.journal.known[request_key],self.registration)
                    with self.lock:durable_line(self.stream,item);self.known[key]=item
                    tree[path]=item['response']
                with self.lock:
                    self.completed+=1
                    if self.completed%16==0:print({'completed_questions':self.completed,'total':512,'logical_cells':len(self.known)},flush=True)
        except BaseException:self.stopped.set();raise


def collect(args):
    with ExitStack() as stack:
        lock=stack.enter_context((args.output/'collection.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'complete.json').exists():raise ValueError('Preserve completed collection; never regenerate')
        rows,record=registered(args)
        paths={n:args.output/n for n in ('train.jsonl','raw-generations.jsonl','attempts.jsonl')}
        streams={n:stack.enter_context(p.open('a')) for n,p in paths.items()}
        for stream in streams.values():fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=pilot.protocol.strict_records(paths['raw-generations.jsonl']);attempts=pilot.protocol.strict_records(paths['attempts.jsonl'])
        known,_=restore(rows,pilot.protocol.strict_records(paths['train.jsonl']),raw,attempts,args.registration_sha256)
        import modal
        worker=modal.Cls.from_name(pilot.protocol.APP,'Ollama')();worker.request.autoscaler.update(max_containers=2)
        errors=args.output/'transport-errors.jsonl'
        journal=Journal(pilot.protocol.resilient(worker,errors,attempts=1),streams['raw-generations.jsonl'],streams['attempts.jsonl'],raw,attempts)
        manifest=journal('/api/tags');tags={v['name']:v['digest'] for v in manifest['models']}
        if tags.get(old.MODEL)!=old.MODEL_DIGEST:raise ValueError('Frozen compact weight digest changed')
        io.append(args.output/'model-tags.jsonl',{'selected_model':old.MODEL,'selected_digest':old.MODEL_DIGEST,
            'response':manifest,'response_sha256':contract.fingerprint(manifest),'live_checked':True})
        collector=Collector(known,streams['train.jsonl'],journal,args.registration_sha256)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:list(pool.map(collector.one,rows))
        known,coverage=restore(rows,pilot.protocol.strict_records(paths['train.jsonl']),pilot.protocol.strict_records(paths['raw-generations.jsonl']),
            pilot.protocol.strict_records(paths['attempts.jsonl']),args.registration_sha256,True)
        registered(args)
        names=['registration.json','public-questions.jsonl','model-tags.jsonl',*paths]
        if errors.exists():names.append(errors.name)
        tokens={kind:sum(item['response'][kind] for item in known.values()) for kind in ('input_tokens','output_tokens')}
        io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,'coverage':coverage,
            'digests':{n:io.sha(args.output/n) for n in names},'acquisition_tokens':tokens,'acquisition_compute':4*sum(tokens.values())/3.09,
            'transport_error_records':len(pilot.protocol.strict_records(errors)),'scored':False,'training_only':True,
            'test_used':False,'new_neural_calls':0,'review_calls':0})
        print({'complete':True,'coverage':coverage,'scored':False},flush=True)


def validated_complete(args):
    rows,record=registered(args);done=io.read(args.output/'complete.json')
    names={'registration.json','public-questions.jsonl','model-tags.jsonl','train.jsonl','raw-generations.jsonl','attempts.jsonl'}
    errors=args.output/'transport-errors.jsonl'
    if errors.exists():names.add(errors.name)
    if (done['schema']!=SCHEMA or done['registration_sha256']!=args.registration_sha256
        or set(done['digests'])!=names or done['scored'] or not done['training_only']
        or done['test_used'] or done['new_neural_calls'] or done['review_calls']):raise ValueError('Changed sealed auxiliary TRAIN grid')
    io.verify({str(args.output/n):v for n,v in done['digests'].items()})
    tags=pilot.protocol.strict_records(args.output/'model-tags.jsonl')
    if not tags or any(v['selected_model']!=old.MODEL or v['selected_digest']!=old.MODEL_DIGEST
        or v['response_sha256']!=contract.fingerprint(v['response']) or v['live_checked'] is not True
        or {t['name']:t['digest'] for t in v['response']['models']}.get(old.MODEL)!=old.MODEL_DIGEST for v in tags):
        raise ValueError('Changed durable live compact model manifest')
    known,coverage=restore(rows,pilot.protocol.strict_records(args.output/'train.jsonl'),
        pilot.protocol.strict_records(args.output/'raw-generations.jsonl'),pilot.protocol.strict_records(args.output/'attempts.jsonl'),
        args.registration_sha256,True)
    tokens={kind:sum(item['response'][kind] for item in known.values()) for kind in ('input_tokens','output_tokens')}
    if (done['coverage']!=coverage or done['acquisition_tokens']!=tokens or done['acquisition_compute']!=4*sum(tokens.values())/3.09
        or done['transport_error_records']!=len(pilot.protocol.strict_records(errors))):raise ValueError('Changed complete cost/coverage evidence')
    return rows,known,done


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','collect','preflight'))
    for name in ('auxiliary','cache','data','observer','legacy-examples','output','priority-completion','blind-first-completion'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');args=parser.parse_args()
    with guarded.training_only_reads():
        if args.mode=='preflight':
            rows,chosen,evidence=contract.public_inputs(args)
            print({'questions':len(rows),'subjects':contract.COUNTS,'public_rows_sha256':contract.fingerprint(rows),
                'selection_sha256':contract.fingerprint(chosen),'source_files':io.sources(sys.modules[__name__]),
                'input_evidence':evidence,'planned_calls':MAX_CALLS,'registered':False,'inference':False,
                'acquisition_priority_gates_checked':False},flush=True)
        else:{'register':register,'collect':collect}[args.mode](args)


if __name__=='__main__':main()
