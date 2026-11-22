"""Bounded new128-call compact-repair headroom pilot; no new LLM review."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
from pathlib import Path
import sys
import threading

import history_fusion_protocol as spec
import completion_v4_diagnostic as guarded
import grounded_root_review_pilot as infrastructure
from prospective_parallel import KeyLocks

io,old,protocol = spec.io,spec.old,spec.protocol
NAMES = ('History fusion after guided','History fusion after retry')
SCHEMA = 'paid-history-fusion-compact-only-runner-v1'


def require_priority(args):
    folder = args.data.parent/'dense-grounded1024-root-review-pilot-v1'
    done = io.read(folder/'complete.json')
    io.verify({str(folder/k):v for k,v in done['digests'].items()})
    if protocol.strict_records(folder/'capacity.jsonl')[-1] != {'min_containers':0,'max_containers':2}:
        raise ValueError('The prior cloud collector must release capacity')


def registration(args):
    require_priority(args)
    rows,trees,reviews,ordinary,ordinary_reviews,late,source,evidence = spec.public_inputs(args)
    return {'schema':spec.SCHEMA,'runner_schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),
        'input_evidence':evidence,'paths':{k:str(getattr(args,k).resolve()) for k in ('data','mentor','complete','late','output')},
        'selection':source['selection'],'questions':64,'subjects':spec.COUNTS,
        'new_solver_calls':128,'new_reviewer_calls':0,'new_neural_calls':0,'workers':4,'maximum_cloud_gpus':2,
        'solver':protocol.old.MODEL,'solver_digest':protocol.old.MODEL_DIGEST,'reviewer_digest':protocol.old.JUDGE_DIGEST,
        'solver_output_cap':128,'solver_context_length':4096,'solver_temperature':.65,
        'first_continuations':list(spec.PREFIXES),'new_paths':spec.PATHS,
        'same_compact_weights_and_plain_baselines':True,'new_repair_protocol_explicit':True,
        'paid_reasoning_excerpt_rule':'Last160 whitespace-separated words of the entire paid first reviewer phase',
        'initial_root_and_first_continuations_reused_exactly':True,
        'archived_reviewer_phases_reused_as_paid_logical_prefix':True,
        'no_fabricated_raw_reviewer_receipts':True,
        'inference_trajectory':'Root compact answer, archived root reviewer with every physical phase billed, exactly one existing first continuation, one new terminal compact repair',
        'maximum_answer_changing_continuations':2,'logical_reviews_per_trajectory':1,
        'all_prefix_and_new_input_output_tokens_billed':True,'screen':spec.SCREEN,'known_prior_union':50,
        'requests':[{'id':r['id'],'parent':p,'payload_sha256':old.digest(spec.payload(r,trees[r['id']],reviews[r['id'],''],p)),
                     'paid_prefix':spec.paid_prefix(trees[r['id']],reviews[r['id'],''],p)} for r in rows for p in spec.PREFIXES],
        'training_only':True,'test_used':False,'development_values_read':False,'references_supplied':False,
        'no_quality_regeneration':True,'transport_attempts':1,
        'promotion_requires':'Same new repair context for strongest matched B; learned POMDP dynamics for the new protocol; fresh frozen evaluation'}


def attempts_checked(records,raw):
    attempts = {}
    for record in records:
        key = record['request_sha256']
        if (key in attempts or record['schema'] != spec.SCHEMA or record['endpoint'] != '/api/generate'
                or record['payload']['model'] != protocol.old.MODEL
                or record['payload_sha256'] != old.digest(record['payload'])
                or key != old.digest({'endpoint':'/api/generate','payload':record['payload']})):
            raise ValueError('Repeated, foreign or changed durable compact attempt')
        attempts[key] = record
    completed = {}
    for record in raw:
        old.RequestJournal.validate(record)
        key = record['request_sha256']
        if key in completed or key not in attempts or record['payload'] != attempts[key]['payload']:
            raise ValueError('Physical response lacks a unique durable attempted request')
        completed[key] = record
    if len(attempts) > 128 or set(attempts) != set(completed):
        raise ValueError('Unresolved attempted call or exceeded128-call budget; no automatic retry')
    return attempts


class Journal(old.RequestJournal):
    def __init__(self,request,stream,attempt_stream,records=(),attempts=()):
        self.attempts = attempts_checked(attempts,records)
        lock = threading.Lock()
        def once(endpoint,payload=None):
            if endpoint == '/api/generate':
                if payload['model'] != protocol.old.MODEL: raise ValueError('No new reviewer inference authorized')
                key = old.digest({'endpoint':endpoint,'payload':payload})
                with lock:
                    if key in self.attempts or len(self.attempts) >= 128: raise ValueError('Repeated attempt or exhausted budget')
                    item = {'schema':spec.SCHEMA,'endpoint':endpoint,'payload':payload,
                            'request_sha256':key,'payload_sha256':old.digest(payload)}
                    infrastructure.paired.durable_line(attempt_stream,item)
                    self.attempts[key] = item
            return request(endpoint,payload)
        super().__init__(once,stream,records,slots=4)


class Collector:
    def __init__(self,trees,reviews,known,stream,journal):
        self.trees,self.reviews,self.known,self.stream,self.journal = trees,reviews,known,stream,journal
        self.keys = KeyLocks()
        self.lock,self.stopped = threading.Lock(),threading.Event()

    def one(self,pair):
        row,parent = pair
        qid = row['id']
        try:
            with self.keys.get((qid,parent)):
                if self.stopped.is_set() or (qid,parent) in self.known: return
                tree,review = self.trees[qid],self.reviews[qid,'']
                payload = spec.payload(row,tree,review,parent)
                self.journal('/api/generate',payload)
                key = old.digest({'endpoint':'/api/generate','payload':payload})
                record = spec.logical(row,tree,review,parent,self.journal.known[key])
                with self.lock:
                    infrastructure.paired.durable_line(self.stream,record)
                    self.known[qid,parent] = record
                    if len(self.known)%16 == 0: print({'fusion_endpoints':len(self.known),'total':128},flush=True)
        except BaseException:
            self.stopped.set()
            raise


def completion_payload(args,coverage):
    paths = ['registration.json','train.jsonl','raw-generations.jsonl','attempts.jsonl','capacity.jsonl']
    error = args.output/'transport-errors.jsonl'
    count = len(protocol.strict_records(error))
    if error.exists(): paths.append(error.name)
    attempts_checked(protocol.strict_records(args.output/'attempts.jsonl'),protocol.strict_records(args.output/'raw-generations.jsonl'))
    return {'schema':spec.SCHEMA,'runner_schema':SCHEMA,'coverage':coverage,
        'digests':{k:io.sha(args.output/k) for k in paths},'registration_sha256':io.sha(args.output/'registration.json'),
        'training_only':True,'test_used':False,'scored':False,'new_reviewer_calls':0,'new_neural_calls':0,
        'transport_error_records':count,'acquisition_compute_is_lower_bound':bool(count)}


def collect(args):
    with ExitStack() as stack:
        lock = stack.enter_context((args.output/'collection.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'complete.json').exists(): raise ValueError('Preserve completed responses')
        record = registration(args)
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered history protocol')
        rows,trees,reviews,*_ = spec.public_inputs(args)
        handles = {k:stack.enter_context((args.output/k).open('a')) for k in ('train.jsonl','raw-generations.jsonl','attempts.jsonl','capacity.jsonl')}
        for handle in handles.values(): fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw = protocol.strict_records(args.output/'raw-generations.jsonl')
        known,_ = spec.restore(rows,trees,reviews,protocol.strict_records(args.output/'train.jsonl'),raw)
        import modal
        worker = modal.Cls.from_name(protocol.APP,'Ollama')()
        journal = Journal(protocol.resilient(worker,args.output/'transport-errors.jsonl',attempts=1),
                          handles['raw-generations.jsonl'],handles['attempts.jsonl'],raw,protocol.strict_records(args.output/'attempts.jsonl'))
        try:
            worker.request.autoscaler.update(min_containers=2,max_containers=2)
            infrastructure.paired.durable_line(handles['capacity.jsonl'],{'min_containers':2,'max_containers':2})
            tags = {v['name']:v['digest'] for v in journal('/api/tags')['models']}
            if tags.get(protocol.old.MODEL) != protocol.old.MODEL_DIGEST: raise ValueError('Changed frozen compact model')
            collector = Collector(trees,reviews,known,handles['train.jsonl'],journal)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(collector.one,[(r,p) for r in rows for p in spec.PREFIXES]))
        finally:
            worker.request.autoscaler.update(min_containers=0,max_containers=2)
            infrastructure.paired.durable_line(handles['capacity.jsonl'],{'min_containers':0,'max_containers':2})
        _,coverage = spec.restore(rows,trees,reviews,protocol.strict_records(args.output/'train.jsonl'),protocol.strict_records(args.output/'raw-generations.jsonl'),True)
        if registration(args) != record: raise ValueError('Registered evidence changed during collection')
        io.write(args.output/'complete.json',completion_payload(args,coverage))
        print({'complete':True,'coverage':coverage,'scored':False},flush=True)


def validated_complete(args):
    done = io.read(args.output/'complete.json')
    if done['scored'] or not done['training_only'] or done['test_used']: raise ValueError('Require the ungraded TRAIN seal')
    io.verify({str(args.output/k):v for k,v in done['digests'].items()})
    if io.read(args.output/'registration.json') != registration(args): raise ValueError('Changed registered sources or inputs')
    bundle = spec.public_inputs(args)
    rows,trees,reviews = bundle[:3]
    known,coverage = spec.restore(rows,trees,reviews,protocol.strict_records(args.output/'train.jsonl'),protocol.strict_records(args.output/'raw-generations.jsonl'),True)
    if completion_payload(args,coverage) != done: raise ValueError('Changed full physical/logical seal')
    return bundle,known,done


def aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,known,references,extra):
    _,private,_ = spec.previous.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,references)
    for item in private:
        qid = item['id']
        review = reviews[qid,'']
        truth = references[qid]
        item['registered_known_union'] = (item['all_collected_oracle'] or item['late_union_correct']
                                          or any(answer and answer == truth for answer in extra[qid]))
        for name,parent in zip(NAMES,spec.PREFIXES):
            node = known[qid,parent]['response']
            item['methods'][name] = {'answer':node['answer'],'correct':bool(node['answer'] and node['answer'] == truth),
                'review_correct':bool(review['answer'] and review['answer'] == truth),
                **spec.previous.trajectory([trees[qid][''],trees[qid][parent],node],review)}
        item['new_union'] = any(item['methods'][name]['correct'] for name in NAMES)
        item['novel'] = item['new_union'] and not item['registered_known_union']
    if sum(v['registered_known_union'] for v in private) != 50: raise ValueError('Do not redefine prior availability after outcomes')
    reports = {}
    for subject in ('all_science','chemistry','physics'):
        panel = [r for r in private if subject == 'all_science' or r['subject'] == subject]
        methods = {}
        for name in panel[0]['methods']:
            values = [r['methods'][name] for r in panel]
            methods[name] = {'n':len(panel),'correct':sum(v['correct'] for v in values),'accuracy':sum(v['correct'] for v in values)/len(panel),
                **{'mean_'+key:sum(v[key] for v in values)/len(panel) for key in spec.previous.trajectory([])}}
            if values[0]['review_correct'] is not None: methods[name]['review_correct'] = sum(v['review_correct'] for v in values)
        reports[subject] = {'n':len(panel),'methods':methods,'registered_known_union':sum(r['registered_known_union'] for r in panel),
            'new_two_endpoint_union':sum(r['new_union'] for r in panel),'novel_questions':sum(r['novel'] for r in panel),
            'combined_hindsight_union':sum(r['new_union'] or r['registered_known_union'] for r in panel)}
    main = reports['all_science']
    ratio = max(main['methods'][name]['mean_compute'] for name in NAMES)/main['methods']['Strong retained second repair']['mean_compute']
    checks = {'at_least6_novel_questions':main['novel_questions'] >= 6,'maximum_endpoint_compute_at_most1p25':ratio <= 1.25}
    return reports,private,{'criteria':spec.SCREEN,'checks':checks,'passed':all(checks.values()),
        'maximum_ratio_to_rootstrong_retained':ratio,'controller_promoted':False,'hindsight_availability_only':True}


def grade(args):
    bundle,known,done = validated_complete(args)
    rows,trees,reviews,ordinary,ordinary_reviews,late,*_ = bundle
    target = args.output/'graded'
    if target.exists(): raise ValueError('Preserve first completed grading')
    labels = {r['id']:r for r in old.fixed.lines(args.data/'splits/train-questions.jsonl')}
    if any(protocol.public_row(labels[r['id']]) != r for r in rows): raise ValueError('Reference differs from paid public question')
    reports,private,screen = aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,known,
        {r['id']:labels[r['id']]['answer'] for r in rows},spec.extra_solver_answers(args,rows))
    validated_complete(args)
    target.mkdir()
    io.write(target/'private-outcomes.json',private)
    result = {'schema':spec.SCHEMA,'reports':reports,'screen':screen,'coverage':done['coverage'],
        'complete_sha256':io.sha(args.output/'complete.json'),'private_outcomes_sha256':io.sha(target/'private-outcomes.json'),
        'training_only':True,'test_used':False,'development_used':False,'new_reviewer_calls':0,'new_neural_calls':0,
        'transport_error_records':done['transport_error_records'],'acquisition_compute_is_lower_bound':done['acquisition_compute_is_lower_bound'],
        'controller_accuracy_claim':False,'matched_B_required_if_promoted':True}
    io.write(target/'summary.json',result)
    print({'screen':screen,'methods':{name:reports['all_science']['methods'][name] for name in NAMES}},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','collect','grade'))
    for key in ('data','mentor','complete','late','output'): parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    with guarded.training_only_reads():
        if args.phase == 'register':
            record = registration(args)
            args.output.mkdir(parents=True,exist_ok=False)
            io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'new_solver_calls':128,'new_reviewer_calls':0},flush=True)
        elif args.phase == 'collect': collect(args)
        else: grade(args)


if __name__ == '__main__': main()
