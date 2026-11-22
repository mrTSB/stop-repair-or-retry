"""Prespecified64×2 TRAIN-only strong late-review timing pilot.

No call is authorized by creating this file. Registration is separate from
collection; grading requires all128 logical reviews/final repairs and every
physical receipt to be sealed. The original action banks remain immutable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading

import strong_late_review_protocol as spec

old,io,protocol=spec.previous,spec.io,spec.protocol


def registration(args):
    rows,trees,reviews,ordinary,ordinary_reviews,chosen,prefix_sha,evidence=spec.public_inputs(args)
    first=[]
    for row in rows:
        for parent in spec.PREFIXES:
            try:spec.review(row,trees[row['id']],parent,{})
            except spec.MissingPhysical as missing:
                first.append({'id':row['id'],'prefix':parent,'seed_identity':spec.seeded_row(row,parent)['id'],
                    'first_review_payload_sha256':old.digest(missing.payload),
                    'first_review_prompt_sha256':hashlib.sha256(missing.payload['prompt'].encode()).hexdigest(),
                    'first_review_seed':missing.payload['options']['seed'],
                    'compact_seed':protocol.seed(row['id'],parent+'/repair'),
                    'root_sha256':old.digest(trees[row['id']]['']),
                    'current_sha256':old.digest(trees[row['id']][parent])})
            else:raise AssertionError('An empty physical cache unexpectedly completed a review')
    return {'schema':spec.SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{name:str(getattr(args,name).resolve()) for name in ('data','mentor','complete','output')},
        'selection':chosen,'questions':64,'subject_counts':spec.COUNTS,'prefixes':list(spec.PREFIXES),
        'logical_reviews':128,'minimum_physical_teacher_calls':128,'maximum_physical_teacher_calls':256,
        'new_compact_calls':128,'maximum_total_physical_calls':384,'new_neural_calls':0,
        'teacher':{'model':protocol.old.JUDGE,'digest':protocol.old.JUDGE_DIGEST,
            'template':'Unchanged long_review_pilot.review on the paid current plain prefix',
            'thinking_cap':1024,'answer_cap':512,'temperature':0,'context_length':8192,
            'seed_namespace':'original qid::strong-late-{repair|retry}-v1, then frozen bounded phase name'},
        'compact':{'model':protocol.old.MODEL,'digest':protocol.old.MODEL_DIGEST,'output_cap':128,
            'temperature':.65,'context_length':4096,'template':'Unchanged short_repair(current, paid late review)',
            'seed':'Original paired {repair|retry}/repair seed'},
        'maximum_answer_changing_continuations':2,'logical_reviews_per_trajectory':1,
        'root_review_supplied_to_late_teacher':False,'first_requests':first,'plain_prefix_sha256':prefix_sha,
        'controls_sha256':old.digest({'trees':trees,'root_reviews':sorted((q,p,v) for (q,p),v in reviews.items()),
            'ordinary_trees':ordinary,'ordinary_reviews':sorted((q,p,v) for (q,p),v in ordinary_reviews.items())}),
        'screen':spec.SCREEN,'fixed_cost_screen':'Mean of all128 complete new trajectories / mean64 root-strong retained trajectories',
        'additional_descriptive_audits':['All already-collected ordinary/late +Complete+rootstrong union and novelty',
            'Review correctness versus final compact correctness for each reviewed method',
            'All three existing validity-based Complete controls; no extra calls or changed screen'],
        'full_trajectory':['original root','ordinary repair OR retry','one paid late strong review (all physical phases)',
                           'one final128-token guided compact repair'],
        'workers':2,'maximum_cloud_gpus':2,'transport_attempts':1,'no_quality_regeneration':True,
        'references_supplied':False,'training_only':True,'test_used':False,'development_values_read':False,
        'grade_gate':'Full64×2 physical/logical/source hash seal before any reference decoding',
        'promotion_requires':'Matched B timing/action availability and separately validated learned policy'}


def register(args):
    record=registration(args);args.output.mkdir(parents=True,exist_ok=False)
    io.write(args.output/'registration.json',record)
    print({'registered':True,'registration_sha256':io.sha(args.output/'registration.json'),
           'logical_reviews':128,'maximum_teacher_phases':256,'compact_calls':128,'calls_started':0},flush=True)


class Collector:
    def __init__(self,trees,known,stream,journal):
        self.trees,self.known,self.stream,self.journal=trees,known,stream,journal
        self.keys=old.KeyLocks();self.lock=threading.RLock();self.stopped=threading.Event()
    def store(self,item):
        key=item['id'],item['kind'],item['path']
        with self.lock:
            if key in self.known:
                if self.known[key]!=item:raise ValueError('Completed logical answer changed')
            else:old.durable_line(self.stream,item);self.known[key]=item
    def acquire(self,function,*args):
        while True:
            if self.stopped.is_set():raise RuntimeError('Another collection request failed')
            try:return function(*args,self.journal.known)
            except spec.MissingPhysical as missing:self.journal('/api/generate',missing.payload)
    def one(self,pair):
        row,parent=pair;qid=row['id'];tree=self.trees[qid]
        try:
            with self.keys.get((qid,parent)):
                teacher,raw=self.acquire(spec.review,row,tree,parent)
                self.store(spec.logical(row,tree,parent,'judge',teacher,raw))
                answer,raw=self.acquire(spec.solver,row,tree,parent,teacher)
                self.store(spec.logical(row,tree,parent,'solver',answer,raw,teacher))
                with self.lock:
                    n=sum(key[1]=='solver' for key in self.known)
                    if n%8==0:print(f'Strong late-review cells {n}/128',flush=True)
        except BaseException:self.stopped.set();raise


def completion_payload(args,coverage):
    names=['registration.json','train.jsonl','raw-generations.jsonl']
    errors=args.output/'transport-errors.jsonl';count=len(protocol.strict_records(errors))
    if errors.exists():names.append(errors.name)
    return {'schema':spec.SCHEMA,'coverage':coverage,'digests':{n:io.sha(args.output/n) for n in names},
        'registration_sha256':io.sha(args.output/'registration.json'),'new_neural_calls':0,
        'training_only':True,'test_used':False,'scored':False,'transport_error_records':count,
        'acquisition_compute_is_lower_bound':bool(count),'maximum_answer_changing_continuations':2,
        'logical_reviews_per_trajectory':1}


def collect(args):
    with ExitStack() as stack:
        lock=stack.enter_context((args.output/'collection.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'complete.json').exists():raise ValueError('Preserve completed pilot; no resampling')
        contract=registration(args)
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=contract):
            raise ValueError('Explicit registered cohort/protocol/source changed')
        rows,trees,*_=spec.public_inputs(args)
        bank,raw_path=args.output/'train.jsonl',args.output/'raw-generations.jsonl'
        stream=stack.enter_context(bank.open('a'));raw_stream=stack.enter_context(raw_path.open('a'))
        for handle in (stream,raw_stream):fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw_records=protocol.strict_records(raw_path)
        known,_=spec.restore(rows,trees,protocol.strict_records(bank),raw_records)
        import modal
        worker=modal.Cls.from_name(protocol.APP,'Ollama')()
        worker.request.autoscaler.update(max_containers=2)
        journal=old.RequestJournal(protocol.resilient(worker,args.output/'transport-errors.jsonl',attempts=1),
                                   raw_stream,raw_records,slots=2)
        tags={r['name']:r['digest'] for r in journal('/api/tags')['models']}
        if tags.get(protocol.old.MODEL)!=protocol.old.MODEL_DIGEST or tags.get(protocol.old.JUDGE)!=protocol.old.JUDGE_DIGEST:
            raise ValueError('Frozen compact or reviewer weights changed')
        collector=Collector(trees,known,stream,journal)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(collector.one,[(r,p) for r in rows for p in spec.PREFIXES]))
        _,coverage=spec.restore(rows,trees,protocol.strict_records(bank),protocol.strict_records(raw_path),True)
        if registration(args)!=contract:raise ValueError('Registered evidence changed during collection')
        io.write(args.output/'complete.json',completion_payload(args,coverage))
        print({'complete':True,'coverage':coverage,'scored':False},flush=True)


def validated_complete(args):
    done=io.read(args.output/'complete.json')
    if (done['schema']!=spec.SCHEMA or done['training_only'] is not True or done['test_used'] is not False
            or done['scored'] is not False or done['new_neural_calls']!=0):raise ValueError('Require full sealed ungraded TRAIN grid')
    io.verify({str(args.output/n):v for n,v in done['digests'].items()})
    if io.read(args.output/'registration.json')!=registration(args):raise ValueError('Registered pilot source/input changed')
    rows,trees,reviews,ordinary,ordinary_reviews,*_=spec.public_inputs(args)
    known,coverage=spec.restore(rows,trees,protocol.strict_records(args.output/'train.jsonl'),
        protocol.strict_records(args.output/'raw-generations.jsonl'),True)
    if done!=completion_payload(args,coverage):raise ValueError('Full pilot seal or physical cost evidence changed')
    return rows,trees,reviews,ordinary,ordinary_reviews,known,done


def trajectory(nodes,review=None):
    paid={**review,'physical_api_calls':review.get('physical_api_calls',1)} if review is not None else None
    return old.trajectory(nodes,paid)


def aggregate(rows,trees,root_reviews,ordinary,ordinary_reviews,known,references):
    if len(rows)!=64 or len(known)!=256:raise ValueError('Score all64×2 cells, without filtering')
    outcomes=[]
    for row in rows:
        qid=row['id'];tree=trees[qid];target=references[qid]
        methods={name:([tree[p] for p in paths],None) for name,paths in old.fixed.FIXED.items()}
        methods.update({'Complete':([tree[''],tree['complete']],None),
            'Repair then Complete':([tree[''],tree['repair'],tree['repair/complete']],None),
            'Retry then Complete':([tree[''],tree['retry'],tree['retry/complete']],None),
            'Strong first guided':([tree[''],tree['guided']],root_reviews[qid,'']),
            'Strong retained second repair':([tree[''],tree['guided'],tree['guided/reuse']],root_reviews[qid,'']),
            'Strong repair after retry':([tree[''],tree['retry'],tree['retry/reuse']],root_reviews[qid,''])})
        methods['Stop if valid else Complete']=([tree['']] if tree['']['answer'] else [tree[''],tree['complete']],None)
        for parent in spec.PREFIXES:
            nodes=[tree[''],tree[parent]]
            if not tree[parent]['answer']:nodes.append(tree[parent+'/complete'])
            methods[parent.title()+' then Complete if invalid']=(nodes,None)
        for parent in spec.PREFIXES:
            methods['Ordinary late after '+parent]=([ordinary[qid][''],ordinary[qid][parent],ordinary[qid][parent+'/reviewed']],
                                                   ordinary_reviews[qid,parent])
            methods['Strong late after '+parent]=([tree[''],tree[parent],known[qid,'solver',parent+'/late-guided']['response']],
                                                 known[qid,'judge',parent]['response'])
        values={name:{'correct':bool(nodes[-1]['answer'] and nodes[-1]['answer']==target),
            'review_correct':bool(review['answer'] and review['answer']==target) if review is not None else None,
            'answer':nodes[-1]['answer'],**trajectory(nodes,review)} for name,(nodes,review) in methods.items()}
        paths=(*old.ORACLE_PATHS,'complete','repair/complete','retry/complete')
        base=any(tree[p]['answer'] and tree[p]['answer']==target for p in paths)
        all_known=base or any(n['answer'] and n['answer']==target for n in ordinary[qid].values())
        new=any(values['Strong late after '+p]['correct'] for p in spec.PREFIXES)
        outcomes.append({'id':qid,'subject':row['subject'],'methods':values,'known_base_oracle':bool(base),
                         'all_collected_oracle':bool(all_known),'novel_beyond_all_collected':new and not all_known,
                         'late_union_correct':new,'novel_question':new and not base})
    reports={}
    for subject in ('all_science','chemistry','physics'):
        panel=[v for v in outcomes if subject=='all_science' or v['subject']==subject];n=len(panel)
        methods={name:{'n':n,'correct':sum(v['methods'][name]['correct'] for v in panel),
            'accuracy':sum(v['methods'][name]['correct'] for v in panel)/n,
            **{'mean_'+key:sum(v['methods'][name][key] for v in panel)/n for key in trajectory([])}}
            for name in panel[0]['methods']}
        for name,summary in methods.items():
            if panel[0]['methods'][name]['review_correct'] is not None:
                summary.update(review_correct=sum(v['methods'][name]['review_correct'] for v in panel),
                    review_accuracy=sum(v['methods'][name]['review_correct'] for v in panel)/n,
                    correct_review_wrong_compact=sum(v['methods'][name]['review_correct'] and not v['methods'][name]['correct'] for v in panel),
                    wrong_review_correct_compact=sum(not v['methods'][name]['review_correct'] and v['methods'][name]['correct'] for v in panel))
        paired={}
        for parent in spec.PREFIXES:
            new='Strong late after '+parent
            paired[new]={old_name:{'wins':sum(v['methods'][new]['correct'] and not v['methods'][old_name]['correct'] for v in panel),
                'losses':sum(not v['methods'][new]['correct'] and v['methods'][old_name]['correct'] for v in panel)}
                for old_name in methods if old_name!=new}
        reports[subject]={'n':n,'methods':methods,'paired':paired,
            'known_plain_Complete_rootstrong_oracle':sum(v['known_base_oracle'] for v in panel),
            'late_two_path_hindsight_union':sum(v['late_union_correct'] for v in panel),
            'novel_questions':sum(v['novel_question'] for v in panel),
            'combined_hindsight_union':sum(v['known_base_oracle'] or v['late_union_correct'] for v in panel),
            'all_collected_ordinary_late_Complete_rootstrong_oracle':sum(v['all_collected_oracle'] for v in panel),
            'novel_questions_beyond_all_collected_paths':sum(v['novel_beyond_all_collected'] for v in panel),
            'all_collected_plus_newlate_hindsight_union':sum(v['all_collected_oracle'] or v['late_union_correct'] for v in panel)}
    main=reports['all_science'];methods=main['methods']
    average=sum(methods['Strong late after '+p]['mean_compute'] for p in spec.PREFIXES)/2
    ratio=average/methods['Strong retained second repair']['mean_compute']
    checks={'at_least6_novel_questions':main['novel_questions']>=6,'mean_full_compute_at_most_1p25_retained':ratio<=1.25}
    return reports,outcomes,{'criteria':spec.SCREEN,'checks':checks,'passed':all(checks.values()),
        'mean_all128_late_trajectory_compute':average,'ratio_to_rootstrong_retained':ratio,
        'controller_promoted':False,'union_is_hindsight_not_learned_accuracy':True}


def grade(args):
    rows,trees,reviews,ordinary,ordinary_reviews,known,done=validated_complete(args)
    target=args.output/'graded'
    if target.exists():raise ValueError('Preserve the first complete diagnostic')
    labels={r['id']:r for r in old.fixed.lines(args.data/'splits/train-questions.jsonl')}
    if any(protocol.public_row(labels[r['id']])!=r for r in rows):raise ValueError('Reference question differs from paid prompt')
    reports,private,screen=aggregate(rows,trees,reviews,ordinary,ordinary_reviews,known,{r['id']:labels[r['id']]['answer'] for r in rows})
    validated_complete(args);target.mkdir()
    io.write(target/'private-outcomes.json',private)
    summary={'schema':spec.SCHEMA,'reports':reports,'screen':screen,'questions':64,'late_prefixes':2,
        'complete_sha256':io.sha(args.output/'complete.json'),'private_outcomes_sha256':io.sha(target/'private-outcomes.json'),
        'coverage':done['coverage'],'training_only':True,'test_used':False,'new_neural_calls':0,
        'controller_accuracy_claim':False,'matched_B_required_if_promoted':True,
        'transport_error_records':done['transport_error_records'],
        'acquisition_compute_is_lower_bound':done['acquisition_compute_is_lower_bound']}
    io.write(target/'summary.json',summary);print({'screen':screen,'reports':reports},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('plan','register','collect','grade'))
    for name in ('data','mentor','complete','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');args=parser.parse_args()
    if args.phase=='plan':
        record=registration(args)
        print({k:record[k] for k in ('schema','questions','logical_reviews','maximum_physical_teacher_calls','new_compact_calls','screen')},flush=True)
    else:{'register':register,'collect':collect,'grade':grade}[args.phase](args)
