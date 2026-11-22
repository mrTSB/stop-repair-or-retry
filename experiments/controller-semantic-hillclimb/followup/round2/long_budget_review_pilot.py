"""Bounded64-question4096-vs1024 TRAIN reasoning-dose pilot.

Retains all wrong answers and all previous control paths. The looser acquisition
cost screen is only for headroom: a promoted C still must cost less than best B.
At most two solver continuations and two existing-app cloud GPUs are permitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
from pathlib import Path
import sys

import long_budget_review_protocol as spec

old,io,protocol,previous=spec.old,spec.io,spec.protocol,spec.previous
NAMES=('Long4096 first guided','Long4096 retained second repair')


def registration(args):
    rows,trees,reviews,ordinary,ordinary_reviews,late,prior,evidence=spec.public_inputs(args)
    first=[]
    for row in rows:
        try:spec.review(row,trees[row['id']],{})
        except spec.MissingPhysical as missing:
            first.append({'id':row['id'],'payload_sha256':old.digest(missing.payload),
                'prompt_sha256':hashlib.sha256(missing.payload['prompt'].encode()).hexdigest(),
                'seed':missing.payload['options']['seed'],'root_sha256':old.digest(trees[row['id']]['']),
                'compact_seeds':[protocol.seed(row['id'],p) for p in ('repair','repair/repair')]})
        else:raise AssertionError('Empty physical cache completed a teacher review')
    return {'schema':spec.SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{n:str(getattr(args,n).resolve()) for n in ('data','mentor','complete','late','output')},
        'selection':prior['selection'],'questions':64,'subject_counts':spec.COUNTS,'logical_reviews':64,
        'minimum_physical_teacher_calls':64,'physical_teacher_calls':128,'compact_calls':128,'maximum_total_physical_calls':256,'new_neural_calls':0,
        'teacher':{'model':protocol.old.JUDGE,'digest':protocol.old.JUDGE_DIGEST,'thinking_cap':4096,'answer_cap':512,
            'temperature':0,'context_length':8192,'phase_seed_names':list(spec.PHASES),'always_two_phases':False,
            'phase1_visibility':'Same original question and compact current solution as retained1024 root review',
            'phase2_visibility':'Same bounded-review continuation containing paid4096 thinking; only if no complete decision'},
        'compact':{'model':protocol.old.MODEL,'digest':protocol.old.MODEL_DIGEST,'output_cap':128,'temperature':.65,
            'context_length':4096,'seed_paths':['repair','repair/repair'],'template':'Unchanged short_repair',
            'second_reuses_same_new_review':True},
        'first_requests':first,'controls_sha256':old.digest({'trees':trees,
            'reviews':sorted((q,p,v) for (q,p),v in reviews.items()),'ordinary':ordinary,
            'ordinary_reviews':sorted((q,p,v) for (q,p),v in ordinary_reviews.items()),
            'late':sorted((q,k,p,v) for (q,k,p),v in late.items())}),
        'screen':spec.SCREEN,'novelty_endpoints':list(spec.PATHS),
        'registered_known_union_scope':['all original ordinary/plain and ordinary late paths',
            'three terminal Complete paths','rootstrong retained paths','both strong-late paths',*spec.EXTRA_BANKS],
        'full_trajectory':['original compact root','new long4096 teacher (one or two physical phases)',
            'new first128-token guided repair','new second128-token retained repair'],
        'old_root_review_is_paid_prefix':False,'maximum_answer_changing_continuations':2,
        'logical_reviews_per_trajectory':1,'workers':2,'maximum_cloud_gpus':2,'transport_attempts':1,
        'no_quality_regeneration':True,'references_supplied':False,'training_only':True,'test_used':False,
        'development_values_read':False,'grade_gate':'Full64 physical/logical/source seal before any reference decoding',
        'scope':spec.SCREEN['scope'],'promotion_requires':'Same new reviewer/action availability for strongest B; separate learned/fresh validation'}


def register(args):
    record=registration(args);args.output.mkdir(parents=True,exist_ok=False)
    io.write(args.output/'registration.json',record)
    print({'registration_sha256':io.sha(args.output/'registration.json'),'physical_calls':256,'calls_started':0},flush=True)


class Collector(previous.Collector):
    def one(self,row):
        qid=row['id'];tree=self.trees[qid]
        try:
            with self.keys.get(qid):
                teacher,raw=self.acquire(spec.review,row,tree)
                self.store(spec.logical(row,tree[''],'judge','',teacher,raw))
                current=tree['']
                for path in spec.PATHS:
                    node,raw=self.acquire(spec.solver,row,current,teacher,path)
                    self.store(spec.logical(row,tree[''],'solver',path,node,raw,current,teacher))
                    current=node
                with self.lock:
                    n=sum(k[2]==spec.PATHS[-1] for k in self.known)
                    if n%8==0:print(f'Long4096 trajectories {n}/64',flush=True)
        except BaseException:self.stopped.set();raise


def completion_payload(args,coverage):
    names=['registration.json','train.jsonl','raw-generations.jsonl'];error=args.output/'transport-errors.jsonl'
    count=len(protocol.strict_records(error))
    if error.exists():names.append(error.name)
    return {'schema':spec.SCHEMA,'coverage':coverage,'digests':{n:io.sha(args.output/n) for n in names},
        'registration_sha256':io.sha(args.output/'registration.json'),'training_only':True,'test_used':False,
        'scored':False,'new_neural_calls':0,'transport_error_records':count,'acquisition_compute_is_lower_bound':bool(count),
        'maximum_answer_changing_continuations':2,'logical_reviews_per_trajectory':1}


def collect(args):
    with ExitStack() as stack:
        handle=stack.enter_context((args.output/'collection.lock').open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'complete.json').exists():raise ValueError('Preserve completed pilot; no resampling')
        contract=registration(args)
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
            or io.read(args.output/'registration.json')!=contract):raise ValueError('Changed registered source/cohort/protocol')
        rows,trees,*_=spec.public_inputs(args)
        bank,raw_path=args.output/'train.jsonl',args.output/'raw-generations.jsonl'
        stream=stack.enter_context(bank.open('a'));raw_stream=stack.enter_context(raw_path.open('a'))
        for item in (stream,raw_stream):fcntl.flock(item,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=protocol.strict_records(raw_path);known,_=spec.restore(rows,trees,protocol.strict_records(bank),raw)
        import modal
        worker=modal.Cls.from_name(protocol.APP,'Ollama')();worker.request.autoscaler.update(max_containers=2)
        journal=old.RequestJournal(protocol.resilient(worker,args.output/'transport-errors.jsonl',attempts=1),raw_stream,raw,slots=2)
        tags={r['name']:r['digest'] for r in journal('/api/tags')['models']}
        if tags.get(protocol.old.MODEL)!=protocol.old.MODEL_DIGEST or tags.get(protocol.old.JUDGE)!=protocol.old.JUDGE_DIGEST:
            raise ValueError('Frozen compact/reviewer model changed')
        collector=Collector(trees,known,stream,journal)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(collector.one,rows))
        _,coverage=spec.restore(rows,trees,protocol.strict_records(bank),protocol.strict_records(raw_path),True)
        if registration(args)!=contract:raise ValueError('Registered evidence changed during acquisition')
        io.write(args.output/'complete.json',completion_payload(args,coverage))
        print({'complete':True,'coverage':coverage,'scored':False},flush=True)


def validated_complete(args):
    done=io.read(args.output/'complete.json')
    if (done['schema']!=spec.SCHEMA or done['scored'] is not False or done['training_only'] is not True
        or done['test_used'] is not False or done['new_neural_calls']!=0):raise ValueError('Require sealed ungraded TRAIN-only grid')
    io.verify({str(args.output/n):v for n,v in done['digests'].items()})
    if io.read(args.output/'registration.json')!=registration(args):raise ValueError('Changed registered pilot evidence')
    inputs=spec.public_inputs(args);rows,trees=inputs[:2]
    known,coverage=spec.restore(rows,trees,protocol.strict_records(args.output/'train.jsonl'),
        protocol.strict_records(args.output/'raw-generations.jsonl'),True)
    if done!=completion_payload(args,coverage):raise ValueError('Changed full physical/logical cost seal')
    return inputs,known,done


def aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,known,references,extra_answers,expected_known=None):
    if len(rows)!=64 or len(known)!=192:raise ValueError('Grade all64 complete trajectories')
    _,outcomes,_=previous.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,references)
    for record in outcomes:
        qid=record['id'];teacher=known[qid,'judge','']['response']
        nodes=[trees[qid]['']];target=references[qid]
        record['registered_known_union']=(record['all_collected_oracle'] or record['late_union_correct']
            or any(answer and answer==target for answer in extra_answers[qid]))
        for name,path in zip(NAMES,spec.PATHS):
            node=known[qid,'solver',path]['response'];nodes.append(node)
            record['methods'][name]={'correct':bool(node['answer'] and node['answer']==target),
                'review_correct':bool(teacher['answer'] and teacher['answer']==target),'answer':node['answer'],
                **previous.trajectory(nodes,teacher)}
        record['new_union']=any(record['methods'][n]['correct'] for n in NAMES)
        record['novel']=record['new_union'] and not record['registered_known_union']
    if expected_known is not None and sum(r['registered_known_union'] for r in outcomes)!=expected_known:
        raise ValueError('The registered full known union changed; do not redefine the novelty target')
    reports={}
    for subject in ('all_science','chemistry','physics'):
        panel=[r for r in outcomes if subject=='all_science' or r['subject']==subject];n=len(panel)
        methods={}
        for name in panel[0]['methods']:
            values=[r['methods'][name] for r in panel]
            methods[name]={'n':n,'correct':sum(v['correct'] for v in values),'accuracy':sum(v['correct'] for v in values)/n,
                **{'mean_'+k:sum(v[k] for v in values)/n for k in previous.trajectory([])}}
            if values[0]['review_correct'] is not None:
                methods[name].update(review_correct=sum(v['review_correct'] for v in values),
                    correct_review_wrong_compact=sum(v['review_correct'] and not v['correct'] for v in values),
                    wrong_review_correct_compact=sum(not v['review_correct'] and v['correct'] for v in values))
        paired={name:{control:{'wins':sum(r['methods'][name]['correct'] and not r['methods'][control]['correct'] for r in panel),
            'losses':sum(not r['methods'][name]['correct'] and r['methods'][control]['correct'] for r in panel)}
            for control in methods if control!=name} for name in NAMES}
        reports[subject]={'n':n,'methods':methods,'paired':paired,
            'registered_known_union':sum(r['registered_known_union'] for r in panel),
            'new_two_endpoint_union':sum(r['new_union'] for r in panel),'novel_questions':sum(r['novel'] for r in panel),
            'combined_hindsight_union':sum(r['registered_known_union'] or r['new_union'] for r in panel)}
    main=reports['all_science'];metrics=main['methods'];cost=metrics[NAMES[1]]['mean_compute']
    ratio=cost/metrics['Strong retained second repair']['mean_compute']
    checks={'at_least6_novel_questions':main['novel_questions']>=6,'second_trajectory_compute_at_most2p5':ratio<=2.5}
    screen={'criteria':spec.SCREEN,'checks':checks,'passed':all(checks.values()),'second_trajectory_mean_compute':cost,
        'ratio_to_rootstrong_retained':ratio,'controller_promoted':False,'hindsight_availability_only':True}
    return reports,outcomes,screen


def grade(args):
    inputs,known,done=validated_complete(args);rows,trees,reviews,ordinary,ordinary_reviews,late,*_=inputs
    target=args.output/'graded'
    if target.exists():raise ValueError('Preserve first completed diagnostic')
    labels={r['id']:r for r in old.fixed.lines(args.data/'splits/train-questions.jsonl')}
    if any(protocol.public_row(labels[r['id']])!=r for r in rows):raise ValueError('Reference question changed')
    reports,private,screen=aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,known,
        {r['id']:labels[r['id']]['answer'] for r in rows},spec.extra_solver_answers(args,rows))
    validated_complete(args);target.mkdir();io.write(target/'private-outcomes.json',private)
    result={'schema':spec.SCHEMA,'reports':reports,'screen':screen,'questions':64,'coverage':done['coverage'],
        'complete_sha256':io.sha(args.output/'complete.json'),'private_outcomes_sha256':io.sha(target/'private-outcomes.json'),
        'training_only':True,'test_used':False,'new_neural_calls':0,'controller_accuracy_claim':False,
        'causal_anchoring_only_claim':False,'comparison':'Same root review protocol;4096 vs1024 thought cap','matched_B_required_if_promoted':True,
        'transport_error_records':done['transport_error_records'],
        'acquisition_compute_is_lower_bound':done['acquisition_compute_is_lower_bound']}
    io.write(target/'summary.json',result)
    print({'screen':screen,'methods':{n:reports['all_science']['methods'][n] for n in NAMES}},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('plan','register','collect','grade'))
    for name in ('data','mentor','complete','late','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');args=parser.parse_args()
    if args.phase=='plan':
        value=registration(args);print({k:value[k] for k in ('schema','questions','physical_teacher_calls','compact_calls','screen')},flush=True)
    else:
        import completion_v4_diagnostic as guarded
        with guarded.training_only_reads():
            {'register':register,'collect':collect,'grade':grade}[args.phase](args)
