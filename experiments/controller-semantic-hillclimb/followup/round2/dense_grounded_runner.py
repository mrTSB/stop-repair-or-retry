"""Sealed TRAIN comparison with local query forwards included in compute."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import contrastive_review_runner as prior
import dense_grounded_protocol as spec

original, io, protocol = prior.original, prior.io, prior.protocol
SCHEMA = 'dense-grounded-local-query-accounting-runner-v1'
NAMES = ('Dense grounded first guided', 'Dense grounded retained second repair')
base_registration = prior.infrastructure.base_registration
base_completion = original.completion_payload
REUSE_BANKS = spec.EXTRA_BANKS


def require_priority(args):
    prior.require_priority(args)
    folder = args.data.parent/'contrastive-three-phase-root-review-pilot-v1'
    done = io.read(folder/'complete.json')
    io.verify({str(folder/k):v for k,v in done['digests'].items()})
    capacity = protocol.strict_records(folder/'capacity.jsonl')
    if not capacity or capacity[-1] != {'min_containers':0,'max_containers':2}:
        raise ValueError('Previous collector must release capacity first')


def observations(rows):
    receipts = [spec.evidence(row)[1] for row in rows]
    unique = len({r['input_sha256'] for r in receipts})
    actual = spec.retrieval.retriever().encoded.cache_info().misses
    if actual != unique: raise ValueError('Unexpected unaccounted local query forward')
    return {'charged_query_observations':len(receipts),'unique_query_inputs':unique,
        'actual_local_query_forwards_this_process':actual,
        'charged_query_token_slots':sum(r['actual_forward_token_slots'] for r in receipts),
        'total_inference_query_compute':sum(r['compute'] for r in receipts),
        'receipts_sha256':original.old.digest(receipts)}


def reused_inputs(args):
    lookup, files = {}, {}
    for name in REUSE_BANKS:
        path = args.data.parent/name/'raw-generations.jsonl'
        if not path.exists(): continue
        files[str(path.resolve())] = io.sha(path)
        for row in protocol.strict_records(path):
            original.old.RequestJournal.validate(row)
            key = row['request_sha256']
            order = lambda r:(r['started_unix'],r['returned_unix'],r['response_sha256'])
            if key not in lookup or order(row) < order(lookup[key]): lookup[key] = row
    rows,trees,*_ = spec.public_inputs(args)
    logical, used, complete = [],set(),[]
    for row in rows:
        tree = trees[row['id']]
        current = tree['']
        try: teacher,calls = spec.review(row,tree,lookup,used)
        except spec.MissingPhysical: continue
        logical.append(spec.logical(row,tree[''],'judge','',teacher,calls))
        for path in spec.PATHS:
            try: node,calls = spec.solver(row,current,teacher,path,lookup,used)
            except spec.MissingPhysical: break
            logical.append(spec.logical(row,tree[''],'solver',path,node,calls,current,teacher))
            current = node
        else: complete.append(row['id'])
    raw = sorted((lookup[k] for k in used),key=lambda r:(r['started_unix'],r['request_sha256']))
    attempts = [{'schema':spec.SCHEMA,'endpoint':r['endpoint'],'payload':r['payload'],
                 'request_sha256':r['request_sha256'],'payload_sha256':original.old.digest(r['payload'])} for r in raw]
    spec.restore(rows,trees,logical,raw)
    return logical,raw,attempts,complete,files


def registration(args):
    require_priority(args)
    record = base_registration(args)
    rows,*_ = spec.public_inputs(args)
    stats = observations(rows)
    logical,raw,attempts,complete,files = reused_inputs(args)
    if len(complete) > 58: raise ValueError('Insufficient new trajectories to meet the novelty screen')
    record.update(runner_schema=SCHEMA,source_files=io.sources(sys.modules[__name__]),
        new_neural_calls=stats['actual_local_query_forwards_this_process'],local_query_observations=stats,
        registered_known_union_questions=50,maximum_total_physical_calls=256,
        reused_physical_calls=len(raw),maximum_new_physical_calls=256-len(raw),
        reused_logical_nodes=len(logical),fully_reused_questions=len(complete),
        reuse_identity=original.old.digest({'logical':logical,'raw':raw,'attempts':attempts}),
        reuse_rule='Earliest exact paid physical response across fixed prior banks, by time then response hash; never correctness',
        all_reused_tokens_charged_at_inference=True,
        retrieval={'new_observation':True,'model':spec.retrieval.index.MODEL,'revision':spec.retrieval.index.REVISION,
            'parameters':33360000,'query':'Public question stem only, capped800 characters; no choices/current answer/references',
            'maximum_encoder_tokens':256,'ranker':'Reciprocal-rank fusion60 of dense and lexical top60; subject filter; two distinct sources',
            'maximum_passage_words':85,'query_forwards_charged_even_if_cached':True,
            'index_built_without_benchmark_records':True,'local_CPU_only':True,'cloud_training':False,
            'inference_cost_formula':'query_parameter_count*actual_forward_token_slots/3.09e9'},
        full_trajectory=['original compact root','paid local semantic query observation',
            'original grounded1024/512 review','unchanged128-token guided compact repair','unchanged128-token retained compact repair'],
        final_C_target={'accuracy_strictly_greater_than':.80,'gap_above_best_A_B_at_least':.10,'compute_below_B':True},
        promotion_requires='Six novel correct TRAIN paths, full query-inclusive cost screen, learned POMDP integration, strongest matched B, fresh validation')
    record['input_evidence'].update(files)
    record['teacher'].update(phase1_visibility='Original question, paid root solution and new semantic textbook excerpts',
        phase2_visibility='Original bounded continuation with paid reasoning and semantic excerpts; conditional on incomplete JSON')
    for key in ('retrieval_neural_calls','retrieval_CPU_not_in_parameter_token_proxy'):
        record.pop(key)
    return record


def seed_reused(args):
    record = registration(args)
    if io.read(args.output/'registration.json') != record or io.sha(args.output/'registration.json') != args.registration_sha256:
        raise ValueError('Changed registered semantic pilot')
    logical,raw,attempts,_,_ = reused_inputs(args)
    payloads = {name:''.join(json.dumps(r,sort_keys=True)+'\n' for r in values).encode()
                for name,values in (('train.jsonl',logical),('raw-generations.jsonl',raw),('attempts.jsonl',attempts))}
    receipt = {'schema':SCHEMA,'registration_sha256':args.registration_sha256,'reuse_identity':record['reuse_identity'],
        'prefixes':{name:{'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()} for name,value in payloads.items()}}
    with io.lock(args.output/'.reuse.lock'):
        target = args.output/'reuse-receipt.json'
        if target.exists():
            if io.read(target) != receipt: raise ValueError('Changed reuse receipt')
            for name,value in payloads.items():
                if not (args.output/name).read_bytes().startswith(value): raise ValueError('Changed reused prefix')
            return
        if any((args.output/name).exists() for name in payloads): raise ValueError('Inspect partial seed instead of repeating calls')
        for name,value in payloads.items(): io.atomic_bytes(args.output/name,value)
        io.write(target,receipt)


def completion_payload(args,coverage):
    result = base_completion(args,coverage)
    rows,*_ = spec.public_inputs(args)
    stats = observations(rows)
    result.update(schema=spec.SCHEMA,new_neural_calls=stats['actual_local_query_forwards_this_process'],
        local_query_observations=stats,neural_call_scope='Actual local query forwards in this collection process; offline registration and verification separate',
        query_cost_included_when_graded=True)
    result['digests']['reuse-receipt.json'] = io.sha(args.output/'reuse-receipt.json')
    return result


def validated_complete(args):
    done = io.read(args.output/'complete.json')
    if (done['schema'] != spec.SCHEMA or done['scored'] or not done['training_only'] or done['test_used']):
        raise ValueError('Complete ungraded TRAIN seal required')
    io.verify({str(args.output/k):v for k,v in done['digests'].items()})
    if io.read(args.output/'registration.json') != registration(args): raise ValueError('Changed semantic pilot evidence')
    bundle = spec.public_inputs(args)
    rows,trees = bundle[:2]
    known,coverage = spec.restore(rows,trees,protocol.strict_records(args.output/'train.jsonl'),
                                  protocol.strict_records(args.output/'raw-generations.jsonl'),True)
    if done != completion_payload(args,coverage): raise ValueError('Changed complete neural/physical ledger')
    return bundle,known,done


def add_query_costs(reports,private,charges):
    if len(private) != len(charges) or {r['id'] for r in private} != set(charges):
        raise ValueError('Full paid query coverage required')
    for row in private:
        bill = charges[row['id']]
        for name in NAMES:
            value = row['methods'][name]
            value['compute'] += bill['compute']
            value['total_parameter_compute'] += bill['compute']
            value['retrieval_query_compute'] = bill['compute']
            value['retrieval_query_token_slots'] = bill['actual_forward_token_slots']
            value['retrieval_query_forwards'] = bill['charged_query_forwards']
    for subject,report in reports.items():
        panel = [r for r in private if subject == 'all_science' or r['subject'] == subject]
        for name in NAMES:
            for key in ('compute','total_parameter_compute','retrieval_query_compute','retrieval_query_token_slots','retrieval_query_forwards'):
                report['methods'][name]['mean_'+key] = sum(r['methods'][name][key] for r in panel)/len(panel)
    main = reports['all_science']
    cost = main['methods'][NAMES[1]]['mean_compute']
    ratio = cost/main['methods']['Strong retained second repair']['mean_compute']
    checks = {'at_least6_novel_questions':main['novel_questions'] >= 6,'second_trajectory_compute_at_most1p25':ratio <= 1.25}
    return {'criteria':spec.SCREEN,'checks':checks,'passed':all(checks.values()),'second_trajectory_mean_compute':cost,
            'ratio_to_rootstrong_retained':ratio,'controller_promoted':False,'hindsight_availability_only':True,
            'all_query_encoder_compute_included':True}


def grade(args):
    bundle,known,done = validated_complete(args)
    rows,trees,reviews,ordinary,ordinary_reviews,late,*_ = bundle
    target = args.output/'graded'
    if target.exists(): raise ValueError('Preserve first completed grading')
    labels = {r['id']:r for r in original.old.fixed.lines(args.data/'splits/train-questions.jsonl')}
    if any(protocol.public_row(labels[r['id']]) != r for r in rows): raise ValueError('Reference differs from paid public question')
    reports,private,_ = original.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,known,
        {r['id']:labels[r['id']]['answer'] for r in rows},spec.extra_solver_answers(args,rows),expected_known=50)
    charges = {r['id']:known[r['id'],'judge','']['response']['retrieval_encoder_charge'] for r in rows}
    screen = add_query_costs(reports,private,charges)
    validated_complete(args)
    target.mkdir()
    io.write(target/'private-outcomes.json',private)
    result = {'schema':spec.SCHEMA,'runner_schema':SCHEMA,'reports':reports,'screen':screen,'coverage':done['coverage'],
        'local_query_observations':done['local_query_observations'],'complete_sha256':io.sha(args.output/'complete.json'),
        'private_outcomes_sha256':io.sha(target/'private-outcomes.json'),'training_only':True,'test_used':False,
        'development_used':False,'controller_accuracy_claim':False,'matched_B_required_if_promoted':True,
        'transport_error_records':done['transport_error_records'],'new_neural_calls':done['new_neural_calls'],
        'acquisition_compute_is_lower_bound':done['acquisition_compute_is_lower_bound']}
    io.write(target/'summary.json',result)
    print({'screen':screen,'methods':{name:reports['all_science']['methods'][name] for name in NAMES}},flush=True)


@contextmanager
def configured_runner():
    names = ('spec','NAMES','registration','require_priority','completion_payload','validated_complete')
    before = {name:getattr(original,name) for name in names}
    for name in names: setattr(original,name,globals()[name])
    try: yield
    finally:
        for name,value in before.items(): setattr(original,name,value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','collect','grade'))
    for name in ('data','mentor','complete','late','output','priority'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    with prior.infrastructure.predecessor.guarded.training_only_reads(),configured_runner():
        if args.phase == 'register':
            record = registration(args)
            args.output.mkdir(parents=True,exist_ok=False)
            io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'reused_calls':record['reused_physical_calls'],
                   'maximum_new_calls':record['maximum_new_physical_calls'],'local_query_observations':record['local_query_observations']},flush=True)
        elif args.phase == 'collect':
            seed_reused(args)
            original.collect(args)
        else: grade(args)


if __name__ == '__main__': main()
