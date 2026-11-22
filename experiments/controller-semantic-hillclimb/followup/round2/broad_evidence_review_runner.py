"""Prespecified active-retrieval pilot with the same frozen reviewer and solver."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time
import threading

import grounded_review_runner_v2 as predecessor
import broad_evidence_review_protocol as spec
import evidence_first_review_runner as prior_runner
import json
from types import SimpleNamespace

original = predecessor.original
io, protocol = original.io, original.protocol
SCHEMA = 'broad-evidence-first-textbook-review-runner-v4'
NAMES = ('Broad evidence V4 first guided', 'Broad evidence V4 retained second repair')
base_registration = original.registration


def require_priority(args):
    predecessor.require_priority(args)
    for name in ('arithmetic1024-root-review-pilot-v1', 'example1024-root-review-pilot-v1', 'active-textbook1024-root-review-pilot-v1', 'active-textbook1024-root-review-pilot-v2', 'evidence-first-textbook-root-review-pilot-v3'):
        folder = args.data.parent/name; done = io.read(folder/'complete.json')
        io.verify({str(folder/n): h for n, h in done['digests'].items()})
        capacity = protocol.strict_records(folder/'capacity.jsonl')
        if not capacity or capacity[-1] != {'min_containers': 0, 'max_containers': 2}:
            raise ValueError('Every prior cloud collector must finish and release capacity')


def registration(args):
    require_priority(args)
    corpus = spec.retrieval.CORPUS; completed = io.read(corpus/'complete.json')
    if completed['benchmark_records_read'] or completed['model_calls']:
        raise ValueError('Require independently built textbook exposition')
    io.verify({str(corpus/n): h for n, h in completed['digests'].items()})
    record = base_registration(args)
    record['source_files'] = io.sources(sys.modules[__name__])
    record['input_evidence'].update({str((corpus/n).resolve()): h for n, h in completed['digests'].items()})
    record['input_evidence'][str((corpus/'complete.json').resolve())] = io.sha(corpus/'complete.json')
    record.update(runner_schema=SCHEMA, minimum_physical_teacher_calls=192, physical_teacher_calls=192, maximum_total_physical_calls=320,
        full_trajectory=['original compact root', '64-token concept query', 'deterministic local textbook retrieval', '960-token reasoning with evidence already visible', '512-token review decision with paid reasoning and retrieved passages',
            '128-token guided compact repair', '128-token retained compact repair'],
        registered_known_union_questions=50, corpus_directory=str(corpus),
        active_retrieval={'query_source': 'Paid64-token public-question-only query; no current answer, labels or future responses',
            'maximum_queries': 2, 'prefilled_opening_search_tag': True, 'literal_placeholders_rejected': True,
            'fallback': 'Public original question stem, capped800 characters, if no valid completed query', 'maximum_generated_query_characters': 200, 'maximum_unique_passages': 3,
            'maximum_words_per_passage': 85, 'minimum_similarity': .06,
            'subject_filter': True, 'model_or_network_calls': 0,
            'question_or_outcome_conditioned_corpus_build': False, 'benchmark_reference_access': False,
            'all_tool_output_tokens_charged': True, 'CPU_time_reported_separately': True,
            'compact_receives_unchanged_answer_and_check_packet': True,
            'matched_B_must_receive_identical_tool_availability': True},
        promotion_requires='Novel correct paths, matched retrieval-capable B, learned POMDP integration and fresh validation')
    record['teacher'].update(always_two_phases=False, always_three_phases=True, query_cap=64, thinking_cap=960, total_output_cap=1536,
        phase1_visibility='Public original question only;64-token concept query',
        phase2_visibility='Original question, paid current solution, retrieved textbook excerpts;960-token reasoning',
        phase3_visibility='Original question, paid current solution, all paid reasoning and textbook excerpts;512-token decision')
    for key in ('retrieval_neural_calls', 'retrieval_CPU_not_in_parameter_token_proxy',
                'questions_with_textbook_context', 'textbook_tokens_charged_in_teacher_input'):
        record.pop(key)
    logical, raw, attempts, full = reused_inputs(args)
    if len(full)>58: raise ValueError('Fewer than six changed trajectories cannot meet the novelty screen')
    record.update(reused_physical_calls=len(raw), reused_logical_nodes=len(logical), fully_reused_questions=len(full), maximum_new_physical_calls=320-len(raw),
        all_reused_tokens_charged_at_inference=True, reuse_identity=original.old.digest({'logical':logical,'raw':raw,'attempts':attempts}),
        corpus_change_only=True, cloud_priority='Complete auxiliary-math512-pairs-v1 and restore min0 before acquisition')
    source = args.data.parent/'evidence-first-textbook-root-review-pilot-v3'
    done = io.read(source/'complete.json')
    record['input_evidence'].update({str((source/k).resolve()):v for k,v in done['digests'].items()})
    record['input_evidence'][str((source/'complete.json').resolve())]=io.sha(source/'complete.json')
    return record


def reused_inputs(args):
    folder=args.data.parent/'evidence-first-textbook-root-review-pilot-v3'
    done=io.read(folder/'complete.json');io.verify({str(folder/k):v for k,v in done['digests'].items()})
    raw=protocol.strict_records(folder/'raw-generations.jsonl');attempts=protocol.strict_records(folder/'attempts.jsonl')
    prior_runner.attempts_checked(attempts,raw)
    lookup={r['request_sha256']:r for r in raw};attempt_lookup={r['request_sha256']:r for r in attempts}
    rows,trees,*_=spec.public_inputs(args);logical=[];used=set();full=[]
    for row in rows:
        tree=trees[row['id']];current=tree['']
        try:teacher,calls=spec.review(row,tree,lookup,used)
        except spec.MissingPhysical:continue
        logical.append(spec.logical(row,tree[''],'judge','',teacher,calls))
        for path in spec.PATHS:
            try:node,calls=spec.solver(row,current,teacher,path,lookup,used)
            except spec.MissingPhysical:break
            logical.append(spec.logical(row,tree[''],'solver',path,node,calls,current,teacher));current=node
        else:full.append(row['id'])
    selected=[r for r in raw if r['request_sha256'] in used]
    recorded=[{**attempt_lookup[r['request_sha256']],'schema':spec.SCHEMA} for r in selected]
    if len(selected)<64 or len(selected)!=len(used):raise ValueError('Every identical paid query must be reused')
    spec.restore(rows,trees,logical,selected)
    return logical,selected,recorded,full


def seed_reused(args):
    record=registration(args)
    if io.read(args.output/'registration.json')!=record or io.sha(args.output/'registration.json')!=args.registration_sha256:
        raise ValueError('Changed broad-corpus registration')
    logical,raw,attempts,full=reused_inputs(args)
    payloads={name:''.join(json.dumps(r,sort_keys=True)+'\n' for r in values).encode()
        for name,values in (('train.jsonl',logical),('raw-generations.jsonl',raw),('attempts.jsonl',attempts))}
    receipt={'schema':SCHEMA,'registration_sha256':args.registration_sha256,'reused_physical_calls':len(raw),
        'reuse_identity':record['reuse_identity'],'prefixes':{name:{'bytes':len(v),'sha256':spec.hashlib.sha256(v).hexdigest()} for name,v in payloads.items()}}
    target=args.output/'reuse-receipt.json'
    with io.lock(args.output/'.reuse.lock'):
        if target.exists():
            if io.read(target)!=receipt:raise ValueError('Changed reuse receipt')
            for name,value in payloads.items():
                if not (args.output/name).read_bytes().startswith(value):raise ValueError('Changed reused raw prefix')
            return
        if any((args.output/name).exists() for name in payloads):raise ValueError('Partial reuse seed requires inspection')
        for name,value in payloads.items():io.atomic_bytes(args.output/name,value)
        io.write(target,receipt)


def collect(args):
    folder=args.data.parent/'auxiliary-math512-pairs-v1';done=io.read(folder/'complete.json')
    if done['coverage']['physical_calls']!=3584 or done['test_used'] or done['scored']:raise ValueError('Math collector must complete first')
    io.verify({str(folder/k):v for k,v in done['digests'].items()})
    capacity=protocol.strict_records(folder/'capacity.jsonl')[-1]
    if capacity['min_containers']!=0 or capacity['max_containers']!=2:raise ValueError('Math collector must release capacity')
    seed_reused(args);original.collect(args)


def attempts_checked(records, raw):
    attempts = {}
    for item in records:
        key = item['request_sha256']
        if (key in attempts or item['schema'] != spec.SCHEMA or item['endpoint'] != '/api/generate'
                or item['payload']['model'] not in (protocol.old.MODEL, protocol.old.JUDGE)
                or item['payload_sha256'] != original.old.digest(item['payload'])
                or key != original.old.digest({'endpoint': item['endpoint'], 'payload': item['payload']})):
            raise ValueError('Invalid or repeated attempted request')
        attempts[key] = item
    completed = {}
    for item in raw:
        original.old.RequestJournal.validate(item); key = item['request_sha256']
        if key in completed or key not in attempts or attempts[key]['payload'] != item['payload']:
            raise ValueError('Physical response lacks its unique durable attempt')
        completed[key] = item
    if len(attempts) > 320 or set(attempts) != set(completed):
        raise ValueError('Attempt budget exceeded or unresolved physical request; no automatic retry')
    return attempts


class Journal(original.old.RequestJournal):
    def __init__(self, request, stream, attempt_stream, records=(), attempts=()):
        self.attempts = attempts_checked(attempts, records); self.attempt_lock = threading.Lock()
        def once(endpoint, payload=None):
            if endpoint == '/api/generate':
                if payload['model'] not in (protocol.old.MODEL, protocol.old.JUDGE): raise ValueError('Undeclared model')
                key = original.old.digest({'endpoint': endpoint, 'payload': payload})
                with self.attempt_lock:
                    if key in self.attempts or len(self.attempts) >= 320: raise ValueError('Duplicate request or exhausted call budget')
                    item = {'schema': spec.SCHEMA, 'endpoint': endpoint, 'payload': payload,
                        'request_sha256': key, 'payload_sha256': original.old.digest(payload)}
                    original.paired.durable_line(attempt_stream, item); self.attempts[key] = item
            return request(endpoint, payload)
        super().__init__(once, stream, records, slots=2)


@contextmanager
def configured_runner():
    before = original.spec, original.NAMES, original.registration, original.require_priority, original.Journal, original.attempts_checked
    original.spec, original.NAMES = spec, NAMES
    original.registration, original.require_priority = registration, require_priority
    original.Journal, original.attempts_checked = Journal, attempts_checked
    try: yield
    finally:
        original.spec, original.NAMES, original.registration, original.require_priority, original.Journal, original.attempts_checked = before


def register(args):
    record = registration(args); args.output.mkdir(parents=True, exist_ok=False)
    io.write(args.output/'registration.json', record)
    print({'registration_sha256': io.sha(args.output/'registration.json'), 'physical_calls': 320, 'reused_physical_calls':record['reused_physical_calls'], 'maximum_new_calls': record['maximum_new_physical_calls'], 'calls_started': 0}, flush=True)


def grade(args):
    args.registration_sha256=io.sha(args.output/'registration.json')
    seed_reused(args)
    bundle, known, done = original.validated_complete(args)
    rows, trees, reviews, ordinary, ordinary_reviews, late, *_ = bundle
    target = args.output/'graded'
    if target.exists(): raise ValueError('Preserve the first completed diagnostic')
    labels = {r['id']: r for r in original.old.fixed.lines(args.data/'splits/train-questions.jsonl')}
    if any(protocol.public_row(labels[r['id']]) != r for r in rows):
        raise ValueError('Reference question differs from paid prompt')
    reports, private, screen = original.aggregate(rows, trees, reviews, ordinary, ordinary_reviews, late, known,
        {r['id']: labels[r['id']]['answer'] for r in rows}, spec.extra_solver_answers(args, rows), expected_known=50)
    tool = [known[r['id'], 'judge', '']['response'] for r in rows]; timing = []
    for row, review in zip(rows, tool):
        text = '<search>'+review['calls'][0]['text']; started = time.perf_counter()
        for _ in range(10): result = spec.retrieval.evaluate_text(text, row['subject'], spec.public(row))
        timing.append((time.perf_counter()-started)*100.)
        if result != review['retrieval_results']: raise ValueError('Retrieval replay differs from acquired evidence')
    original.validated_complete(args); target.mkdir(); io.write(target/'private-outcomes.json', private)
    result = {'schema': spec.SCHEMA, 'runner_schema': SCHEMA, 'reports': reports, 'screen': screen,
        'questions': 64, 'coverage': done['coverage'], 'complete_sha256': io.sha(args.output/'complete.json'),
        'private_outcomes_sha256': io.sha(target/'private-outcomes.json'),
        'training_only': True, 'test_used': False, 'new_neural_calls': 0, 'controller_accuracy_claim': False,
        'matched_B_required_if_promoted': True, 'reused_physical_calls':io.read(args.output/'registration.json')['reused_physical_calls'], 'new_physical_calls':io.read(args.output/'registration.json')['maximum_new_physical_calls'], 'full_inference_costs_charged':True, 'transport_error_records': done['transport_error_records'],
        'acquisition_compute_is_lower_bound': done['acquisition_compute_is_lower_bound'],
        'active_retrieval': {'questions_with_queries': sum(bool(v['retrieval_results']) for v in tool),
            'queries': sum(v['retrieval_query_count'] for v in tool),
            'public_question_fallbacks': sum(v['public_question_fallback'] for v in tool),
            'questions_with_passages': sum(any(r['passages'] for r in v['retrieval_results']) for v in tool),
            'passages': sum(len(r['passages']) for v in tool for r in v['retrieval_results']),
            'mean_replay_cpu_ms': sum(timing)/64, 'max_question_replay_cpu_ms': max(timing),
            'repetitions_per_question': 10, 'included_in_parameter_token_proxy': False}}
    io.write(target/'summary.json', result)
    print({'screen': screen, 'active_retrieval': result['active_retrieval'],
        'methods': {name: reports['all_science']['methods'][name] for name in NAMES}}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'collect', 'grade'))
    for name in ('data', 'mentor', 'complete', 'late', 'output', 'priority'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with predecessor.guarded.training_only_reads(), configured_runner():
        {'register': register, 'collect': collect, 'grade': grade}[args.phase](args)


if __name__ == '__main__': main()
