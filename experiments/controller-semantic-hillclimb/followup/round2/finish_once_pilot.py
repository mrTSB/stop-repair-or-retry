"""Training-only new finish-reasoning action study; no controller promotion.

All128 existing new-mentor questions receive one continuation. Collection
never retains reference answers. Original five fixed controls remain intact.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys
import threading
import time

import science_mentor_expansion as gates
import format_fixed_controls as fixed
import prospective_collect as protocol
from prospective_parallel import RequestJournal, KeyLocks, durable_line, raw_coverage

SCHEMA = 'training-finish-current-reasoning-once-v1'
WORKERS = 2
sha, digest, read, write = fixed.sha, fixed.digest, gates.read, protocol.atomic_json


def prompt(row, root):
    if row.get('answer'):
        raise ValueError('Finish-action inference requires a blank reference')
    question = row['question'].removesuffix(protocol.old.SCIENCE_SUFFIX)
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    return protocol.old.chat(f'{question}\n\nWork so far:\n{root["text"]}\n\n'
        'Continue this solution from its last useful step. Do not restart or repeat the question or completed calculations. '
        'Complete any unfinished calculation or reasoning. If a specific earlier step is inconsistent, '
        'correct that step briefly and continue. '
        f'Give at most 90 words of additional scratch work, then exactly one line "Final: X" where X is {unit}.')


def sources():
    project = Path(__file__).resolve().parents[4]
    here = Path(__file__).resolve().parent
    def allowed(path):
        if path is None:
            return None
        path = Path(path).resolve()
        if (path.is_file() and path.suffix == '.py' and path.is_relative_to(project)
                and not any(part.startswith('.') or part in ('site-packages', '__pycache__')
                            for part in path.relative_to(project).parts)):
            return path
        return None
    def locate(name):
        if not name:
            return None
        if name in sys.modules:
            return allowed(getattr(sys.modules[name], '__file__', None))
        relative = Path(*name.split('.'))
        for folder in dict.fromkeys([str(here), *sys.path]):
            for path in (Path(folder or '.')/relative.with_suffix('.py'), Path(folder or '.')/relative/'__init__.py'):
                if path.is_file():
                    return allowed(path)
        return None
    pending = [(Path(__file__).resolve(), 'finish_once_pilot')]
    found = {Path(path) for path in gates.source_manifest()}
    while pending:
        path, module = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                package = module.split('.')[:-node.level] if node.level else []
                name = '.'.join([*package, *([node.module] if node.module else [])])
                names = [name, *[name+'.'+alias.name for alias in node.names if alias.name != '*']]
            for name in names:
                dependency = locate(name)
                if dependency is not None:
                    pending.append((dependency, name))
    return {str(path): sha(path) for path in sorted(found)}


def plain_controls(data, ids):
    receipt = read(data / 'train-plain-complete.json')
    with (data / 'train-observations.jsonl').open('rb') as stream:
        raw = b''.join(stream.readline() for _ in range(receipt['base_records']))
    if not raw.endswith(b'\n') or fixed.hashlib.sha256(raw).hexdigest() != receipt['base_bank_sha256']:
        raise ValueError('Changed sealed original TRAIN prefix')
    trees = {qid: {} for qid in ids}
    for line in raw.splitlines():
        item = json.loads(line)
        # The original128 cohort already had some ordinary guided/review nodes
        # when the immutable plain prefix was sealed. They remain hash-bound,
        # but are not part of this seven-path fixed-action control grid.
        if item['id'] not in trees or item['kind'] != 'solver' or item['path'] not in protocol.PLAIN:
            continue
        if item['path'] in trees[item['id']]:
            raise ValueError('Duplicate original plain control')
        trees[item['id']][item['path']] = item['response']
    if any(set(tree) != set(protocol.PLAIN) for tree in trees.values()):
        raise ValueError('Incomplete original seven-path grid')
    return trees, receipt['base_bank_sha256']


def inputs(args):
    required = (args.data.parent / 'development-metadata-newsource-v1/metadata-replay-complete.json').resolve()
    if required not in [path.resolve() for path in args.priority_completion]:
        raise ValueError('Require the exact final metadata-newsource C seal before cloud work')
    rows = gates.public_train(args.data)
    parent = read(args.mentor / 'contract.json'); completion = read(args.mentor / 'complete.json')
    if (completion['questions'] != 192 or completion['records'] != 960
            or sha(args.mentor / 'contract.json') != completion['contract_sha256']
            or sha(args.mentor / 'train.jsonl') != completion['bank_sha256']
            or parent['new_question_split_sha256'] != sha(args.data / 'splits/train-questions.jsonl')):
        raise ValueError('Changed completed original mentor cohort')
    chosen = fixed.selection(rows, parent)
    original, prefix_sha = plain_controls(args.data, {r['id'] for r in chosen})
    by_id = {r['id']: r for r in rows}
    for item in chosen:
        row, tree = by_id[item['id']], original[item['id']]
        for path, node in tree.items():
            protocol.validate_node(row, 'plain', 'solver', path, node, tree, {})
    evidence = {str((args.mentor / name).resolve()): sha(args.mentor / name)
                for name in ('contract.json', 'complete.json', 'train.jsonl')}
    evidence[str((args.data / 'splits/train-questions.jsonl').resolve())] = sha(args.data / 'splits/train-questions.jsonl')
    return [by_id[r['id']] for r in chosen], original, chosen, prefix_sha, required, evidence


def registration(args):
    rows, original, chosen, prefix_sha, required, evidence = inputs(args)
    return {'schema': SCHEMA, 'source_files': sources(), 'input_evidence': evidence,
        'selection': chosen, 'subject_counts': fixed.COUNTS, 'questions': 128,
        'train_question_sha256': sha(args.data / 'splits/train-questions.jsonl'),
        'matched_seven_path_controls_sha256': digest(original), 'sealed_plain_prefix_sha256': prefix_sha,
        'prompt_sha256': {r['id']: fixed.hashlib.sha256(prompt(r, original[r['id']]['']).encode()).hexdigest() for r in rows},
        'solver_digest': protocol.old.MODEL_DIGEST, 'reviewer_digest': protocol.old.JUDGE_DIGEST,
        'temperature': .65, 'output_cap': 256, 'context_length': 4096,
        'seed': 'Exact original logical repair seed; no seed search',
        'new_calls': 128, 'root_reused': True, 'review_calls': 0, 'neural_calls': 0,
        'additional_solver_slots_used': 1, 'maximum_additional_solver_slots': 2,
        'workers': WORKERS, 'max_cloud_gpus': 2,
        'required_final_metadata_completion': str(required),
        'priority_completions': [str(p.resolve()) for p in args.priority_completion],
        'paths': {name: str(getattr(args, name).resolve()) for name in ('data', 'mentor', 'b_reference', 'output')},
        'scale_only_if': {'net_correct_above_stronger_original_one_step': 7,
            'mean_compute_at_most_original_repair_once': True,
            'new_correct_cases_beyond_original_seven_path_oracle': 4},
        'fixed_heuristics': ['valid root -> Stop; invalid root -> Finish', 'valid root -> Stop; invalid root -> Repair',
                             'valid root -> Stop; invalid root -> Retry'],
        'validity': 'Boolean frozen parsed root answer; never correctness/confidence',
        'training_only': True, 'test_used': False, 'references_supplied': False,
        'score_only_after': 'Complete hash-sealed all128 response/raw grid',
        'scope': 'NEW finish-reasoning action/protocol, not a POMDP/gate improvement or unchanged benchmark result',
        'promotion_requires': 'Matched controls and full baseline fairness, updated POMDP action model and fresh prospective validation; no hindsight promotion'}


def register(args):
    if args.output.exists():
        raise ValueError('Use a new Finish-once experiment area')
    record = registration(args); args.output.mkdir(parents=True)
    write(args.output / 'registration.json', record)
    print(json.dumps({'registered': True, 'registration_sha256': sha(args.output / 'registration.json'),
                      'new_solver_calls': 128, 'inference_started': False}), flush=True)


def validate_node(row, root, node):
    expected = (protocol.old.MODEL, protocol.old.MODEL_DIGEST,
        fixed.hashlib.sha256(prompt(row, root).encode()).hexdigest(), fixed.logical_seed(row['id'], 'repair'), .65, 256, 4096)
    actual = tuple(node[k] for k in ('model', 'model_digest', 'prompt_sha256', 'seed', 'temperature', 'output_cap', 'context_length'))
    if expected != actual or node['answer'] != protocol.old.parse_answer(row['domain'], node['text']):
        raise ValueError('Changed Finish prompt/seed/model/parser/decoding receipt')
    if node['input_tokens'] <= 0 or not 0 <= node['output_tokens'] <= 256:
        raise ValueError('Invalid completed Finish physical token receipt')


def restore(rows, original, records, raw):
    by_id = {r['id']: r for r in rows}; known = {}
    for item in records:
        qid = item['id']
        if qid not in by_id or qid in known or (item['kind'], item['path']) != ('solver', 'repair'):
            raise ValueError('Duplicate or foreign Finish response')
        if item['protocol'] != SCHEMA or item['root_sha256'] != digest(original[qid]['']):
            raise ValueError('Changed paid root prefix')
        validate_node(by_id[qid], original[qid][''], item['response']); known[qid] = item
    coverage = raw_coverage({'plain': records}, raw)
    return known, coverage


class Collector:
    def __init__(self, original, known, stream, request):
        self.original, self.known, self.stream, self.request = original, known, stream, request
        self.lock, self.keys, self.stopped = threading.RLock(), KeyLocks(), threading.Event()

    def one(self, row):
        try:
            if row.get('answer') != '':
                raise ValueError('Reference reached Finish collection')
            with self.keys.get(row['id']):
                if self.stopped.is_set() or row['id'] in self.known:
                    return
                root = self.original[row['id']]['']
                node = protocol.actor(self.request, row, prompt(row, root), fixed.logical_seed(row['id'], 'repair'), 256)
                validate_node(row, root, node)
                item = {'id': row['id'], 'kind': 'solver', 'path': 'repair', 'protocol': SCHEMA,
                        'root_sha256': digest(root), 'response': node}
                with self.lock:
                    durable_line(self.stream, item); self.known[row['id']] = item
        except BaseException:
            self.stopped.set(); raise


def collect(args):
    path = args.output / 'registration.json'
    if sha(path) != args.registration_sha256 or read(path) != registration(args):
        raise ValueError('Externally recorded Finish registration changed')
    record = read(path)
    with ExitStack() as stack:
        lockfile = stack.enter_context((args.output / 'collection.lock').open('a'))
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'complete.json').exists():
            raise ValueError('Finish pilot already complete')
        previous = None
        while True:
            gates.verify_sources(record)
            receipts, missing = gates.prerequisite_receipts(args, record)
            if not missing:
                break
            if not args.wait:
                raise ValueError('Core comparisons must finish first: ' + json.dumps(missing))
            if missing != previous:
                print(json.dumps({'waiting_for': missing, 'physical_calls_started': False}), flush=True); previous = missing
            time.sleep(30)
        receipt_path = args.output / 'prerequisites.json'
        if receipt_path.exists() and read(receipt_path) != receipts:
            raise ValueError('Previously completed prerequisite changed')
        write(receipt_path, receipts)
        rows, original, *_ = inputs(args)
        bank, raw_path = args.output / 'train.jsonl', args.output / 'raw-generations.jsonl'
        stream = stack.enter_context(bank.open('a')); raw_stream = stack.enter_context(raw_path.open('a'))
        for handle in (stream, raw_stream):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records, raw = protocol.strict_records(bank), protocol.strict_records(raw_path)
        known, _ = restore(rows, original, records, raw)
        import modal
        worker = modal.Cls.from_name(protocol.APP, 'Ollama')()
        worker.request.autoscaler.update(max_containers=2)
        request = RequestJournal(protocol.resilient(worker, args.output / 'transport-errors.jsonl'), raw_stream, raw, slots=WORKERS)
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        if tags.get(protocol.old.MODEL) != record['solver_digest']:
            raise ValueError('Deployed compact weights changed')
        collector = Collector(original, known, stream, request)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(collector.one, rows))
        saved, returned = protocol.strict_records(bank), protocol.strict_records(raw_path)
        known, coverage = restore(rows, original, saved, returned)
        if set(known) != {r['id'] for r in rows} or len(saved) != 128 or len(returned) != 128:
            raise ValueError('Require the entire128-question physical/logical grid')
        gates.verify_sources(record)
        now, missing = gates.prerequisite_receipts(args, record)
        if missing or now != receipts or record != registration(args):
            raise ValueError('Registered inputs or prerequisites changed during collection')
        write(args.output / 'complete.json', {'schema': SCHEMA, 'questions': 128,
            'registration_sha256': sha(path), 'bank_sha256': sha(bank), 'raw_sha256': sha(raw_path),
            'prerequisites_sha256': sha(receipt_path), 'coverage': coverage, 'physical_solver_calls': 128,
            'transport_errors_sha256': sha(args.output / 'transport-errors.jsonl') if (args.output / 'transport-errors.jsonl').exists() else None,
            'new_review_calls': 0, 'new_neural_calls': 0, 'training_only': True, 'scored': False})
        print('Finish-once128 complete; no accuracy scored', flush=True)


def grade(args):
    complete = read(args.output / 'complete.json')
    if complete['questions'] != 128 or complete['physical_solver_calls'] != 128:
        raise ValueError('Do not grade partial Finish cohorts')
    for file, key in (('registration.json', 'registration_sha256'), ('train.jsonl', 'bank_sha256'),
                       ('raw-generations.jsonl', 'raw_sha256'), ('prerequisites.json', 'prerequisites_sha256')):
        if sha(args.output / file) != complete[key]:
            raise ValueError('Changed complete Finish bank')
    errors = args.output / 'transport-errors.jsonl'
    if (sha(errors) if errors.exists() else None) != complete['transport_errors_sha256']:
        raise ValueError('Changed sealed transport accounting')
    contract = read(args.output / 'registration.json')
    if contract != registration(args):
        raise ValueError('Changed registered Finish protocol')
    public, original, *_ = inputs(args)
    known, coverage = restore(public, original, protocol.strict_records(args.output / 'train.jsonl'),
                              protocol.strict_records(args.output / 'raw-generations.jsonl'))
    if len(known) != 128 or coverage != complete['coverage']:
        raise ValueError('Incomplete Finish coverage')
    # First retained reference read: all physical and logical receipts are sealed.
    references = {r['id']: r for r in fixed.lines(args.data / 'splits/train-questions.jsonl')}
    rows = [references[r['id']] for r in public]
    from scoring import correct
    methods = {name: paths for name, paths in fixed.FIXED.items()}
    methods.update({'Finish once': ('', 'finish'), 'Valid-stop else Finish': None,
                    'Valid-stop else Repair': None, 'Valid-stop else Retry': None})
    trees = {r['id']: {**original[r['id']], 'finish': known[r['id']]['response']} for r in rows}
    def evaluate(group, method, paths):
        outcomes, tokens, calls = [], [], []
        for row in group:
            selected = paths
            if selected is None:
                edge = {'Valid-stop else Finish': 'finish', 'Valid-stop else Repair': 'repair', 'Valid-stop else Retry': 'retry'}[method]
                selected = ('',) if trees[row['id']]['']['answer'] else ('', edge)
            nodes = [trees[row['id']][p] for p in selected]
            outcomes.append(bool(correct(nodes[-1], row['answer'])))
            tokens.append(sum(n['input_tokens'] + n['output_tokens'] for n in nodes)); calls.append(len(nodes))
        return {'n': len(group), 'correct': sum(outcomes), 'accuracy': sum(outcomes)/len(group),
                'mean_tokens': sum(tokens)/len(group), 'mean_compute': 4*sum(tokens)/(3.09*len(group)),
                'mean_solver_calls': sum(calls)/len(group), 'review_calls': 0, 'neural_calls': 0}, outcomes
    panels = [('all', rows), *[(s, [r for r in rows if r['subject'] == s]) for s in fixed.COUNTS],
        ('root_valid', [r for r in rows if trees[r['id']]['']['answer']]),
        ('root_invalid', [r for r in rows if not trees[r['id']]['']['answer']])]
    results = {}
    for label, group in panels:
        if not group:
            results[label] = {'n': 0}; continue
        summaries, outcomes = {}, {}
        for name, paths in methods.items():
            summaries[name], outcomes[name] = evaluate(group, name, paths)
        pairs = {}
        for left, right in (('Finish once', 'Repair once'), ('Finish once', 'Retry once'),
                            ('Valid-stop else Finish', 'Valid-stop else Repair'), ('Valid-stop else Finish', 'Valid-stop else Retry')):
            a, b = outcomes[left], outcomes[right]
            pairs[left + ' vs ' + right] = {'wins': sum(x and not y for x, y in zip(a, b)),
                'losses': sum(y and not x for x, y in zip(a, b)), 'net_correct': sum(a)-sum(b),
                'mean_compute_change': summaries[left]['mean_compute']-summaries[right]['mean_compute']}
        novel = sum(outcomes['Finish once'][i] and not any(correct(original[r['id']][p], r['answer']) for p in protocol.PLAIN)
                    for i, r in enumerate(group))
        results[label] = {'n': len(group), 'methods': summaries, 'paired': pairs,
                         'new_correct_beyond_original_seven_path_oracle': novel}
    all_methods = results['all']['methods']; finish = all_methods['Finish once']
    strongest = max(all_methods[name]['correct'] for name in ('Repair once', 'Retry once'))
    conditions = {'net_gain_at_least7': finish['correct'] - strongest >= 7,
        'compute_no_greater_than_repair_once': finish['mean_compute'] <= all_methods['Repair once']['mean_compute'],
        'at_least4_new_correct_beyond_original_seven_path_oracle': results['all']['new_correct_beyond_original_seven_path_oracle'] >= 4}
    physical = [v['response'] for v in known.values()]
    summary = {'schema': SCHEMA, 'complete_sha256': sha(args.output / 'complete.json'), 'panels': results,
        'scaling_conditions': conditions, 'scaling_threshold_met': all(conditions.values()),
        'training_only': True, 'test_used': False, 'no_controller_result': True, 'scope': contract['scope'],
        'future_promotion_requirements': contract['promotion_requires'],
        'acquisition': {'new_solver_calls': 128, 'new_solver_tokens': sum(n['input_tokens']+n['output_tokens'] for n in physical),
            'transport_attempts_without_usable_receipt': len(protocol.strict_records(args.output / 'transport-errors.jsonl')),
            'unknown_transport_compute_included': False},
        'format': {'root_valid': sum(bool(original[r['id']]['']['answer']) for r in rows),
            'finish_valid': sum(bool(n['answer']) for n in physical),
            'finish_at_cap': sum(n['output_tokens'] >= 256 for n in physical)},
        'compute_unit': '4B*(all paid root and continuation input+output tokens)/3.09B; reused roots are charged'}
    if (args.output / 'summary.json').exists():
        raise ValueError('Never overwrite a completed Finish score')
    write(args.output / 'summary.json', summary)
    print(json.dumps({'complete': True, 'scaling_conditions': conditions, 'training_only': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'collect', 'grade'))
    for name in ('data', 'mentor', 'b-reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--priority-completion', type=Path, action='append', required=True)
    parser.add_argument('--registration-sha256')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    {'register': register, 'collect': collect, 'grade': grade}[args.phase](args)
