"""Explicit metadata-C development replay with sealed, whole-bank inheritance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import pickle
import shutil
import threading
import time

import numpy as np

import fit_metadata_review as fitting
import metadata_review_hybrid as policy
import replay_new_source_mentor as inherited
import development_mixed_mentor as study
from prospective_parallel import RequestJournal, durable_line, raw_coverage
from prospective_collect import strict_records
from prepare_mixed_mentor import sha

SCHEMA = 'metadata-review-development-v1'
PENALTIES = [0., .02, .04, .08, .12, .18, .24]
THRESHOLDS = [0., .03, .08]
UTILITIES = [0., .02, .04, .08, .16]
read, write = inherited.read, inherited.write


def atomic_pickle(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=5); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def register(args):
    expected = {'schema': SCHEMA, 'source_files': fitting.source_files(),
        'policy_module': 'metadata_review_hybrid', 'policy_sha256': sha(policy.__file__),
        'paths': {name: str(getattr(args, name).resolve()) for name in (
            'data', 'models', 'responses', 'cache', 'cache_models', 'b_reference', 'output')},
        'device': args.device, 'dev_questions_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
        'development_questions': 256, 'penalties': PENALTIES, 'thresholds': THRESHOLDS,
        'subject_utility_penalties': UTILITIES, 'smoothing': [8., 32.],
        'acquisition': 'Only genuinely missing paid nodes; exact durable physical request reuse; no completed-answer regeneration',
        'maximum_cloud_requests': 2, 'test_used': False, 'final_gap_claim': False}
    path = args.output / 'registration.json'
    if path.exists():
        recorded = read(path)
        if recorded != expected:
            raise ValueError('Changed registered metadata development recipe; use a new area')
    else:
        if any(p.name != '.replay.lock' for p in args.output.iterdir()):
            raise ValueError('Inspect unregistered output before any overwrite')
        write(path, expected)
    return expected


def wait_completed(directory):
    inherited.wait_completed(directory)
    imported = directory / 'cache-import.json'
    if imported.exists() and read(imported).get('schema') == SCHEMA:
        inherited.wait(directory / 'metadata-replay-complete.json')


def seal_payload(directory):
    optional = ['transport-errors.jsonl'] if (directory / 'transport-errors.jsonl').exists() else []
    return {'schema': SCHEMA, 'source_sha256': sha(__file__),
        'files': {name: sha(directory / name) for name in (
            'registration.json', 'identity.json', 'cache-import.json', 'summary.json', 'search.json',
            'dev-observations.jsonl', 'private-neural-cache.pkl', 'raw-generations.jsonl', *optional)}}


def completed_bank(directory, panel):
    payload, receipt = inherited.completed_bank(directory, panel)
    imported = directory / 'cache-import.json'
    if imported.exists() and read(imported).get('schema') == SCHEMA:
        marker = directory / 'metadata-replay-complete.json'
        if not marker.exists() or read(marker) != seal_payload(directory):
            raise ValueError('Metadata replay has no valid final source/cache/summary seal')
        receipt['metadata_seal_sha256'] = sha(marker)
    return payload, receipt


def model_assets(directory):
    complete = read(directory / 'complete.json')
    if complete.get('policy_module') != 'metadata_review_hybrid':
        return inherited.model_assets(directory)
    identity = read(directory / 'identity.json')
    contract = read(directory / 'registration.json')
    if (complete['identity_sha256'] != sha(directory / 'identity.json')
            or complete['registration_sha256'] != sha(directory / 'registration.json')
            or identity['registration_sha256'] != complete['registration_sha256']
            or complete['development_or_test_used'] is not False):
        raise ValueError('Changed registered metadata fit')
    fitting.verify_hashes(contract['source_files'])
    for name, digest in complete['artifacts'].items():
        if sha(directory / name) != digest:
            raise ValueError('Changed metadata coefficients or OOF alignment')
    if [(r['smoothing'], r['metadata_review']) for r in complete['models']] != [(8., True), (32., True)]:
        raise ValueError('Require exactly two metadata smoothing variants')
    recipes = set()
    for spec in complete['models']:
        path = directory / (spec['key'] + '.pkl')
        if sha(path) != spec['sha256']:
            raise ValueError('Changed metadata candidate pickle')
        model = pickle.loads(path.read_bytes())
        if not isinstance(model, policy.Planner) or model.fit_receipt['policy_module'] != 'metadata_review_hybrid':
            raise ValueError('Do not dispatch metadata C through an ordinary semantic evaluator')
        model.metadata_adapter.validate()
        if model.review_observer is not None or model.mentor.review_observer is not None:
            raise ValueError('Metadata replacement must not encode the review with a neural model')
        for observer in (model.base.classifier, model.mentor.classifier):
            if observer._model is not None:
                raise ValueError('Use archived observer assets without resident models')
            recipe_path = Path(observer.directory) / 'recipe.json'; recipe = sha(recipe_path)
            recipes.add(recipe)
            if recipe not in model.fit_receipt['observer_recipe_sha256']:
                raise ValueError('Changed registered raw observer recipe')
            fitting.verify_hashes({str(Path(observer.directory) / name): digest
                                  for name, digest in read(recipe_path)['weights'].items()})
    if len(recipes) != 1:
        raise ValueError('This replay requires identical old-V1 student/child observation assets')
    return complete, next(iter(recipes))


def import_prefix(output, responses, cache, raw, receipt):
    """Copy everything, preserving already acquired suffixes on every resume."""
    paths = {name: output / name for name in ('cache-import.json', 'dev-observations.jsonl',
             'imported-neural-cache.pkl', 'private-neural-cache.pkl', 'raw-generations.jsonl')}
    imported = inherited.cache_values(cache)
    for payload in (responses, raw):
        if payload and not payload.endswith(b'\n'):
            raise ValueError('Partial source receipt tail')
    if paths['cache-import.json'].exists():
        if read(paths['cache-import.json']) != receipt:
            raise ValueError('Changed immutable bank/cache import')
        for name, prefix in (('dev-observations.jsonl', responses), ('raw-generations.jsonl', raw)):
            current = paths[name].read_bytes()
            if (current and not current.endswith(b'\n')) or not current.startswith(prefix):
                raise ValueError('Changed or partial imported/resumed receipt bank')
        if sha(paths['imported-neural-cache.pkl']) != receipt['neural_cache_sha256']:
            raise ValueError('Changed immutable raw neural import')
        if not paths['private-neural-cache.pkl'].exists():
            shutil.copyfile(paths['imported-neural-cache.pkl'], paths['private-neural-cache.pkl'])
        resumed = inherited.cache_values(paths['private-neural-cache.pkl'].read_bytes())
        if any(key not in resumed or any(not np.array_equal(x, y) for x, y in zip(value, resumed[key]))
               for key, value in imported.items()):
            raise ValueError('Resumed cache changed a prior raw forward observation')
        return
    if any(p.name not in ('.replay.lock', 'registration.json') for p in output.iterdir()):
        raise ValueError('Inspect incomplete unregistered import; never overwrite it')
    paths['dev-observations.jsonl'].write_bytes(responses)
    paths['imported-neural-cache.pkl'].write_bytes(cache)
    paths['private-neural-cache.pkl'].write_bytes(cache)
    paths['raw-generations.jsonl'].write_bytes(raw)
    write(paths['cache-import.json'], receipt)


def validate_nodes(rows, trees, judges, records):
    """Strengthen frozen bank validation with phase counts, context and exact retry reuse."""
    by_id = {row['id']: row for row in rows}
    for record in records:
        qid, kind, path, node = record['id'], record['kind'], record['path'], record['response']
        safe = {**by_id[qid], 'answer': ''}
        if kind == 'judge':
            inherited.control.validate_review_charge(node)
            if node['model_digest'] != study.old.JUDGE_DIGEST:
                raise ValueError('Changed strong reviewer weights')
            for index, phase in enumerate(node['calls']):
                if (phase['temperature'], phase['context_length'], phase['output_cap']) != (0, 8192, 1024 if index == 0 else 512):
                    raise ValueError('Changed physical strong-review protocol')
                if phase['input_tokens'] <= 0 or not 0 <= phase['output_tokens'] <= phase['output_cap']:
                    raise ValueError('Invalid strong physical token charge')
            study.mentor_pairs.verify_review(safe, trees[qid][''], node, 1024, 512)
        else:
            if (node['temperature'], node.get('context_length', 4096)) != (.65, 4096):
                raise ValueError('Changed compact decoding/context')
            if path == 'guided/retry' and node != trees[qid]['repair/retry']:
                raise ValueError('Pure retry must exactly reuse the original paired control')
            if node['input_tokens'] <= 0 or node['output_tokens'] < 0:
                raise ValueError('Invalid compact token charge')


def coverage(records, raw):
    # Older imported banks have already validated physical phase receipts but
    # no raw API journal. Every newly acquired node carries a durable raw link.
    selected = [r for r in records if r.get('metadata_acquired') and not r.get('exact_control_reused')]
    return raw_coverage({'strong': selected}, raw)


class Acquisition:
    def __init__(self, rows, trees, judges, stream, journal, cache_save):
        self.rows, self.trees, self.judges = rows, trees, judges
        self.stream, self.journal, self.cache_save = stream, journal, cache_save
        self.lock, self.keys, self.failed, self.added = threading.RLock(), {}, {}, 0

    def acquire(self, row, missing):
        qid, kind, path = row['id'], missing.kind, missing.path
        with self.lock:
            keylock = self.keys.setdefault((qid, kind, path), threading.RLock())
        with keylock:
            key = qid, kind, path
            if key in self.failed:
                raise RuntimeError('Prior acquisition failed; inspect durable receipts before resume') from self.failed[key]
            tree = self.trees[qid]
            if (kind == 'judge' and (qid, path) in self.judges) or (kind == 'solver' and path in tree):
                return
            safe = {**row, 'answer': ''}; reused = False
            try:
                if kind == 'solver' and path == 'guided/retry':
                    node = tree['repair/retry']
                    if node['seed'] != study.mentor_pairs.seed_for(qid, path):
                        raise ValueError('Shared retry seed differs')
                    reused = True
                elif kind == 'judge' and path == '':
                    node = study.canonical_review(study.review(self.journal, safe, tree[''], 1024, 512))
                elif kind == 'solver' and path in ('guided', 'guided/reuse', 'retry/reuse'):
                    parent = '' if path == 'guided' else path.rsplit('/', 1)[0]
                    if parent not in tree or (qid, '') not in self.judges:
                        raise ValueError('Missing paid parent/review prefix')
                    node = study.actor(self.journal, safe, study.short_repair(safe, tree[parent], self.judges[qid, '']),
                                       study.mentor_pairs.seed_for(qid, path), 128)
                else:
                    raise ValueError('Unexpected acquisition outside the fixed hybrid protocol')
                record = {'id': qid, 'kind': kind, 'path': path, 'response': node,
                          'metadata_acquired': True, 'exact_control_reused': reused}
                # Raw API return is already durable. Validate before publication;
                # any completed invalid physical receipt remains in the journal.
                validate_nodes(self.rows, self.trees, self.judges, [record])
                if kind == 'solver' and not reused:
                    expected = hashlib.sha256(study.short_repair(safe, tree[parent], self.judges[qid, '']).encode()).hexdigest()
                    if (node['prompt_sha256'], node['seed'], node['model_digest'], node['output_cap']) != (
                            expected, study.mentor_pairs.seed_for(qid, path), study.old.MODEL_DIGEST, 128):
                        raise ValueError('Changed acquired solver prompt or weights')
                    if node['output_tokens'] > 128:
                        raise ValueError('Acquired compact answer exceeded cap')
                with self.lock:
                    durable_line(self.stream, record)
                    if kind == 'judge':
                        self.judges[qid, path] = node
                    else:
                        tree[path] = node
                    self.added += 1
            except BaseException as error:
                self.failed[key] = error
                raise


def evaluate(model, settings, rows, trees, judges, acquire, policy_lock):
    def one(row):
        while True:
            try:
                with policy_lock:
                    return policy.evaluate(row, trees[row['id']], judges, model, settings, 'C')
            except study.MissingObservation as missing:
                acquire.acquire(row, missing)
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(one, rows))


def measured(items):
    result = study.statistics(items)
    result.update(mean_metadata_calls=float(np.mean([r['metadata_calls'] for r in items])),
                  mean_metadata_cpu_ms=float(1000 * np.mean([r['metadata_cpu_seconds'] for r in items])),
                  mean_judge_physical_calls=float(np.mean([r['judge_physical_calls'] for r in items])),
                  mean_review_neural_calls=float(np.mean([r['review_neural_calls'] for r in items])))
    return result


def finalize(output):
    marker = output / 'metadata-replay-complete.json'
    expected = seal_payload(output)
    if marker.exists():
        if read(marker) != expected:
            raise ValueError('Changed final metadata summary/cache seal')
    else:
        write(marker, expected)


def run_locked(args):
    registration = register(args)
    for source in (args.responses, args.cache):
        wait_completed(source)
    inherited.wait(args.b_reference); inherited.wait(args.models / 'complete.json')
    fitting.verify_hashes(registration['source_files'])
    panel = registration['dev_questions_sha256']
    if sha(args.data / 'splits/dev-questions.jsonl') != panel:
        raise ValueError('Development panel changed while waiting')
    responses, response_receipt = completed_bank(args.responses, panel)
    _, cache_receipt = completed_bank(args.cache, panel)
    fitted, recipe = model_assets(args.models)
    _, cache_recipe = model_assets(args.cache_models)
    cache_identity = read(args.cache / 'identity.json')
    if cache_identity['models_sha256'] != sha(args.cache_models / 'complete.json') or recipe != cache_recipe:
        raise ValueError('Raw neural cache belongs to different models/observation assets')
    allowed = sha(__file__) if cache_identity.get('policy_module') == 'metadata_review_hybrid' else sha(study.__file__)
    if cache_identity['source_sha256'] != allowed:
        raise ValueError('Unrecognized source neural-cache evaluator')
    b = read(args.b_reference)
    inherited.validate_b(b, read(args.b_reference.parent / 'identity.json'), panel)
    cache_bytes = (args.cache / 'private-neural-cache.pkl').read_bytes()
    inherited.cache_values(cache_bytes)
    raw_source = args.responses / 'raw-generations.jsonl'
    raw_bytes = raw_source.read_bytes() if raw_source.exists() else b''
    receipt = {'schema': SCHEMA, 'source_sha256': sha(__file__), 'policy_sha256': sha(policy.__file__),
        'response_source': response_receipt, 'neural_source': cache_receipt,
        'model_complete_sha256': sha(args.models / 'complete.json'),
        'cache_model_complete_sha256': sha(args.cache_models / 'complete.json'),
        'neural_recipe_sha256': recipe, 'neural_cache_sha256': hashlib.sha256(cache_bytes).hexdigest(),
        'raw_import_sha256': hashlib.sha256(raw_bytes).hexdigest(),
        'B_reference_sha256': sha(args.b_reference), 'dev_questions_sha256': panel,
        'selection_by_correctness': False, 'test_used': False, 'final_gap_claim': False}
    import_prefix(args.output, responses, cache_bytes, raw_bytes, receipt)
    identity = {'schema': SCHEMA, 'source_sha256': sha(__file__), 'policy_module': 'metadata_review_hybrid',
        'hybrid_sha256': sha(policy.__file__), 'models_sha256': sha(args.models / 'complete.json'),
        'models_directory': str(args.models.resolve()), 'dev_questions_sha256': panel,
        'ordinary_plain_completion_sha256': sha(args.data / 'dev-plain-complete.json'),
        'registration_sha256': sha(args.output / 'registration.json'),
        'cache_import_sha256': sha(args.output / 'cache-import.json'), 'penalties': PENALTIES, 'thresholds': THRESHOLDS,
        'subject_utility_penalties': UTILITIES, 'test_used': False, 'include_reviews': False}
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Changed resumed metadata-development identity')
    write(identity_path, identity)
    if (args.output / 'summary.json').exists():
        finalize(args.output); completed_bank(args.output, panel)
        print('Verified completed metadata replay; no inference repeated.', flush=True)
        return
    bank = args.output / 'dev-observations.jsonl'; cache_path = args.output / 'private-neural-cache.pkl'
    policy_lock = threading.RLock(); client_lock = threading.Lock(); client = None
    def request(endpoint, payload=None):
        nonlocal client
        with client_lock:
            if client is None:
                import modal
                candidate = study.resilient(modal.Cls.from_name(study.APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
                tags = {r['name']: r['digest'] for r in candidate('/api/tags')['models']}
                if tags.get(study.old.MODEL) != study.old.MODEL_DIGEST or tags.get(study.old.JUDGE) != study.old.JUDGE_DIGEST:
                    raise ValueError('Changed compact solver or reviewer')
                client = candidate
        return client(endpoint, payload)
    with ExitStack() as stack:
        streams = {}
        for name in ('dev-observations.jsonl', 'raw-generations.jsonl'):
            stream = stack.enter_context((args.output / name).open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); streams[name] = stream
        records, raw_records = strict_records(bank), strict_records(args.output / 'raw-generations.jsonl')
        rows, trees, judges = study.development_view(args.data, bank)
        if len(rows) != 256:
            raise ValueError('Require the full registered256development panel')
        validate_nodes(rows, trees, judges, records); coverage(records, raw_records)
        journal = RequestJournal(request, streams['raw-generations.jsonl'], raw_records, slots=2)
        cache = inherited.cache_values(cache_path.read_bytes())
        acquisition = Acquisition(rows, trees, judges, streams['dev-observations.jsonl'], journal, None)
        candidates, observer = [], None
        for spec in fitted['models']:
            path = args.models / (spec['key'] + '.pkl')
            if sha(path) != spec['sha256']:
                raise ValueError('Changed fitted metadata candidate')
            model = pickle.loads(path.read_bytes())
            for leaf in (model.base.classifier, model.mentor.classifier):
                leaf.device = args.device; leaf._cache = cache
            if observer is not None:
                for name in ('_model', '_tokenizer', '_recipe', '_cache', '_retriever', '_backgrounds'):
                    setattr(model.base.classifier, name, getattr(observer, name))
            observer = model.base.classifier
            groups = {subject: [] for subject in ('math', 'chemistry', 'physics')}
            def result(settings):
                items = evaluate(model, settings, rows, trees, judges, acquisition, policy_lock)
                return items, {'model': spec['key'], 'model_path': str(path.resolve()),
                    'policy_module': 'metadata_review_hybrid', 'settings': settings, 'dev': measured(items)}
            try:
                for penalty, threshold in product(PENALTIES, THRESHOLDS):
                    settings = {'penalty': penalty, 'threshold': threshold}
                    items, candidate = result(settings); candidates.append(candidate)
                    for subject in groups:
                        groups[subject].append({'settings': settings, 'dev': measured(
                            [item for row, item in zip(rows, items) if row['subject'] == subject])})
                for utility in UTILITIES:
                    settings = {'by_subject': {subject: max(group, key=lambda r: (
                        r['dev']['accuracy'] - utility * r['dev']['mean_compute'] / 1000,
                        r['dev']['accuracy'], -r['dev']['mean_compute']))['settings'] for subject, group in groups.items()}}
                    _, candidate = result(settings); candidate['selection_utility_penalty'] = utility; candidates.append(candidate)
            finally:
                atomic_pickle(cache_path, observer._cache)
            write(args.output / 'search.json', candidates)
            print('Completed metadata C candidate', spec['key'], flush=True)
        final_records, final_raw = strict_records(bank), strict_records(args.output / 'raw-generations.jsonl')
        raw_receipt = coverage(final_records, final_raw)
        baselines = {name: study.statistics([study.fixed(row, trees[row['id']], *parameters) for row in rows])
                     for name, parameters in study.BASELINES.items()}
        if any(baselines[name] != b['baselines'][name] for name in baselines):
            raise ValueError('Original fixed baseline results changed across the shared panel')
        strongest_b = b['B']; chosen = study.select(candidates, strongest_b['dev']['mean_compute'])
        summary = {'development_questions': 256, 'baselines': baselines, **chosen,
            'A': fitted['base_A'], 'B': strongest_b, 'ordinary_B': b['ordinary_B'], 'matched_B': b['matched_B'],
            'B_reference_sha256': sha(args.b_reference), 'bank_sha256': sha(bank),
            'neural_cache_sha256': sha(cache_path), 'raw_generations_sha256': sha(args.output / 'raw-generations.jsonl'),
            'raw_coverage': raw_receipt, 'new_development_nodes': acquisition.added,
            'new_physical_generations': len(journal.known) - len(raw_records),
            'policy_module': 'metadata_review_hybrid', 'source_sha256': sha(__file__),
            'B_comparison': 'Maximum accuracy ordinary/matched B; cheaper mean compute breaks an accuracy tie.',
            'A_reference_scope': 'Inherited A8 only; strongest global A comparison remains pending.',
            'final_gap_claim': False, 'test_used': False,
            'cost_accounting': 'Every policy pays its own solver/review and regular-child neural observations; no review neural pass; metadata CPU timing separate.'}
        selected = chosen['selected_C']
        if selected is not None:
            higher = max(strongest_b['dev']['accuracy'], fitted['base_A']['dev']['accuracy'])
            summary.update(C_above_inherited_A_or_B=selected['dev']['accuracy'] - higher,
                C_plus_10_points_inherited_reference_only=selected['dev']['accuracy'] >= higher + .1 - 1e-12,
                C_compute_saved_vs_B=1. - selected['dev']['mean_compute'] / strongest_b['dev']['mean_compute'])
        fitting.verify_hashes(registration['source_files'])
        if sha(args.b_reference) != receipt['B_reference_sha256'] or sha(args.models / 'complete.json') != receipt['model_complete_sha256']:
            raise ValueError('Registered model/B reference changed during development')
        write(args.output / 'summary.json', summary)
    finalize(args.output); completed_bank(args.output, panel)
    print(json.dumps({'complete': True, 'best': chosen['best'], 'selected_C': chosen['selected_C'],
                      'new_physical_generations': summary['new_physical_generations']}), flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.replay.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'models', 'responses', 'cache', 'cache-models', 'b-reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    run(parser.parse_args())
