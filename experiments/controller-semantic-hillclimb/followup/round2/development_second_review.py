"""Development comparison with a capability-matched LLM-only second-review B.

The stronger existing B family is always retained. Both new policies may buy
the same critical review only after guided first continuation on science.
"""

import argparse
import fcntl
import hashlib
from itertools import product
import json
from pathlib import Path
import pickle
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import committed, canonical_review
from long_review_pilot import review as root_review
from reasoning_pilot import short_repair
from transfer_variants import actor
from transport import resilient
from study import fixed
from study_common import BASELINES, statistics
from prepare_mixed_mentor import plain_data, sha
from prospective_freeze import check_development_identity, check_no_test_used, observer_slots, best
from matched_review_b import ROOT_STOPS, validate_review_charge
from fit_second_review import read, wait
import mentor_pairs
import second_review_pilot as pilot
import second_review_pomdp as semantic

B_QUERY = ('always', 'root_disagreement')
B_STOP = ('action', 'agreement', .95, 'never')


def llm_policy(row, tree, judges, settings):
    """LLM action/confidence/agreement only; no POMDP or neural observation."""
    qid = row['id']
    if (qid, '') not in judges:
        raise semantic.action_core.MissingObservation('judge', '')
    original = judges[qid, '']; validate_review_charge(original)
    action = original['action']; rule = settings['root_stop']
    if rule == 'agreement':
        action = 'stop' if semantic.mentor_core.equal(original['answer'], tree['']['answer']) else 'repair'
    elif rule != 'action':
        action = 'stop' if original['p_correct'] >= rule else action if action != 'stop' else 'repair'
    decisions = [{'path': '', 'action': action, 'queried': True}]
    path, visited, acquired = '', [''], ['']
    if action != 'stop':
        path = 'guided' if action == 'repair' else 'retry'
        if path not in tree:
            raise semantic.action_core.MissingObservation('solver', path)
        visited.append(path)
        query = path == 'guided' and row['subject'] in ('chemistry', 'physics') and (
            settings['critical_query'] == 'always' or not semantic.mentor_core.equal(original['answer'], tree[path]['answer']))
        if query:
            if (qid, 'guided') not in judges:
                raise semantic.action_core.MissingObservation('judge', 'guided')
            judge = judges[qid, 'guided']; validate_review_charge(judge); acquired.append('guided')
            rule = settings['critical_stop']
            stop = (judge['action'] == 'stop' if rule == 'action' else
                    semantic.mentor_core.equal(judge['answer'], tree[path]['answer']) if rule == 'agreement' else
                    False if rule == 'never' else judge['p_correct'] >= rule)
            action = 'stop' if stop else 'retry' if judge['action'] == 'retry' else 'repair'
        else:
            action = 'stop' if semantic.mentor_core.equal(original['answer'], tree[path]['answer']) else 'repair'
        decisions.append({'path': path, 'action': action, 'queried': bool(query)})
        if action != 'stop':
            path += '/' + ('retry' if action == 'retry' else 'critical-repair' if query else 'reuse')
            if path not in tree:
                raise semantic.action_core.MissingObservation('solver', path)
            visited.append(path)
    item = semantic.mentor_core.outcome(row, tree, judges, path, visited, acquired, decisions)
    item.update(verifier_calls=0, verifier_tokens=0, verifier_token_slots=0, verifier_parameters_billions=0.,
                judge_physical_calls=sum(judges[qid, p]['physical_api_calls'] for p in acquired))
    return item


def view(data, bank):
    rows, trees = plain_data(data, 'dev'); by_id = {r['id']: r for r in rows}
    judges, seen = {}, set()
    for record in committed(bank):
        qid, kind, path = record['id'], record['kind'], record['path']; node = record['response']
        if qid not in by_id or (qid, kind, path) in seen:
            raise ValueError('Unexpected or duplicate second-review development receipt')
        seen.add((qid, kind, path))
        if kind == 'judge' and path in ('', 'guided'):
            judges[qid, path] = canonical_review(node)
        elif kind == 'solver' and path in ('guided', 'guided/reuse', 'guided/retry', 'retry/reuse', 'guided/critical-repair'):
            trees[qid][path] = node
        else:
            raise ValueError('Forbidden second-review development path')
    for record in committed(bank):
        qid, kind, path = record['id'], record['kind'], record['path']; node = record['response']
        safe, tree = {**by_id[qid], 'answer': ''}, trees[qid]
        if kind == 'judge':
            if not path:
                mentor_pairs.verify_review(safe, tree[''], node, 1024, 512)
            else:
                if safe['subject'] not in ('chemistry', 'physics'):
                    raise ValueError('Critical science review appeared on another subject')
                pilot.verify_review(safe, judges[qid, ''], tree['guided'], node)
        else:
            if path == 'guided/retry':
                prompt, cap = pilot.old.retry_prompt(safe), 256
                if node != tree['repair/retry']:
                    raise ValueError('A shared pure retry must be the exact sealed control receipt')
            else:
                ancestor = '' if path == 'guided' else path.rsplit('/', 1)[0]
                judge = judges[qid, 'guided' if path == 'guided/critical-repair' else '']
                prompt, cap = short_repair(safe, tree[ancestor], judge), 128
            logical = 'guided/reuse' if path == 'guided/critical-repair' else path
            if (node['prompt_sha256'], node['seed'], node.get('output_cap', cap if path == 'guided/retry' else None), node['model_digest']) != (
                    hashlib.sha256(prompt.encode()).hexdigest(), mentor_pairs.seed_for(qid, logical), cap, pilot.old.MODEL_DIGEST):
                raise ValueError('Changed second-review paid prompt/seed/cap')
            if (node['temperature'], node.get('context_length', 4096 if path == 'guided/retry' else None)) != (.65, 4096):
                raise ValueError('Changed second-review compact generation settings')
            if (type(node['input_tokens']) is not int or node['input_tokens'] <= 0 or
                    type(node['output_tokens']) is not int or not 0 <= node['output_tokens'] <= cap):
                raise ValueError('Invalid second-review compact token receipt')
    return rows, trees, judges


def metrics(items):
    result = statistics(items)
    result['mean_judge_physical_calls'] = sum(v['judge_physical_calls'] for v in items) / len(items)
    return result


def cache_key(observer):
    # Adapters can transform the same checkpoint's raw outputs differently.
    return semantic.base_core.asset_id(observer), type(observer).__module__, type(observer).__name__


def run(args):
    fitted = wait(args.models / 'complete.json', args.wait)
    source = wait(args.seed_development / 'summary.json', args.wait)
    panel = sha(args.data / 'splits/dev-questions.jsonl')
    seed_identity = check_development_identity(args.seed_development, panel); check_no_test_used(source)
    if fitted['identity_sha256'] != sha(args.models / 'identity.json') or fitted.get('test_used', False):
        raise ValueError('Changed critical model fitting provenance')
    source_bank = args.seed_development / 'dev-observations.jsonl'
    if source['bank_sha256'] != sha(source_bank):
        raise ValueError('Source development bank changed')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'source_sha256': sha(__file__), 'policy_sha256': sha(semantic.__file__),
                'pilot_source_sha256': sha(pilot.__file__), 'models_sha256': sha(args.models / 'complete.json'),
                'models_directory': str(args.models.resolve()), 'dev_questions_sha256': panel, 'test_used': False,
                'source_bank_sha256': sha(source_bank), 'source_summary_sha256': sha(args.seed_development / 'summary.json'),
                'maximum_reviews': 2, 'maximum_additional_solver_calls': 2,
                'critical_available': 'Science only, after guided, at one remaining solver call',
                'penalties': [0., .02, .08, .16], 'root_thresholds': [0., .03], 'critical_thresholds': [0., .03, .08],
                'B_root_stops': list(ROOT_STOPS), 'B_query_rules': list(B_QUERY), 'B_critical_stops': list(B_STOP),
                'B_role': 'Capability-matched LLM-only actions/confidence/agreement; no NN or POMDP',
                'B_references': [str(p.resolve()) for p in args.b_summary],
                'A_references': [str(p.resolve()) for p in args.a_summary], 'maximum_cloud_gpus': 2}
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Changed second-review development protocol')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    bank = args.output / 'dev-observations.jsonl'
    if not bank.exists():
        shutil.copy2(source_bank, bank)
    if not bank.read_bytes().startswith(source_bank.read_bytes()) or not bank.read_bytes().endswith(b'\n'):
        raise ValueError('Changed source prefix or partial last receipt')
    rows, trees, judges = view(args.data, bank)
    if len(rows) != 256:
        raise ValueError('Use the entire256-question development panel')
    request = None
    if args.collect:
        import modal
        request = resilient(modal.Cls.from_name(pilot.APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        if tags.get(pilot.old.MODEL) != pilot.old.MODEL_DIGEST or tags.get(pilot.old.JUDGE) != pilot.old.JUDGE_DIGEST:
            raise ValueError('Changed frozen compact solver or reviewer')
    additions = 0; b_candidates, c_candidates = [], []
    cache_path = args.output / 'private-neural-caches.pkl'
    caches = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    # Copy every compatible source cache, with no outcome-dependent filtering.
    # A singular old cache is used only for the exact shared parent checkpoint.
    source_caches = args.seed_development / 'private-neural-caches.pkl'
    source_cache = args.seed_development / 'private-neural-cache.pkl'
    imported_caches = pickle.loads(source_caches.read_bytes()) if source_caches.exists() else {}
    imported_single = pickle.loads(source_cache.read_bytes()) if source_cache.exists() else None
    fit_identity = read(args.models / 'identity.json')
    seed_matches_parent = seed_identity.get('models_sha256') == fit_identity['critical_parent_complete_sha256']
    pool = {}
    with bank.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def acquire(row, missing):
            nonlocal additions
            qid, kind, path = row['id'], missing.kind, missing.path
            safe, tree = {**row, 'answer': ''}, trees[qid]
            if kind == 'solver' and path == 'guided/retry':
                node = tree['repair/retry']
            elif request is None:
                raise RuntimeError('Missing paid response; use --collect') from missing
            elif kind == 'judge' and not path:
                node = canonical_review(root_review(request, safe, tree[''], 1024, 512))
            elif kind == 'judge' and path == 'guided' and row['subject'] in ('chemistry', 'physics'):
                node = pilot.review(request, safe, judges[qid, ''], tree['guided'])
            elif kind == 'solver' and path in ('guided', 'guided/reuse', 'guided/critical-repair', 'retry/reuse'):
                ancestor = '' if path == 'guided' else path.rsplit('/', 1)[0]
                mentor = judges[qid, 'guided' if path == 'guided/critical-repair' else '']
                logical = 'guided/reuse' if path == 'guided/critical-repair' else path
                node = actor(request, safe, short_repair(safe, tree[ancestor], mentor), mentor_pairs.seed_for(qid, logical), 128)
            else:
                raise ValueError('Requested an unavailable second-review action')
            stream.write(json.dumps({'id': qid, 'kind': kind, 'path': path, 'response': node}) + '\n'); stream.flush()
            if kind == 'judge':
                judges[qid, path] = node
            else:
                tree[path] = node
            additions += 1
            if additions % 16 == 0:
                print(f'Collected {additions} matched second-review development responses', flush=True)

        def evaluate(function):
            values = []
            for row in rows:
                while True:
                    try:
                        values.append(function(row, trees[row['id']], judges)); break
                    except semantic.action_core.MissingObservation as missing:
                        acquire(row, missing)
            return values

        # Fit no B models: all rules use only the LLM's paid outputs. This also
        # fills matched responses before loading the neural C observers.
        for stop, query, second in product(ROOT_STOPS, B_QUERY, B_STOP):
            settings = {'root_stop': stop, 'critical_query': query, 'critical_stop': second}
            values = evaluate(lambda r, t, j: llm_policy(r, t, j, settings))
            b_candidates.append({'family': 'B', 'protocol': 'second-critical-review', 'settings': settings, 'dev': metrics(values)})
        print('Completed capability-matched second-review B search', json.dumps(best(b_candidates)), flush=True)
        (args.output / 'B-search.json').write_text(json.dumps(b_candidates, indent=2) + '\n')
        for spec in fitted['models']:
            path = args.models / (spec['key'] + '.pkl')
            if sha(path) != spec['sha256']:
                raise ValueError('Changed critical-review POMDP')
            model = pickle.loads(path.read_bytes())
            slots = observer_slots(model)
            adapter_classes = {}
            for _, observer in slots:
                key = cache_key(observer)
                adapter_classes.setdefault(key[0], set()).add(key[1:])
            for _, observer in slots:
                observer.device = args.device
                key = cache_key(observer)
                legacy = imported_caches.get(key[0], {}) if len(adapter_classes[key[0]]) == 1 else {}
                inherited = imported_caches.get(key, legacy) if seed_matches_parent else {}
                if imported_single is not None and seed_matches_parent and key == cache_key(model.mentor.classifier):
                    inherited = imported_single
                observer._cache = caches.get(key, inherited)
            model.primary._asset_pool = pool
            subjects = {s: [] for s in ('math', 'chemistry', 'physics')}
            for penalty, threshold, critical_threshold in product(identity['penalties'], identity['root_thresholds'], identity['critical_thresholds']):
                settings = {'penalty': penalty, 'threshold': threshold, 'critical_threshold': critical_threshold}
                values = evaluate(lambda r, t, j: semantic.evaluate(r, t, j, model, settings, 'C'))
                c_candidates.append({'model': spec['key'], 'policy_module': 'second_review_pomdp', 'settings': settings, 'dev': metrics(values)})
                for subject in subjects:
                    subjects[subject].append({'settings': settings, 'dev': metrics([v for r, v in zip(rows, values) if r['subject'] == subject])})
            for utility in (0., .04, .08, .16):
                settings = {'by_subject': {s: max(v, key=lambda r: (r['dev']['accuracy'] - utility * r['dev']['mean_compute'] / 1000,
                                                                   r['dev']['accuracy'], -r['dev']['mean_compute']))['settings'] for s, v in subjects.items()}}
                values = evaluate(lambda r, t, j: semantic.evaluate(r, t, j, model, settings, 'C'))
                c_candidates.append({'model': spec['key'], 'policy_module': 'second_review_pomdp', 'settings': settings, 'dev': metrics(values)})
            for observer in pool.values():
                caches[cache_key(observer)] = observer._cache
            cache_path.write_bytes(pickle.dumps(caches, protocol=5))
            (args.output / 'search.json').write_text(json.dumps(c_candidates, indent=2) + '\n')
            print('Completed second-review C', spec['key'], json.dumps(best(c_candidates)), flush=True)
    view(args.data, bank)  # Revalidate all newly acquired physical phase/prompt receipts.
    prior_b, saw_ordinary, saw_matched = [], False, False
    for path in args.b_summary:
        value = wait(path, args.wait); reference = check_development_identity(path.parent, panel); check_no_test_used(value)
        saw_ordinary |= reference.get('source_sha256') == sha(Path(__file__).with_name('development_b.py'))
        saw_matched |= 'ordinary_B_source' in reference and 'root_stop_rules' in reference
        prior_b.append({**value['B'], 'source_summary': str(path.resolve()), 'source_summary_sha256': sha(path)})
    if not saw_ordinary or not saw_matched:
        raise ValueError('Retain both completed ordinary and matched full-review B references')
    previous_b = best(prior_b); matched_b = best(b_candidates); strongest_b = best([previous_b, matched_b])
    a = []
    for path in args.a_summary:
        value = wait(path, args.wait); check_development_identity(path.parent, panel); check_no_test_used(value)
        a.append({**value['selected_A'], 'source_summary': str(path.resolve()), 'source_summary_sha256': sha(path)})
    strongest_a = best(a)
    eligible = [c for c in c_candidates if c['dev']['mean_judge_calls'] > 0 and c['dev']['mean_compute'] < strongest_b['dev']['mean_compute']]
    chosen = best(eligible) if eligible else None
    summary = {'development_questions': 256, 'test_used': False, 'maximum_reviews': 2,
               'baselines': {name: metrics([dict(fixed(r, trees[r['id']], *s), judge_physical_calls=0) for r in rows]) for name, s in BASELINES.items()},
               'A': strongest_a, 'B': strongest_b, 'prior_B': previous_b, 'second_review_B': matched_b,
               'best': best(c_candidates), 'selected_C': chosen, 'bank_sha256': sha(bank),
               'new_development_nodes': additions, 'models_directory': str(args.models.resolve()),
               'policy_module': 'second_review_pomdp',
               'capability_comparison': 'C-specific claims use max(prior full B family, matched second-review LLM-only B).'}
    if chosen:
        summary['C_above_higher_A_B'] = chosen['dev']['accuracy'] - max(strongest_a['dev']['accuracy'], strongest_b['dev']['accuracy'])
        summary['C_plus_10_points'] = summary['C_above_higher_A_B'] >= .1 - 1e-12
        summary['C_compute_saved_vs_B'] = 1. - chosen['dev']['mean_compute'] / strongest_b['dev']['mean_compute']
    else:
        summary['C_plus_10_points'] = False
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'models', 'seed-development', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--a-summary', type=Path, nargs='+', required=True)
    p.add_argument('--b-summary', type=Path, nargs='+', required=True)
    p.add_argument('--wait', action='store_true'); p.add_argument('--collect', action='store_true')
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    run(p.parse_args())
