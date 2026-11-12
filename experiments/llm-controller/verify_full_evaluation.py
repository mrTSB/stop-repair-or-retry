"""Offline, portable verification of the full Modal controller archive.

Stored absolute provenance paths are never followed. Development evidence and the
freeze are resolved inside --run, and the published cache inside --dataset.
No inference, cloud APIs, or provider credentials are used.
"""
import argparse
from collections import Counter
from datetime import datetime
import importlib.util
import hashlib
import json
import random
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from stop_repair_retry import llm_experiment as experiment


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bootstrap95(values):
    """Reproduce saved percentile endpoints, not a claim about coverage."""
    rng = random.Random(20261110)
    count = len(values)
    samples = sorted(sum(values[rng.randrange(count)] for _ in range(count)) / count
                     for _ in range(2000))
    return [samples[50], samples[1949]]


def verify_split(directory, rows, frozen_experiment, backend, scorer):
    directory = Path(directory)
    manifest = read(directory / 'manifest.json')
    check(manifest['experiment'] == frozen_experiment, 'Split experiment differs from freeze')
    check(manifest['question_ids'] == [r['id'] for r in rows], 'Question IDs/order differ')
    check(manifest['split'] == rows[0]['split'], 'Split label differs')
    config = frozen_experiment['config']
    requests, raw, accepted = {}, {}, {}
    journal = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    for event in journal:
        rid = event['request_id']
        if event['type'] == 'request':
            check(rid not in requests, 'Duplicate request')
            requests[rid] = event
        elif event['type'] == 'raw_response':
            check(rid in requests and rid not in raw, 'Orphan/duplicate raw response')
            raw[rid] = event
        elif event['type'] == 'response':
            check(rid in raw and rid not in accepted, 'Orphan/duplicate accepted response')
            check(event['response'] == raw[rid]['response'], 'Raw and accepted response differ')
            accepted[rid] = event
        else:
            raise ValueError('Unexpected event or unresolved failure in full run')
    check(set(requests) == set(raw) == set(accepted), 'Unresolved request')
    reported = read(directory / 'controller-results.json')
    check(len(reported) == len(rows), 'Result count differs')
    used, reconstructed, sequences, totals = set(), [], Counter(), Counter()
    for row, saved in zip(rows, reported):
        check(saved['id'] == row['id'], 'Result ID/order differs')
        path, visited, actions, decisions = '', [''], [], []
        for depth in range(2):
            rid = experiment.digest([row['id'], path])
            check(rid in accepted, 'Missing visited decision')
            used.add(rid)
            request, response = requests[rid], accepted[rid]['response']
            check(request['question_id'] == row['id'] and request['path'] == path, 'Request provenance mismatch')
            check(request['body'] == experiment.request_body(row, path, config, frozen_experiment['prompt']), 'Request differs from allowlisted visited history')
            check(response['model'] == backend['reported_model'], 'Unexpected reported model')
            measurement = response['measurement']
            check(measurement['model_revision'] == backend['revision'], 'Model revision differs')
            check(all(m['content'] in measurement['formatted_prompt'] for m in request['body']['messages']), 'Formatted prompt differs')
            usage = response['usage']
            inp, out = usage['prompt_tokens'], usage['completion_tokens']
            check(type(inp) is int and inp > 0 and type(out) is int and 0 <= out <= config['max_output_tokens'], 'Invalid token usage/cap')
            check(usage['total_tokens'] == inp + out, 'Usage total mismatch')
            check(len(response['choices']) == 1, 'Expected one completion')
            text = response['choices'][0]['message']['content']
            valid, action = False, 'stop'
            try:
                obj = json.loads(text)
                valid = isinstance(obj, dict) and set(obj) == {'action'} and obj['action'] in ('stop', 'repair', 'retry')
                if valid:
                    action = obj['action']
            except (ValueError, TypeError):
                pass
            decisions.append((inp, out, valid, accepted[rid]['latency_seconds']))
            actions.append(action)
            if action == 'stop':
                break
            path += ('/' if path else '') + action
            visited.append(path)
        nodes = [row['tree'][p] for p in visited]
        si, so = sum(n['input_tokens'] for n in nodes), sum(n['output_tokens'] for n in nodes)
        ci, co = sum(v[0] for v in decisions), sum(v[1] for v in decisions)
        expected = {'id': row['id'], 'domain': row['domain'], 'policy': 'llm_controller', 'actions': actions,
                    'path': path, 'answer': nodes[-1]['answer'], 'correct': int(scorer.correct(nodes[-1], row['answer'])),
                    'initial_correct': int(scorer.correct(nodes[0], row['answer'])), 'solver_input_tokens': si,
                    'solver_output_tokens': so, 'controller_input_tokens': ci, 'controller_output_tokens': co,
                    'tokens': si + so + ci + co, 'controller_calls': len(decisions),
                    'invalid_actions': sum(not v[2] for v in decisions),
                    'controller_latency_seconds': sum(v[3] for v in decisions)}
        check(saved == expected, 'Result differs from independent path/score/cost reconstruction')
        reconstructed.append(expected)
        sequences[' -> '.join(actions)] += 1
        totals.update(solver_input=si, solver_output=so, controller_input=ci, controller_output=co)
    check(used == set(requests), 'Extra/unvisited requests')
    summary = read(directory / 'summary.json')
    check(summary['mode'] == 'live' and summary['n'] == len(rows) and summary['split'] == manifest['split'], 'Summary scope mismatch')
    check(summary['invalid_actions'] == sum(v['invalid_actions'] for v in reconstructed), 'Invalid-action count differs')
    check(summary['estimated_controller_usd'] is None, 'Compute billing misrepresented as token USD')
    check(summary['reported_models'] == [backend['reported_model']], 'Summary reported model mismatch')
    check(summary['penalty'] == config['penalty'], 'Penalty mismatch')
    fixed = {'stop': [''], 'repair_once': ['', 'repair'], 'repair_twice': ['', 'repair', 'repair/repair'],
             'retry_once': ['', 'retry'], 'retry_twice': ['', 'retry', 'retry/retry']}
    results = {'llm_controller': [(v['correct'], v['tokens'], v['domain'], v['controller_calls']) for v in reconstructed]}
    for name, paths in fixed.items():
        results[name] = [(int(scorer.correct(row['tree'][paths[-1]], row['answer'])),
                          sum(row['tree'][p]['input_tokens'] + row['tree'][p]['output_tokens'] for p in paths),
                          row['domain'], 0) for row in rows]
    check(len(summary['summaries']) == 18 and {(s['domain'], s['policy']) for s in summary['summaries']} == {(d, p) for d in ['all', 'math', 'science'] for p in results}, 'Expected six distinct methods across three domains')
    for s in summary['summaries']:
        values = [v for v in results[s['policy']] if s['domain'] == 'all' or v[2] == s['domain']]
        check(s['n'] == len(values), 'Summary denominator mismatch')
        check(s['accuracy_ci95'] == bootstrap95([v[0] for v in values]), 'Accuracy bootstrap endpoints differ')
        for field, expected in [('accuracy', statistics.mean(v[0] for v in values)),
                                ('mean_tokens', statistics.mean(v[1] for v in values)),
                                ('mean_controller_calls', statistics.mean(v[3] for v in values)),
                                ('mean_utility', statistics.mean(v[0] - config['penalty'] * v[1] / 1000 for v in values))]:
            check(abs(s[field] - expected) < 1e-10, 'Summary aggregate mismatch: ' + field)
    check(len(summary['paired_utility_differences']) == 5 and {v['baseline'] for v in summary['paired_utility_differences']} == set(fixed), 'Expected five distinct paired comparisons')
    for comparison in summary['paired_utility_differences']:
        differences = [a[0] - b[0] - config['penalty'] * (a[1] - b[1]) / 1000
                       for a, b in zip(results['llm_controller'], results[comparison['baseline']])]
        check(abs(comparison['mean_utility_difference'] - statistics.mean(differences)) < 1e-10, 'Paired utility mismatch')
        check(comparison['ci95'] == bootstrap95(differences), 'Paired bootstrap endpoints differ')
    return {'questions': len(rows), 'requests': len(requests), 'pending': 0,
            'invalid_actions': sum(v['invalid_actions'] for v in reconstructed),
            'finish_reasons': dict(Counter(v['response']['choices'][0]['finish_reason'] for v in accepted.values())),
            'output_token_lengths': dict(Counter(v['response']['usage']['completion_tokens'] for v in accepted.values())),
            'action_sequences': dict(sequences), 'token_totals': dict(totals),
            'accuracy': statistics.mean(v['correct'] for v in reconstructed),
            'first_request_utc': journal[0]['created_utc'], 'last_response_utc': journal[-1]['created_utc']}


def verify(run, dataset):
    run, dataset = Path(run), Path(dataset)
    freeze, protocol, backend = read(run / 'frozen.json'), read(run / 'protocol.json'), read(run / 'backend.json')
    frozen = freeze['experiment']
    check(frozen['mode'] == 'live' and freeze['development_questions'] == 60, 'Not a complete live development freeze')
    check(freeze['fingerprint_sha256'] == experiment.digest(frozen), 'Freeze digest mismatch')
    check(frozen == experiment.fingerprint(frozen['config'], frozen['prompt'], dataset, 'live'), 'Code/prompt/reward/dataset fingerprint mismatch')
    check(hashlib.sha256(frozen['prompt'].encode()).hexdigest() == protocol['prompt_sha256'], 'Protocol prompt hash mismatch')
    check(experiment.sha(run / 'modal_full_evaluation.frozen.py') == protocol['driver_sha256'], 'Archived driver hash mismatch')
    check(experiment.sha(run / 'dev/manifest.json') == freeze['dev_manifest_sha256'], 'Development manifest hash mismatch')
    check(experiment.sha(run / 'dev/events.jsonl') == freeze['dev_events_sha256'], 'Development journal hash mismatch')
    test_manifest = read(run / 'test/manifest.json')
    check(test_manifest['test_freeze']['sha256'] == experiment.sha(run / 'frozen.json'), 'Test freeze hash mismatch')
    check(backend['reported_model'] == frozen['config']['model'] == protocol['model'], 'Backend/config model mismatch')
    check(backend['reported_model'] in freeze['reported_models'], 'Model absent from development freeze')
    check(protocol['max_decisions'] == 920 and protocol['cumulative_reserved_usd'] < 20, 'Unexpected collection limits')
    spec = importlib.util.spec_from_file_location('cached_tree_scoring', dataset / 'scoring.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    all_rows = {s: experiment.load_dataset(dataset, s) for s in ('train', 'dev', 'test')}
    all_ids = {s: {r['id'] for r in rr} for s, rr in all_rows.items()}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        check(not all_ids[a] & all_ids[b], 'Global split IDs overlap')
    check(len(all_rows['train']) == 100, 'Unexpected training size')
    verified = {}
    for split, count in [('dev', 60), ('test', 400)]:
        rows = all_rows[split]
        check(len(rows) == count, 'Unexpected split size')
        check(Counter(r['domain'] for r in rows) == {'math': count // 2, 'science': count // 2}, 'Unbalanced domains')
        verified[split] = verify_split(run / split, rows, frozen, backend, scorer)
    check(datetime.fromisoformat(verified['dev']['last_response_utc']) <= datetime.fromisoformat(freeze['frozen_utc'])
          <= datetime.fromisoformat(verified['test']['first_request_utc']), 'Test started before completed development freeze')
    # These are the five published baselines on this same cache, not older controller results.
    published = read(dataset / 'results.json')['summaries']
    aliases = {'stop': 'stop', 'repair_once': 'repair_once', 'repair_twice': 'repair',
               'retry_once': 'retry_once', 'retry_twice': 'retry'}
    for summary in read(run / 'test/summary.json')['summaries']:
        if summary['policy'] not in aliases:
            continue
        baseline = next(v for v in published if v['policy'] == aliases[summary['policy']]
                        and v['domain'] == summary['domain'])
        for field in ('n', 'accuracy', 'accuracy_ci95', 'mean_tokens', 'mean_utility'):
            check(summary[field] == baseline[field], 'Published same-cache baseline differs: ' + field)
    calls = sum(s['requests'] for s in verified.values())
    check(calls <= 920 and calls == read(run / 'completion.json')['controller_calls'], 'Collection call total mismatch')
    return {'status': 'pass', 'verified_utc': experiment.now(), 'portable_archive_verification': True,
            'stored_absolute_paths_followed': False, 'network_calls': 0, 'splits': verified,
            'controller_calls': calls, 'model': backend['reported_model'],
            'checks': ['frozen code/prompt/reward/data hashes', 'complete development evidence before test',
                       'raw journal uniqueness/order', 'visited-history request allowlist', 'raw response equality',
                       'reported model/revision', 'usage totals and output cap', 'independent actions/correctness/token accounting',
                       'global train/dev/test ID separation', 'five baseline summaries exactly match published same-cache characterization', 'all six-method/domain aggregate means', 'all five paired utility means',
                       'accuracy and paired bootstrap endpoint reproduction (seed20261110, 2000resamples)'],
            'limitations': ['Provider billing and app shutdown require separate operational checks.',
                            'Usage verified from recorded backend totals; token IDs are not archived for independent retokenization.',
                            'Existing test baseline outcomes were previously inspected; this is not an untouched confirmation set.',
                            'Bootstrap endpoint reproduction checks arithmetic and the fixed resampling algorithm, not nominal coverage or validity of inference.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(ROOT / 'results/llm-controller-full-evaluation'))
    parser.add_argument('--dataset', default=str(experiment.DEFAULT_DATA))
    parser.add_argument('--output', help='Optional JSON validation record; otherwise read-only.')
    args = parser.parse_args()
    result = verify(args.run, args.dataset)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
