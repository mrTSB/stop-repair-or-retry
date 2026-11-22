"""Seal and verify the completed eight-row DEV checkpoint without model calls.

Run with uv run --no-project python .../concrete_checkpoint.py seal|verify.
Private model/trajectory files stay in runs; the manifest contains only hashes,
settings, aggregate measurements, and repository-relative paths. This is a
reproducibility checkpoint, not a prospective freeze or fresh evaluation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RUNS = ROOT/'runs/controller-semantic-followup/round2'
BASELINES = ('Stop', 'Repair once', 'Repair twice', 'Retry once', 'Retry twice')
MANIFEST = HERE/'current_checkpoint.json'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def sealed(folder):
    done = read(folder/'complete.json')
    files = done.get('artifacts', done.get('files'))
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing completed artifact seal')
    for name, expected in files.items():
        path = folder/name
        if path.resolve().parent != folder.resolve() or sha(path) != expected:
            raise ValueError('Completed artifact changed: '+name)
    return {relative(folder/'complete.json'): sha(folder/'complete.json'),
            **{relative(folder/name): value for name, value in files.items()}}


def metrics(items, compute_key='compute'):
    if len(items) != 256 or len({r['id'] for r in items}) != 256:
        raise ValueError('Require all256 unique DEV questions')
    if any(r['correct'] not in (0, 1, False, True) for r in items):
        raise ValueError('Invalid binary outcome')
    return {'n': 256, 'correct': sum(int(r['correct']) for r in items),
            'accuracy': sum(int(r['correct']) for r in items)/256,
            'mean_compute': math.fsum(r[compute_key] for r in items)/256}


def same(actual, expected):
    for key in ('n', 'correct', 'accuracy', 'mean_compute'):
        if not math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError('Per-question aggregate differs: '+key)


def build():
    a_dir = RUNS/'development-math-specialization-v1'
    c_dir = RUNS/'development-expanded-C-canonical-cost-v1'
    evidence = {**sealed(a_dir), **sealed(c_dir)}
    a, c = read(a_dir/'summary.json'), read(c_dir/'summary.json')
    selected_a, selected_c = a['selected_A'], c['canonical_selection']['selected_C']
    if a['test_used'] or a['fresh_validation'] or c['test_used'] or c['fresh_validation']:
        raise ValueError('This checkpoint must be reused development only')
    for chosen in (selected_a, selected_c):
        if sha(chosen['model_path']) != chosen['model_sha256']:
            raise ValueError('Selected controller model changed')
        evidence[relative(chosen['model_path'])] = chosen['model_sha256']
    a_items = read(a_dir/'private-candidate-items.json')[selected_a['key']]
    with (c_dir/'private-paired-outcomes.jsonl').open() as stream:
        c_items = [r for line in stream if (r := json.loads(line))['candidate_index'] == selected_c['original_candidate_index']]
    same(metrics(a_items), selected_a['dev'])
    same(metrics(c_items, 'canonical_compute'), selected_c['dev'])
    if {r['id'] for r in a_items} != {r['id'] for r in c_items}:
        raise ValueError('A and C panels differ')
    rows = [{'method': name, **a['baselines'][name]} for name in BASELINES]
    rows += [{'method': 'A', **selected_a['dev']}, {'method': 'B', **c['B']['dev']},
             {'method': 'C', **selected_c['dev']}]
    latest = read(HERE/'latest_results.json')
    for row, reported in zip(rows, latest['rows'], strict=True):
        if row['method'] != reported['method']:
            raise ValueError('Changed table ordering')
        same(row, {'n': reported['n'], 'correct': reported['correct'],
                   'accuracy': reported['accuracy_percent']/100,
                   'mean_compute': reported['mean_compute_proxy']})
    for path in (Path(__file__), HERE/'latest_results.json'):
        evidence[relative(path)] = sha(path)
    identities = {
        'A': {k: v for k, v in selected_a.items() if k != 'dev'},
        'B': {k: v for k, v in c['B'].items() if k != 'dev'},
        'C': {k: selected_c[k] for k in ('policy_module', 'builder_module', 'model', 'model_path', 'model_sha256', 'settings', 'smoothing', 'original_candidate_index')},
    }
    for identity in identities.values():
        if 'model_path' in identity:
            identity['model_path'] = relative(identity['model_path'])
    best_baseline = max(r['accuracy'] for r in rows[:5])
    best_ab = max(rows[5]['accuracy'], rows[6]['accuracy'])
    return {'schema': 'concrete-completed-DEV-checkpoint-v1',
            'split': 'Repeatedly reused256-question development; not fresh confirmation',
            'model_calls': 0, 'test_access': False, 'fresh_test_reserved': 1024,
            'solver': {'model': 'qwen3.5:4b', 'digest': '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd', 'maximum_continuations': 2},
            'reviewer': {'model': 'qwen3.5:35b', 'digest': '3460ffeede5453ead027dbd2f821b12ad0aa3de54630971993babdb2165221f7'},
            'compute': {'formula': '(4*solver_tokens +3*reviewer_tokens +observer_billions*paid_token_slots)/3.09',
                        'scope': 'Inference parameter-token proxy; every paid phase included; full35B sensitivity in rows; not FLOPs or wall time'},
            'baselines': {'Stop': 'Submit initial answer', 'Repair once': 'Submit after one plain repair',
                          'Repair twice': 'Submit after two plain repairs', 'Retry once': 'Submit after one fresh-context retry',
                          'Retry twice': 'Submit after two fresh-context retries'},
            'identities': identities, 'rows': rows, 'evidence': evidence,
            'verification': {'all_completed_artifact_hashes': True, 'selected_model_hashes': True,
                             'A_C_recomputed_from_individual_outcomes': True, 'baselines_B_bound_to_completed_aggregate_receipts': True},
            'targets': {'A_minimum_correct_DEV': math.ceil(256*(best_baseline+.10)),
                        'B_minimum_correct_DEV': math.ceil(256*(best_baseline+.10)),
                        'C_minimum_correct_DEV': max(math.floor(256*.80)+1, math.ceil(256*(best_ab+.10))),
                        'C_accuracy_strictly_above': .80, 'C_gap_above_best_A_B': .10,
                        'C_compute_strictly_below_B': True, 'achieved': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('seal', 'verify'))
    args = parser.parse_args()
    result = build()
    if args.phase == 'seal':
        with MANIFEST.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True)+'\n')
    elif read(MANIFEST) != result:
        raise ValueError('Checkpoint manifest or underlying evidence changed')
    print(json.dumps({'manifest': relative(MANIFEST), 'verified': True,
                      'rows': [{k: r[k] for k in ('method', 'correct', 'n', 'accuracy', 'mean_compute')} for r in result['rows']],
                      'targets': result['targets']}, indent=2))


if __name__ == '__main__':
    main()
