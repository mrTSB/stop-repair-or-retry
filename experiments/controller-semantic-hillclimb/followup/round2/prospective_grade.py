"""Open references once, only after all1024x8 blind decisions and audits complete."""

import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path

import numpy as np

from prospective_freeze import verify, sha
from prospective_collect import (METHODS, STORES, ResponseStore, strict_records, fingerprint,
                                 verify_decision, atomic_json, source_group, read_public_rows)
from study_common import BASELINES, statistics
from scoring import correct


def prerequisites(study, freeze_sha):
    """Check complete immutable inference evidence before any reference-file parse."""
    frozen = verify(study, freeze_sha)
    directory = study / 'collection'
    complete = json.loads((directory / 'complete.json').read_bytes())
    audit = json.loads((directory / 'collection-audit.json').read_bytes())
    if (not complete['passed'] or not audit['passed'] or complete['freeze_sha256'] != freeze_sha
            or audit['freeze_sha256'] != freeze_sha or complete['questions'] != 1024
            or complete['method_question_audits'] != 8192 or audit['method_question_audits'] != 8192
            or complete['methods'] != list(METHODS) or complete['digests'] != audit['digests']):
        raise ValueError('Complete all eight methods on all1024questions before opening grades')
    if complete['audit_sha256'] != sha(directory / 'collection-audit.json'):
        raise ValueError('Collection audit changed after completion')
    if complete['development_audit_sha256'] != sha(study / 'development-audit.json'):
        raise ValueError('Development audit changed after completion')
    development = json.loads((study / 'development-audit.json').read_bytes())
    if (not development['passed'] or development['freeze_sha256'] != freeze_sha
            or development['method_question_audits'] != 2048):
        raise ValueError('Missing complete frozen development audit')
    if development['decision_receipts_sha256'] != sha(study / 'development-replay.json'):
        raise ValueError('Frozen development decision audit changed')
    expected_files = {name + '.jsonl' for name in STORES} | {'ungraded-decisions.jsonl'}
    if set(complete['digests']) != expected_files:
        raise ValueError('Completion omitted a response or decision bank')
    for name, digest in complete['digests'].items():
        if sha(directory / name) != digest:
            raise ValueError('Completed inference bank changed: ' + name)
    records = strict_records(directory / 'ungraded-decisions.jsonl')
    known = {(r['id'], r['method']): r for r in records}
    ids = {r['id'] for r in records}
    if len(records) != 8192 or len(known) != len(records) or len(ids) != 1024 or set(known) != {(qid, method) for qid in ids for method in METHODS}:
        raise ValueError('Incomplete, duplicate, or filtered policy decision grid')
    if fingerprint(sorted(ids)) != complete['question_ids_sha256']:
        raise ValueError('Completed question membership changed')
    public = read_public_rows(Path(frozen['data_directory']) / 'splits/test-questions.jsonl')
    public_by_id = {r['id']: r for r in public}
    if len(public) != 1024 or set(public_by_id) != ids:
        raise ValueError('Completed decisions differ from the reserved public question membership')
    store = ResponseStore(records={name: strict_records(directory / (name + '.jsonl')) for name in STORES})
    if any(qid not in ids for bank in store.known.values() for qid, _, _ in bank):
        raise ValueError('Nonpanel response in completed bank')
    for record in records:
        if record['freeze_sha256'] != freeze_sha:
            raise ValueError('Decision belongs to another freeze')
        verify_decision(frozen, record, store, public_by_id[record['id']])
    return frozen, complete, known, store


def paired_margins(rows, results, repetitions=5000):
    names = list(METHODS)
    accuracy = np.asarray([[r['correct'] for r in results[name]] for name in names], dtype=float)
    work = np.asarray([[r['compute'] for r in results[name]] for name in names], dtype=float)
    fixed = [names.index(name) for name in BASELINES]
    a, b, c = (names.index(name) for name in ('A', 'B', 'C'))
    def compare(values, cost):
        maximum = values[fixed].max(axis=0)
        return {'A_above_highest_baseline': values[a] - maximum,
                'B_above_highest_baseline': values[b] - maximum,
                'C_above_highest_A_B': values[c] - np.maximum(values[a], values[b]),
                'C_above_B': values[c] - values[b],
                'C_compute_saved_vs_B': cost[b] - cost[c],
                'C_compute_fraction_saved_vs_B': 1. - cost[c] / cost[b]}
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[source_group(row), row['subject']].append(i)
    random = np.random.default_rng(2026112201)
    draw = np.concatenate([random.choice(indexes, (repetitions, len(indexes)), replace=True)
                           for _, indexes in sorted(groups.items())], axis=1)
    point = compare(accuracy.mean(1), work.mean(1))
    bootstrap = compare(accuracy[:, draw].mean(2), work[:, draw].mean(2))
    return {key: {'point': float(value), 'ci95': np.quantile(bootstrap[key], [.025, .975]).tolist()}
            for key, value in point.items()}


def table(rows, results):
    metrics = {method: {**statistics(values),
                       'mean_judge_physical_calls': float(np.mean([r['judge_physical_calls'] for r in values]))}
               for method, values in results.items()}
    margins = paired_margins(rows, results)
    required = {'A_plus_10_points': margins['A_above_highest_baseline']['point'] >= .1 - 1e-12,
                'B_plus_10_points': margins['B_above_highest_baseline']['point'] >= .1 - 1e-12,
                'C_plus_10_points': margins['C_above_highest_A_B']['point'] >= .1 - 1e-12,
                'C_compute_below_B': margins['C_compute_saved_vs_B']['point'] > 0.}
    return {'questions': len(rows), 'methods': metrics, 'margins': margins, 'criteria': required,
            'all_requested_point_targets_passed': all(required.values()),
            'C_accuracy_clearly_above_B': margins['C_above_B']['ci95'][0] > 0.,
            'C_compute_clearly_below_B': margins['C_compute_fraction_saved_vs_B']['ci95'][0] > 0.,
            'highest_baseline': max(BASELINES, key=lambda name: (metrics[name]['accuracy'], -metrics[name]['mean_compute']))}


def run(args):
    with (args.study / 'grading.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = args.study / 'graded'
        if destination.exists():
            raise ValueError('This panel has already been opened for grading; never overwrite or reselect')
        frozen, complete, decisions, store = prerequisites(args.study, args.freeze_sha256)
        destination.mkdir()
        atomic_json(destination / 'opening.json', {'freeze_sha256': args.freeze_sha256,
            'completion_sha256': sha(args.study / 'collection/complete.json'), 'source_sha256': sha(__file__),
            'all8192blind_decisions_precede_reference_access': True})
        # This is the first reference-answer parse in the test evaluation path.
        path = Path(frozen['data_directory']) / 'splits/test-questions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ids = {r['id'] for r in rows}
        if len(rows) != 1024 or len(ids) != 1024 or fingerprint(sorted(ids)) != complete['question_ids_sha256']:
            raise ValueError('Reference membership differs from the complete blind panel')
        results = {method: [] for method in METHODS}
        sanitized = []
        for row in rows:
            for method in METHODS:
                record = decisions[row['id'], method]
                if record['subject'] != row['subject'] or record['source'] != source_group(row):
                    raise ValueError('Blind source/subject stratum changed')
                node = verify_decision(frozen, record, store)
                item = {**record['item'], 'correct': int(correct(node, row['answer']))}
                results[method].append(item)
                sanitized.append({'id': row['id'], 'source': source_group(row), 'subject': row['subject'],
                                  'method': method, **item})
        panels = {'joint': list(range(len(rows)))}
        for source in sorted({source_group(row) for row in rows}):
            panels[source] = [i for i, row in enumerate(rows) if source_group(row) == source]
        for source, subject in sorted({(source_group(row), row['subject']) for row in rows}):
            panels[source + '/' + subject] = [i for i, row in enumerate(rows)
                                              if (source_group(row), row['subject']) == (source, subject)]
        report = {name: table([rows[i] for i in indexes], {method: [values[i] for i in indexes]
                                                         for method, values in results.items()})
                  for name, indexes in panels.items()}
        summary = {'freeze_sha256': args.freeze_sha256, 'completion_sha256': sha(args.study / 'collection/complete.json'),
                   'test_question_sha256': frozen['reserved_question_sha256'], 'panels': report,
                   'source_extension': 'GSM8K math and SuperGPQA chemistry/physics; a separate population from the earlier912panel.',
                   'freshness': 'Unused in fitting and controller selection; pretraining contamination is not measured.',
                   'bootstrap': '5000paired draws stratified by source and subject; best fixed baseline and best A/B recomputed within every draw.',
                   'compute': frozen['inference_contract']['compute'],
                   'offline_training_excluded_from_inference_proxy': True,
                   'all_requested_point_targets_passed': report['joint']['all_requested_point_targets_passed']}
        atomic_json(destination / 'summary.json', summary)
        with (destination / 'per-question.jsonl').open('w') as output:
            output.write(''.join(json.dumps(item) + '\n' for item in sanitized))
            output.flush()
        atomic_json(destination / 'complete.json', {'passed': True, 'questions': 1024, 'decisions': 8192,
            'freeze_sha256': args.freeze_sha256, 'summary_sha256': sha(destination / 'summary.json'),
            'per_question_sha256': sha(destination / 'per-question.jsonl')})
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--freeze-sha256', required=True)
    run(p.parse_args())
