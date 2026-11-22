"""TRAIN-only comparison of frozen review critics on all256 science pairs.

No solver or teacher call, new training of an encoder, or DEV/TEST reads.
The option observer excludes every evaluated question from neural training.
Calibration is cross-fitted by whole question. A passing prediction screen
does not establish a controller gain or additional solver answer availability.
"""
import argparse
from collections import Counter
import gc
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

import completion_v4_diagnostic as guarded
import expanded_mentor_inputs as inputs
import expanded_mentor_io as io
import extract_expanded_mentor as extractor
import fit_expanded_mentor as fitted
import science_option_observer as option_model
import strong_late_review_protocol as protocol

SCHEMA = 'science-root-review-critic-TRAIN-v1'
SEED = 2381123


def margin(raw):
    raw = np.asarray(raw, dtype=float)
    return np.logaddexp(raw[..., 1], raw[..., 3]) - np.logaddexp(raw[..., 0], raw[..., 2])


def public(row):
    return {key: row[key] for key in ('id', 'subject', 'question', 'options') if key in row} | {'answer': ''}


def fold_ids(rows):
    folds = np.empty(len(rows), dtype=int)
    for subject in sorted({r['subject'] for r in rows}):
        indexes = sorted((i for i, r in enumerate(rows) if r['subject'] == subject),
                         key=lambda i: hashlib.sha256((str(SEED)+rows[i]['id']).encode()).hexdigest())
        for rank, i in enumerate(indexes): folds[i] = rank % 3
    return folds


def crossfit(features, labels, folds):
    """Neither held features nor labels fit scaler or calibration coefficients."""
    result = np.empty(len(labels))
    for fold in range(3):
        train, held = folds != fold, folds == fold
        if len(np.unique(labels[train])) == 1:
            result[held] = labels[train][0]
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=SEED))
            model.fit(features[train], labels[train]); result[held] = model.predict_proba(features[held])[:, 1]
    return result


def load_inputs(args):
    fit_record = io.read(args.parent/'registration.json')
    parent_args = SimpleNamespace(**{k: Path(v) for k, v in fit_record['paths'].items()})
    bundle = inputs.load_inputs(parent_args)
    saved = io.read(parent_args.neural/'registration.json')
    bundle = fitted.bind_registered_input_identity(bundle, saved)
    arrays, _ = extractor.verify_complete(parent_args.neural, bundle, io.sha(parent_args.neural/'registration.json'))
    rows = sorted([r for r in bundle['rows'] if r['subject'] in ('chemistry', 'physics')], key=lambda r: r['id'])
    if Counter(r['subject'] for r in rows) != {'chemistry': 128, 'physics': 128}:
        raise ValueError('Require every original TRAIN science question')
    summary = io.read(args.verifier/'training-summary.json')
    registered = io.read(args.verifier/'training-registration.json')
    recipe = io.read(args.verifier/'full/recipe.json')
    auxiliary = parent_args.new_data.parent/'science-auxiliary'
    metadata = io.read(auxiliary/'metadata.json'); seal = io.read(auxiliary/'provenance/seal-receipt.json')
    if (summary['training_steps'] != 2160 or summary['training_questions'] != 1440
            or summary['project_dev_or_test_used'] or not summary['calibration_used_reloaded_archive']
            or summary['recipe_sha256'] != io.sha(args.verifier/'full/recipe.json')
            or summary['registration_sha256'] != io.sha(args.verifier/'training-registration.json')
            or recipe['input_profile'] != option_model.PROFILE or recipe['training_questions'] != 1440
            or metadata['protected_split_sha256']['train'] != io.sha(parent_args.new_data/'splits/train-questions.jsonl')
            or seal['metadata_sha256'] != io.sha(auxiliary/'metadata.json') or seal['protected_overlap'] != 0
            or registered['auxiliary_metadata_sha256'] != io.sha(auxiliary/'metadata.json')
            or registered['seal_receipt_sha256'] != io.sha(auxiliary/'provenance/seal-receipt.json')
            or recipe['auxiliary_metadata_sha256'] != registered['auxiliary_metadata_sha256']
            or recipe['protected_split_sha256'] != metadata['protected_split_sha256']
            or recipe['weights'] != summary['calibration_weight_sha256']):
        raise ValueError('Final V3 training/exclusion provenance changed')
    io.verify(registered['source_files'])
    io.verify({str(args.verifier/'full'/n): h for n, h in recipe['weights'].items()})
    lookup = {(r['id'], r['kind'], r['path']): i for i, r in enumerate(bundle['metadata'])}
    old_logits = np.array([margin(arrays['state'][lookup[r['id'], 'original-review', '']] /
                           bundle['identity']['inherited_temperature']) for r in rows])
    judges = {r['id']: bundle['judges'][r['id'], ''] for r in rows}
    trees = {r['id']: {'': bundle['trees'][r['id']]['']} for r in rows}
    evidence = dict(bundle['identity']['input_evidence'])
    assets = [args.parent/'registration.json', args.verifier/'training-summary.json',
              args.verifier/'training-registration.json', auxiliary/'metadata.json', auxiliary/'provenance/seal-receipt.json']
    assets += [p for p in (args.verifier/'full').iterdir() if p.is_file()]
    assets += [parent_args.neural/n for n in ('registration.json', 'neural-complete.json', 'neural-receipt.json', 'metadata.jsonl', 'neural.npz')]
    evidence.update({str(p.resolve()): io.sha(p) for p in assets})
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {n: str(getattr(args, n).resolve()) for n in ('parent', 'verifier', 'output')},
        'questions': 256, 'public_rows_sha256': inputs.fingerprint([public(r) for r in rows]),
        'review_sha256': inputs.fingerprint(judges), 'old_logits_sha256': inputs.fingerprint(old_logits.tolist()),
        'calibration': 'Three whole-question hash folds, subject balanced; standardized logistic C=1; no tuning',
        'variants': ['old', 'option', 'combined'], 'folds': fold_ids(rows).tolist(),
        'screen': 'OOF Brier at least 0.005 lower and OOF AUC strictly higher than old for a new variant',
        'paired_bootstrap': {'draws': 2000, 'seed': SEED, 'scope': 'Descriptive with fitted folds fixed'},
        'parameters': recipe['parameters'], 'temperature': recipe['temperature'], 'device': 'cpu', 'threads': 2,
        'neural_passes': 256, 'trim_actual_inputs': True, 'new_cloud_calls': 0,
        'new_encoder_training': False, 'development_or_test_read': False, 'reference_answers_supplied': False,
        'resume': 'Per-question attempt before forward; unresolved attempts require inspection'}
    return rows, trees, judges, old_logits, record


def restore(args, rows, complete=False):
    attempts = protocol.protocol.strict_records(args.output/'attempts.jsonl')
    saved = protocol.protocol.strict_records(args.output/'observations.jsonl')
    by_id = {r['id']: r for r in rows}; seen = set(); known = {}
    for item in attempts:
        qid = item['id']
        if (qid not in by_id or qid in seen or item != {'id': qid,
                'public_sha256': inputs.fingerprint(public(by_id[qid])), 'registration_sha256': args.registration_sha256}):
            raise ValueError('Changed or duplicate forward attempt')
        seen.add(qid)
    for item in saved:
        qid = item['id']
        if (qid not in seen or qid in known or item['registration_sha256'] != args.registration_sha256
                or not np.isfinite(item['logit']) or type(item['token_slots']) is not int
                or not 0 < item['token_slots'] <= 512): raise ValueError('Changed observation')
        known[qid] = item
    if set(known) != seen: raise ValueError('Unresolved forward attempt; do not automatically repeat')
    if complete and set(known) != set(by_id): raise ValueError('Full256 cohort required before scoring')
    return known


def extract(args, rows, trees, judges, record):
    with io.lock(args.output/'.run.lock'):
        if (args.output/'complete.json').exists(): raise ValueError('Preserve completed extraction')
        known = restore(args, rows); observer = None
        try:
            for row in rows:
                qid = row['id']
                if qid in known: continue
                extractor.memory_guard(args.output, observer is not None)
                if observer is None:
                    observer = option_model.from_checkpoint(args.verifier/'full', 'cpu'); observer.load(); torch.set_num_threads(2)
                io.append(args.output/'attempts.jsonl', {'id': qid, 'public_sha256': inputs.fingerprint(public(row)),
                                                      'registration_sha256': args.registration_sha256})
                start = time.perf_counter(); raw, _, _, slots = observer.values(public(row), '', trees[qid], judges[qid])
                entry = observer.entry(public(row), '', trees[qid], judges[qid])
                item = {'id': qid, 'logit': float(margin(raw)/observer.temperature), 'token_slots': int(slots),
                    'input_sha256': inputs.fingerprint(entry), 'elapsed_seconds': time.perf_counter()-start,
                    'registration_sha256': args.registration_sha256}
                io.append(args.output/'observations.jsonl', item); known[qid] = item; observer._cache.clear()
                if len(known) % 32 == 0: print({'reviews': len(known), 'total': 256}, flush=True)
        finally: observer = None; gc.collect()
        known = restore(args, rows, True); io.verify(record['source_files']); io.verify(record['input_evidence'])
        slots = sum(v['token_slots'] for v in known.values())
        io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
            'digests': {n: io.sha(args.output/n) for n in ('registration.json', 'attempts.jsonl', 'observations.jsonl', 'memory.jsonl')},
            'questions': 256, 'encoder_passes': 256, 'actual_token_slots': slots,
            'new_observer_compute': slots*record['parameters']/1e9/3.09, 'new_cloud_calls': 0, 'development_or_test_read': False})


def grade(args, rows, judges, old_logits, record):
    done = io.read(args.output/'complete.json')
    if done['registration_sha256'] != args.registration_sha256: raise ValueError('Changed completion')
    io.verify({str(args.output/n): h for n, h in done['digests'].items()}); known = restore(args, rows, True)
    if (args.output/'graded').exists(): raise ValueError('Preserve first complete diagnostic')
    new_logits = np.array([known[r['id']]['logit'] for r in rows])
    labels = np.array([int(bool(judges[r['id']]['answer']) and judges[r['id']]['answer'] == r['answer']) for r in rows])
    features = {'old': old_logits[:, None], 'option': new_logits[:, None], 'combined': np.stack([old_logits, new_logits], axis=1)}
    predictions = {k: crossfit(x, labels, np.array(record['folds'])) for k, x in features.items()}
    rng = np.random.default_rng(SEED); boot = rng.integers(0, len(rows), (2000, len(rows)))
    reports = {}
    for name, prob in predictions.items():
        loss = (prob-labels)**2; delta = loss-(predictions['old']-labels)**2
        reports[name] = {'OOF_Brier': float(loss.mean()), 'OOF_AUC': float(roc_auc_score(labels, prob)),
            'Brier_delta_vs_old': float(delta.mean()), 'paired_Brier_delta_CI95': np.quantile(delta[boot].mean(1), [.025, .975]).tolist(),
            'subject_Brier': {s: float(loss[[r['subject'] == s for r in rows]].mean()) for s in ('chemistry', 'physics')}}
    for name, report in reports.items():
        report['screen_passed'] = name != 'old' and report['Brier_delta_vs_old'] <= -.005 and report['OOF_AUC'] > reports['old']['OOF_AUC']
    summary = {'schema': SCHEMA, 'questions': 256, 'review_correct': int(labels.sum()), 'reports': reports,
        'mean_new_observer_compute': done['new_observer_compute']/256,
        'combined_cost': 'Runtime must charge both old and option observer forwards for each review',
        'new_cloud_calls': 0, 'development_or_test_read': False, 'fresh_validation': False,
        'controller_accuracy_claim': False, 'new_solver_answer_availability': False,
        'complete_sha256': io.sha(args.output/'complete.json')}
    (args.output/'graded').mkdir(); io.write(args.output/'graded/summary.json', summary)
    io.npz(args.output/'graded/predictions.npz', {'labels': labels, 'old_logits': old_logits, 'option_logits': new_logits, **predictions})
    print(summary, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('plan', 'register', 'extract', 'grade'))
    for n in ('parent', 'verifier', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with guarded.training_only_reads():
        with guarded.guarded.no_model_calls(): rows, trees, judges, old, record = load_inputs(args)
        if args.phase == 'plan': print({'questions': len(rows), 'passes': 256, 'cloud_calls': 0}); return
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Registered diagnostic changed')
        if args.phase == 'extract': extract(args, rows, trees, judges, record)
        else:
            with guarded.guarded.no_model_calls(): grade(args, rows, judges, old, record)


if __name__ == '__main__': main()
