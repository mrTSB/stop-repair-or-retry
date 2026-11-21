"""Fixed training-only screen after the entire192-call scoring grid is sealed.

Two C1/T1 eight-state forecasts use the same64 questions, three grouped folds,
and cached numerical/neural features. This is not a controller evaluation.
"""
from collections import Counter
import json
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits

import scoring_observation_pilot as pilot
import training_sparse_text as prior

core = prior.sparse_core
VARIANTS = ('numeric_C1', 'numeric_plus_scoring_C1')
PANELS = ('all', 'chemistry', 'physics', 'root', 'repair', 'retry', *[
    subject + '_' + path for subject in ('chemistry', 'physics') for path in ('root', 'repair', 'retry')])


def verify_archive(directory, all_data, provenance):
    """Verify the existing all512 numeric OOF context; it is not the screen control."""
    summary = pilot.read(directory / 'summary.json')
    registration = pilot.read(directory / 'registration.json')
    if (summary['registration_sha256'] != pilot.sha(directory / 'registration.json')
            or summary['predictions_sha256'] != pilot.sha(directory / 'private-oof.npz')
            or summary['fold_receipts_sha256'] != pilot.sha(directory / 'fold-receipts.json')
            or summary['alignment_sha256'] != pilot.sha(directory / 'training-alignment.json')
            or summary.get('training_questions') != 512 or summary.get('decision_nodes') != 1536
            or summary.get('test_used') is not False or summary.get('development_values_read') is not False
            or summary.get('neural_calls') != 0 or summary.get('cloud_calls') != 0):
        raise ValueError('Archived numeric OOF completion changed')
    for path, value in {**registration['source_files'], **registration['input_evidence']}.items():
        if pilot.sha(path) != value:
            raise ValueError('Archived numeric OOF source or input changed: ' + path)
    for key, value in provenance.items():
        if registration.get(key) != value:
            raise ValueError('Archived numeric OOF input alignment differs: ' + key)
    folds = np.asarray([core.fold_for(q) for q in all_data['ids']])
    receipts = pilot.read(directory / 'fold-receipts.json')['numeric_C1']
    if len(receipts) != 3 or {r['fold'] for r in receipts} != {0, 1, 2}:
        raise ValueError('Require all three archived numeric question-held folds')
    for receipt in receipts:
        fold = receipt['fold']
        fit_ids = sorted(set(all_data['ids'][folds != fold]))
        held_ids = sorted(set(all_data['ids'][folds == fold]))
        if (receipt['fit_ids_sha256'] != core.fingerprint(fit_ids)
                or receipt['held_ids_sha256'] != core.fingerprint(held_ids)
                or receipt['question_overlap'] != 0 or receipt['temperature'] != 1.
                or receipt['model_sha256'] != pilot.sha(directory / f'numeric_C1-fold{fold}.pkl')):
            raise ValueError('Archived numeric predictions do not exclude each held question')
    with np.load(directory / 'private-oof.npz') as saved:
        probabilities = saved['numeric_C1'].copy()
    core.losses(all_data['labels'], probabilities)
    return probabilities


def aligned_data(all_data, public_rows, trees, known, expected_questions=64):
    """Use all three original prefixes for every fixed question, including censored rows."""
    row_by_id = {r['id']: r for r in public_rows}
    if len(row_by_id) != expected_questions or len(public_rows) != expected_questions:
        raise ValueError('Every fixed question must occur exactly once')
    expected = {(qid, path) for qid in row_by_id for path in pilot.PATHS}
    keys = list(zip(all_data['ids'].tolist(), all_data['paths'].tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate original cached decision node')
    indexes = [i for i, key in enumerate(keys) if key[0] in row_by_id]
    if {keys[i] for i in indexes} != expected or set(known) != expected:
        raise ValueError('Require the exact full cohort times all three original prefixes')
    scores, alignment = [], []
    for i in indexes:
        qid, path = keys[i]; item = known[qid, path]
        _, prefix = pilot.request_payload(row_by_id[qid], path, trees[qid])
        if (item['prefix'] != prefix or item['original_answer'] != trees[qid][path]['answer']
                or item['subject'] != all_data['subjects'][i]
                or len(item['features']) != len(pilot.FEATURES)):
            raise ValueError('Scoring features changed original prefix/answer/cache alignment')
        scores.append(item['features'])
        alignment.append({'id': qid, 'path': path, 'original_cache_index': i,
            'original_answer': item['original_answer'], 'input_sha256': prefix['prompt_sha256'],
            'scoring_record_sha256': pilot.digest(item)})
    data = {key: ([value[i] for i in indexes] if isinstance(value, list) else value[indexes].copy())
            for key, value in all_data.items()}
    data['scoring'] = np.asarray(scores, dtype=float)
    if not np.isfinite(data['scoring']).all():
        raise ValueError('Nonfinite scoring feature or missingness sentinel')
    if set(Counter(data['ids']).values()) != {3}:
        raise ValueError('Equal question weights require every root/repair/retry triplet')
    return data, indexes, alignment


def crossfit(data, variant):
    if variant not in VARIANTS:
        raise ValueError('Only the two fixed C1/T1 feature configurations are permitted')
    numeric = data['numeric'] if variant == VARIANTS[0] else np.hstack((data['numeric'], data['scoring']))
    inputs = {**data, 'numeric': numeric}
    predictions = np.full((len(data['labels']), 8), np.nan)
    receipts, models = [], []
    for fold in (0, 1, 2):
        held, probabilities, model, receipt = core.outer_prediction(inputs, 'numeric_C1', fold)
        predictions[held] = probabilities
        receipts.append({**receipt, 'variant': variant,
            'fit_node_weight': 1., 'fit_question_weight': 3.,
            'equal_question_weights': True, 'scoring_columns': 0 if variant == VARIANTS[0] else len(pilot.FEATURES)})
        models.append(model)
    core.losses(data['labels'], predictions)
    return predictions, receipts, models


def mask(data, panel):
    selected = np.ones(len(data['ids']), dtype=bool)
    for part in panel.split('_'):
        if part in ('chemistry', 'physics'):
            selected &= data['subjects'] == part
        elif part in ('root', 'repair', 'retry'):
            selected &= data['paths'] == ('' if part == 'root' else part)
        elif part != 'all':
            raise ValueError('Unknown fixed reporting panel')
    return selected


def reports(data, predictions, known):
    losses = {name: core.losses(data['labels'], values) for name, values in predictions.items()}
    result = {}
    for panel in PANELS:
        selected = mask(data, panel)
        qids = sorted(set(data['ids'][selected]))
        if not qids:
            raise ValueError('Empty fixed subject/path reporting panel')
        groups = [np.flatnonzero(selected & (data['ids'] == qid)) for qid in qids]
        draws = np.random.default_rng(core.SEED).integers(0, len(qids), size=(2000, len(qids)))
        rows = [known[data['ids'][i], data['paths'][i]] for i in np.flatnonzero(selected)]
        valid = sum(r['original_answer_valid'] for r in rows)
        observed = sum(r['original_answer_valid'] and r['current_letter_observed'] for r in rows)
        part = {'questions': len(qids), 'decision_nodes': int(selected.sum()), 'models': {}, 'paired_differences': {},
            'coverage': {'originally_valid_answers': valid, 'invalid_original_answers': len(rows) - valid,
                'valid_current_letter_observed': observed,
                'valid_current_letter_coverage': observed / valid if valid else None,
                'denominator': 'All originally valid current answers, not only observed or correctly answered rows'},
            'observation_cost': {'physical_calls': len(rows), 'input_tokens': sum(r['observer_input_tokens'] for r in rows),
                'output_tokens': sum(r['observer_output_tokens'] for r in rows),
                'active_parameter_token_proxy': sum(r['observer_compute'] for r in rows),
                'summed_physical_latency_seconds': sum(r['latency_seconds'] for r in rows),
                'mean_tokens_per_decision': float(np.mean([r['observer_input_tokens']+r['observer_output_tokens'] for r in rows])),
                'new_answer_changing_calls': 0, 'new_neural_calls': 0, 'review_calls': 0}}
        for name, values in losses.items():
            part['models'][name] = {metric: float(np.mean([values[metric][g].mean() for g in groups])) for metric in core.METRICS}
            if name == 'numeric_C1':
                continue
            differences = {}
            for metric in core.METRICS:
                diff = values[metric] - losses['numeric_C1'][metric]
                per_question = np.asarray([diff[g].mean() for g in groups])
                differences[metric] = {'point': float(per_question.mean()),
                    'ci95': np.quantile(per_question[draws].mean(1), [.025, .975]).tolist(), 'lower_is_better': True}
            part['paired_differences'][name + '_minus_numeric_C1'] = differences
        result[panel] = part
    return result


def screen(reported):
    all_rows = reported['all']
    diff = all_rows['paired_differences']['numeric_plus_scoring_C1_minus_numeric_C1']
    coverage = all_rows['coverage']['valid_current_letter_coverage']
    tests = {'joint_logloss_gain': diff['joint_log_loss']['point'] <= -pilot.SCREEN['joint_logloss_improvement_at_least'],
        'joint_logloss_paired_upper95_below_zero': diff['joint_log_loss']['ci95'][1] < 0.,
        'current_repair_retry_brier_each_no_worse': all(diff[k]['point'] <= 0. for k in ('current_brier', 'repair_brier', 'retry_brier')),
        'current_letter_coverage_at_least_80percent': coverage is not None and coverage >= .8}
    return {'criteria': pilot.SCREEN, 'passed': all(tests.values()), 'checks': tests,
            'controller_promoted': False, 'scope': pilot.SCREEN['scope']}


def run(args):
    # Seal verification always precedes any training reference, outcome, or cache-label read.
    rows, trees, known, complete = pilot.validated_complete(args)
    registration = pilot.read(args.output / 'registration.json')
    target = args.output / 'diagnostic'
    if target.exists():
        raise ValueError('Preserve the existing fixed diagnostic; do not search or overwrite it')
    with prior.no_model_calls(), threadpool_limits(limits=1):
        all_data, _, provenance = prior.load_inputs(args)
        archived = verify_archive(args.baseline_diagnostic, all_data, provenance)
        data, indexes, alignment = aligned_data(all_data, rows, trees, known)
        target.mkdir()
        pilot.write(target / 'input-receipt.json', {'complete_sha256': pilot.sha(args.output / 'complete.json'),
            'registration_sha256': pilot.sha(args.output / 'registration.json'), 'prior_cache_provenance': provenance,
            'aligned_numeric_sha256': core.array_sha(data['numeric']), 'scoring_features_sha256': core.array_sha(data['scoring']),
            'labels_sha256': core.array_sha(data['labels']), 'alignment': alignment, 'training_only': True,
            'development_or_test_outcomes_used': False, 'new_neural_calls': 0})
        predictions = {'archived_all512_numeric_OOF_context_only': archived[indexes]}
        receipts = {}
        for variant in VARIANTS:
            values, receipt, models = crossfit(data, variant)
            predictions[variant] = values
            for fold, model in enumerate(models):
                path = target / f'{variant}-fold{fold}.pkl'
                path.write_bytes(pickle.dumps(model, protocol=5))
                receipt[fold]['model_sha256'] = pilot.sha(path)
            receipts[variant] = receipt
        np.savez_compressed(target / 'private-oof.npz', **predictions)
        pilot.write(target / 'fold-receipts.json', receipts)
        reported = reports(data, predictions, known)
        for path, value in {**registration['source_files'], **registration['input_evidence']}.items():
            if pilot.sha(path) != value:
                raise ValueError('Registered source or input changed during the fixed diagnostic')
        pilot.validated_complete(args)
        result = {'schema': pilot.SCHEMA, 'questions': 64, 'decision_nodes': 192,
            'source_registration_sha256': pilot.sha(args.output / 'registration.json'),
            'complete_sha256': pilot.sha(args.output / 'complete.json'),
            'input_receipt_sha256': pilot.sha(target / 'input-receipt.json'),
            'fold_receipts_sha256': pilot.sha(target / 'fold-receipts.json'),
            'predictions_sha256': pilot.sha(target / 'private-oof.npz'),
            'reports': reported, 'screen': screen(reported), 'training_only': True,
            'test_used': False, 'development_outcomes_used': False, 'POMDP_runtime_changed': False,
            'new_neural_calls': 0, 'cloud_calls_in_diagnostic': 0, 'paid_observation_calls': 192,
            'answers_changed': 0, 'physical_complete': complete['coverage'],
            'transport_error_records': complete['transport_error_records'],
            'acquisition_compute_is_lower_bound': complete['acquisition_compute_is_lower_bound'],
            'model_parameters': {**core.MODEL_PARAMETERS, 'temperature': 1., 'node_weight': 1., 'question_weight': 3.},
            'interpretation': 'Same64-question grouped OOF observation screen. The all512 archived model is context only. No policy accuracy, fresh result, or historical-logit recovery is claimed. All original answers and192 observations remain retained.'}
        pilot.write(target / 'summary.json', result)
        print(json.dumps({'complete': True, 'training_only': True, 'screen': result['screen'],
            'summary_sha256': pilot.sha(target / 'summary.json')}), flush=True)
