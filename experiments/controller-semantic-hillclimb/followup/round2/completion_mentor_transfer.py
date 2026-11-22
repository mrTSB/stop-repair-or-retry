"""Fixed TRAIN-only16-to8 transfer specification and path-availability audit.

No observer, classifier, emission or mentor dynamics are refitted here. The
kernel estimator is exposed for reviewed future fitting; preparation can join
and audit all256 science questions without invoking it. No DEV/test reader.
"""
from collections import Counter
import hashlib
import json

import numpy as np

import completion_mentor_hybrid as runtime

plain, mentor = runtime.plain, runtime.mentor_core
SMOOTHING = 32.
PLAIN_PATHS = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
COMPLETE_PATHS = ('complete', 'repair/complete', 'retry/complete')
STRONG_PATHS = ('guided', 'guided/reuse', 'guided/retry', 'retry/reuse')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def hardness_receipt():
    """Bind the exact original mentor TRAIN function, paths and strict threshold."""
    return {'definition': 'int(mean(correct(tree[p], reference) for p in FIXED) < 0.6)',
            'fixed_paths': list(mentor.core.FIXED), 'threshold': .6, 'comparison': 'strictly_less',
            'supported_parent_functions': runtime.parent_hardness_sources(),
            'state_index': '2*(2*hard + current_correct) + mentor_correct',
            'uses_complete_or_guided_outcomes': False}


def join(rows, complete_trees, expanded_rows, expanded_trees, judges):
    """Full fixed cohort only; exact question/reference/root/plain response join."""
    expected = {'chemistry': 128, 'physics': 128}
    if (len(rows) != 256 or len({r['id'] for r in rows}) != 256
            or Counter(r['subject'] for r in rows) != expected):
        raise ValueError('Use all256 TRAIN science questions with the fixed128/128 mix')
    expanded = {r['id']: r for r in expanded_rows}
    science_ids = {r['id'] for r in expanded_rows if r['subject'] in runtime.SCIENCE}
    if len(expanded) != len(expanded_rows) or science_ids != {r['id'] for r in rows}:
        raise ValueError('Expanded strong cohort and Complete science cohort differ')
    records, joined = [], {}
    for row in rows:
        qid = row['id']; complete, strong = complete_trees[qid], expanded_trees[qid]
        if row != expanded[qid] or (qid, '') not in judges:
            raise ValueError('Question/reference changed or paid root review absent')
        if not set((*PLAIN_PATHS, *COMPLETE_PATHS)) <= set(complete):
            raise ValueError('Complete all ten plain/Complete paths before any fit/audit')
        if not set((*PLAIN_PATHS, *STRONG_PATHS)) <= set(strong):
            raise ValueError('Complete the full retained mentor path grid')
        if any(complete[path] != strong[path] for path in PLAIN_PATHS):
            raise ValueError('Original solver root/continuation controls changed')
        if strong['guided/retry'] != strong['repair/retry']:
            raise ValueError('Pure-retry alias is not another independent response')
        tree = {**complete, **{path: strong[path] for path in STRONG_PATHS}}
        correctness = lambda path: bool(mentor.correct(tree[path], row['answer']))
        current = correctness('')
        source = plain.state(current, correctness('repair'), correctness('retry'), correctness('complete'))
        hard = int(np.mean([correctness(path) for path in mentor.core.FIXED]) < .6)
        target = mentor.state(2 * hard + current, mentor.correct(judges[qid, ''], row['answer']))
        if plain.CURRENT[source] != mentor.REWARD[target]:
            raise ValueError('Current-answer truth changed between observed modes')
        records.append({'id': qid, 'subject': row['subject'], 'source': int(source), 'target': int(target),
            'hard': hard, 'reference_sha256': fingerprint(row['answer']),
            'public_question_sha256': fingerprint({**row, 'answer': ''}),
            'root_sha256': fingerprint(tree['']), 'review_sha256': fingerprint(judges[qid, '']),
            'plain_and_complete_sha256': fingerprint({p:tree[p] for p in (*PLAIN_PATHS, *COMPLETE_PATHS)})})
        joined[qid] = tree
    receipt = {'schema': 'science-complete-mentor-transfer-inputs-v1', 'paired_questions': 256,
        'subject_counts': expected, 'source_bits': runtime.SOURCE_BITS, 'target_states': runtime.TARGET_STATES,
        'smoothing': SMOOTHING, 'supported_pseudocount': .1, 'hardness': hardness_receipt(),
        'records_sha256': fingerprint(records), 'joint_tree_sha256': fingerprint(joined),
        'question_selection': 'All256 science TRAIN; no correctness or source-subset filtering',
        'new_neural_or_cloud_calls': 0, 'development_or_test_used': False}
    return records, joined, receipt


def fit(records, input_receipt):
    """Fixed pooled+subject shrinkage, preserving the source/target C bit exactly."""
    if (len(records) != 256 or len({r['id'] for r in records}) != 256
            or Counter(r['subject'] for r in records) != {'chemistry':128, 'physics':128}
            or input_receipt.get('records_sha256') != fingerprint(records)
            or input_receipt.get('smoothing') != SMOOTHING
            or input_receipt.get('hardness') != hardness_receipt()):
        raise ValueError('Changed immutable full256 transfer input recipe')
    support = (plain.CURRENT[:, None] == mentor.REWARD[None, :]).astype(float)
    counts = .1 * support.copy()
    for record in records:
        s, t = record['source'], record['target']
        if s not in range(16) or t not in range(8) or not support[s, t]:
            raise ValueError('Invalid hidden-state label or forbidden current-truth transition')
        counts[s, t] += 1
    pooled = counts / counts.sum(1, keepdims=True)
    result = {}
    for subject in runtime.SCIENCE:
        local = .1 * support.copy()
        for record in records:
            if record['subject'] == subject: local[record['source'], record['target']] += 1
        local += SMOOTHING * pooled
        result[subject] = local / local.sum(1, keepdims=True)
    runtime.validate_transfer(result)
    return result, {**input_receipt, 'learned_parameters': 'Only new16x8 transfer kernels',
                   'mentor_dynamics_refitted': False, 'neural_or_classifier_refitted': False,
                   'fit_uses_training_labels': True, 'subject_question_weights': 1.}


def availability(rows, trees):
    if len(rows) != 256 or Counter(r['subject'] for r in rows) != {'chemistry':128, 'physics':128}:
        raise ValueError('Availability requires the entire fixed TRAIN science panel')
    groups = {'original7': PLAIN_PATHS, 'plain_plus_Complete': (*PLAIN_PATHS,*COMPLETE_PATHS),
              'strong10_distinct': (*PLAIN_PATHS,*STRONG_PATHS),
              'combined13_distinct': (*PLAIN_PATHS,*STRONG_PATHS,*COMPLETE_PATHS)}
    panels = {}
    for subject in ('all_science', *runtime.SCIENCE):
        panel = [r for r in rows if subject == 'all_science' or r['subject'] == subject]
        values = {name: [any(mentor.correct(trees[r['id']][p],r['answer']) for p in paths) for r in panel]
                  for name,paths in groups.items()}
        panels[subject] = {'questions':len(panel), 'hindsight_available_correct':{k:sum(v) for k,v in values.items()},
            'Complete_novel_beyond_strong_tree':sum(a and not b for a,b in zip(
                values['combined13_distinct'], values['strong10_distinct'])),
            'strong_novel_beyond_plain_Complete':sum(a and not b for a,b in zip(
                values['combined13_distinct'], values['plain_plus_Complete']))}
    return {'schema':'science-complete-mentor-hindsight-availability-v1', 'panels':panels,
        'interpretation':'Hindsight action availability only; neither controller accuracy nor a learnability claim',
        'training_only':True,'development_or_test_used':False,'new_calls':0,
        'logical_paths':14,'distinct_solver_responses':13,'guided_retry_is_existing_repair_retry':True}
