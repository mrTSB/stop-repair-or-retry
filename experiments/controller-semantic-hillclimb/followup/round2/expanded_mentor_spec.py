"""Explicit training weights for the new320-root expanded mentor variant.

This is a different calibration/counting variant, not a pure more-data
ablation. No frozen fit contract is relaxed. This module reads no data.
"""
from collections import Counter
import hashlib
import json
import math

SCHEMA = 'round2-expanded-mentor-weighted-fit-v1'
SUBJECTS = ('math', 'chemistry', 'physics')
TARGET = {'math': .5, 'chemistry': .25, 'physics': .25}
POOLED = {'math': 4., 'chemistry': 1., 'physics': 1.}
PLAIN_COUNTS = {'math': 256, 'chemistry': 128, 'physics': 128}
REVIEW_COUNTS = {'math': 64, 'chemistry': 128, 'physics': 128}
CALIBRATION_PATHS = ('', 'repair', 'retry', 'guided')
SMOOTHING = (8., 32.)
SEED = 2026112307


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def question_weights(records):
    """Subject-balanced question mass, then equal mass across a qid's rows.

The input must contain only the current fitting fold and eligible rows.
Total weight is its number of independent questions. This explicitly makes
C1 regularization independent of duplicated siblings, not dependent on the
number of rows. All weighting depends on observed IDs/subjects only.
"""
    if not records:
        raise ValueError('No eligible question rows')
    subject_by_id = {}
    counts = Counter()
    for record in records:
        qid, subject = record['id'], record['subject']
        if not isinstance(qid, str) or not qid or subject not in SUBJECTS:
            raise ValueError('Invalid observed question/subject')
        if qid in subject_by_id and subject_by_id[qid] != subject:
            raise ValueError('A question changed observed subject')
        subject_by_id[qid] = subject
        counts[qid] += 1
    subjects = Counter(subject_by_id.values())
    if set(subjects) != set(SUBJECTS):
        raise ValueError('Every fitting fold must contain all three observed subjects')
    n = len(subject_by_id)
    values = [n * TARGET[r['subject']] / subjects[r['subject']] / counts[r['id']] for r in records]
    if not math.isclose(math.fsum(values), n, rel_tol=1e-12):
        raise ValueError('Question mass did not normalize')
    return values


def weight_receipt(records, weights):
    if len(records) != len(weights) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Invalid fit weights')
    by_id = {}
    for row in records:
        by_id.setdefault(row['id'], row['subject'])
        if by_id[row['id']] != row['subject']:
            raise ValueError('A question changed observed subject')
    return {'records': len(records), 'questions': len(by_id),
            'questions_by_subject': dict(sorted(Counter(by_id.values()).items())),
            'subject_mass': {s: math.fsum(w for r, w in zip(records, weights) if r['subject'] == s)
                             for s in SUBJECTS},
            'total_mass': math.fsum(weights), 'normalization_unit': 'independent eligible question',
            'input_order_sha256': digest([(r['id'], r['subject']) for r in records]),
            'weights_sha256': digest(list(weights))}


def calibration_fold(qid):
    return int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 3


def validate_counts(rows, reviewed_ids):
    if len(rows) != 512 or len({r['id'] for r in rows}) != 512:
        raise ValueError('Require all512 unique new TRAIN questions')
    if Counter(r['subject'] for r in rows) != PLAIN_COUNTS:
        raise ValueError('Changed512 TRAIN observed subject counts')
    reviewed_ids = set(reviewed_ids)
    if len(reviewed_ids) != 320 or reviewed_ids - {r['id'] for r in rows}:
        raise ValueError('Require all320 registered new-source roots')
    if Counter(r['subject'] for r in rows if r['id'] in reviewed_ids) != REVIEW_COUNTS:
        raise ValueError('Changed320 paired observed subject counts')
