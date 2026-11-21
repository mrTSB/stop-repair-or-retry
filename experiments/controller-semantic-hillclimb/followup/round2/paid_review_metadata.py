"""Cheap, explicitly paid root-review correctness adapter; no neural operations."""
from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np

import review_metadata_diagnostic as diagnostic
import review_metadata_spec as spec


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def public_entry(row, current, judge):
    """Whitelist exactly the paid fields used by the frozen diagnostic."""
    if row.get('answer'):
        raise ValueError('Blank the reference before metadata inference')
    return {'row': {'subject': row['subject']},
            'current': {key: current[key] for key in ('answer', 'text')},
            'judge': {**{key: judge[key] for key in ('answer', 'action', 'parse_valid',
                       'physical_api_calls', 'input_tokens', 'output_tokens')},
                'json_complete': judge.get('json_complete', False),
                'p_correct': judge.get('p_correct', .5),
                'verdict': {key: judge['verdict'].get(key, '') for key in ('solution', 'check')},
                'calls': [{key: call.get(key) for key in ('input_tokens', 'output_tokens', 'output_cap', 'finish')}
                          for call in judge['calls']]}, 'protocol': 'root'}


def signal(judge, current, probability):
    # The inherited signal API accepts a four-state difficulty/correctness
    # distribution but consumes only P(correct)=p[1]+p[3]. No difficulty claim.
    import mentor_pomdp
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid metadata correctness probability')
    return mentor_pomdp.signal(judge, current, np.asarray([1. - probability, probability, 0., 0.]))


@dataclass
class Adapter:
    fitted: dict
    temperature: float
    receipt: dict

    def state(self):
        pipeline = self.fitted['model']
        scaler, classifier = pipeline['scale'], pipeline['model']
        return {'features': spec.FEATURES, 'indexes': self.fitted['indexes'],
            'mean': scaler.mean_.tolist(), 'variance': scaler.var_.tolist(),
            'scale': scaler.scale_.tolist(), 'classes': classifier.classes_.tolist(),
            'coefficients': classifier.coef_.tolist(), 'intercepts': classifier.intercept_.tolist(),
            'temperature': self.temperature}

    def validate(self):
        state = self.state()
        arrays = [state[key] for key in ('mean', 'variance', 'scale', 'coefficients', 'intercepts')]
        if (not all(np.isfinite(value).all() for value in arrays)
                or np.min(state['scale']) <= 0 or not 0 < self.temperature < float('inf')
                or state['classes'] != [0, 1]
                or state['indexes'] != list(range(len(spec.FEATURES)))
                or digest(state) != self.receipt['learned_state_sha256']):
            raise ValueError('Changed or nonfinite metadata model')

    def predict_paid(self, row, current, judge):
        started = time.perf_counter()
        self.validate()
        entry = public_entry(row, current, judge)
        values = diagnostic.features(entry['row'], entry['current'], entry['judge'], 'root')
        record = {'features': values.tolist(), 'answer_present': bool(judge['answer'])}
        raw = diagnostic.predict(self.fitted, [record])
        probability = float(diagnostic.calibrated(raw, self.temperature, [record])[0])
        if not np.isfinite(probability):
            raise ValueError('Nonfinite metadata probability')
        return probability, {'kind': 'paid-review-metadata', 'path': '',
            'input_sha256': digest(entry), 'feature_sha256': digest(values.tolist()),
            'model_sha256': self.receipt['learned_state_sha256'],
            'probability': probability, 'cpu_seconds': time.perf_counter() - started,
            'neural_calls': 0, 'encoder_token_slots': 0, 'include_reviews': False}
