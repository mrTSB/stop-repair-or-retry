from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import science_review_critic as critic


class ReviewCriticTests(unittest.TestCase):
    def test_public_input_excludes_references_and_fold_ignores_labels(self):
        rows = [{'id': str(i), 'subject': 'physics', 'question': 'Q', 'answer': 'A',
                 'reference': 'secret', 'solution': 'secret'} for i in range(18)]
        changed = [{**r, 'answer': 'B', 'reference': 'different'} for r in rows]
        self.assertEqual([critic.public(r) for r in rows], [critic.public(r) for r in changed])
        self.assertNotIn('reference', critic.public(rows[0]))
        np.testing.assert_array_equal(critic.fold_ids(rows), critic.fold_ids(changed))

    def test_held_labels_cannot_change_own_calibrated_predictions(self):
        rng = np.random.default_rng(3); x = rng.normal(size=(60, 2)); y = np.arange(60) % 2
        folds = np.arange(60) % 3; before = critic.crossfit(x, y, folds)
        changed = y.copy(); changed[folds == 1] = 1-changed[folds == 1]
        after = critic.crossfit(x, changed, folds)
        np.testing.assert_array_equal(before[folds == 1], after[folds == 1])

    def test_unresolved_forward_cannot_be_repeated(self):
        row = {'id': 'q', 'subject': 'physics', 'question': 'Q', 'answer': 'A'}
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), registration_sha256='reg')
            critic.io.append(args.output/'attempts.jsonl', {'id': 'q',
                'public_sha256': critic.inputs.fingerprint(critic.public(row)), 'registration_sha256': 'reg'})
            with self.assertRaisesRegex(ValueError, 'Unresolved'): critic.restore(args, [row])
            critic.io.append(args.output/'observations.jsonl', {'id': 'q', 'registration_sha256': 'reg',
                'logit': 0., 'token_slots': 30})
            self.assertEqual(len(critic.restore(args, [row], True)), 1)

    def test_margin_matches_tempered_four_state_correctness(self):
        raw = np.array([1., 3., -2., 4.]); temperature = 2.3
        p = np.exp(raw/temperature); p /= p.sum()
        logit = critic.margin(raw/temperature)
        self.assertAlmostEqual(float(1/(1+np.exp(-logit))), float(p[1]+p[3]))


if __name__ == '__main__': unittest.main()
