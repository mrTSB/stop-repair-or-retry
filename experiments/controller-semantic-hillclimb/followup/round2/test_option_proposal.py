"""No model/cloud calls: selection, blind ranking, feedback, and full costs."""

import unittest

import numpy as np

from option_proposal_pairs import selection, options, rank_options, feedback, charges


class FakeObserver:
    temperature = 2.
    _recipe = {'parameters': 400000000}

    def __init__(self):
        self.seen = []

    def entry(self, row, path, tree):
        assert row['answer'] == ''
        assert set(tree) == {path} and set(tree[path]) == {'answer'}
        return {'left': row['question'], 'right': tree[path]['answer']}

    def values(self, row, path, tree):
        self.entry(row, path, tree)
        letter = tree[path]['answer']
        self.seen.append(letter)
        score = {'A': 0., 'B': 6., 'C': 2., 'D': 4.}[letter]
        return np.array([0., score, 0., score]), np.zeros(8), np.zeros(8), ord(letter) - 60


class ProposalTests(unittest.TestCase):
    def row(self):
        return {'id': 'synthetic', 'subject': 'chemistry', 'answer': 'D',
                'question': 'Question\n\nOptions:\nA. red\nB. blue\nC. green\nD. white'}

    def test_fixed_selection_is_outcome_and_order_blind(self):
        rows = [{'id': f'{s}-{i}', 'subject': s, 'answer': 'A'}
                for s in ('chemistry', 'physics') for i in range(50)]
        chosen = selection(rows)
        self.assertEqual(len(chosen), 64)
        self.assertEqual(sum(r['subject'] == 'chemistry' for r in chosen), 32)
        self.assertEqual(chosen, selection([{**r, 'answer': 'wrong'} for r in reversed(rows)]))

    def test_all_options_ranked_with_blank_reference(self):
        observer = FakeObserver()
        proposal = rank_options(observer, self.row())
        self.assertEqual(observer.seen, list('ABCD'))
        self.assertEqual(proposal['proposed_letter'], 'B')
        self.assertEqual(proposal['proposed_option'], 'blue')
        self.assertAlmostEqual(proposal['calibrated_logit'], 3.)
        self.assertAlmostEqual(proposal['calibrated_margin'], 1.)
        self.assertEqual(proposal, rank_options(FakeObserver(), {**self.row(), 'answer': 'A'}))
        self.assertIn('may be wrong', feedback(proposal))
        self.assertEqual(proposal['encoder_calls'], 4)
        self.assertEqual(proposal['encoder_token_slots'], 26)

    def test_all_passes_and_paid_solver_prefix_charged(self):
        p = rank_options(FakeObserver(), self.row())
        cost = charges({'input_tokens': 10, 'output_tokens': 20},
                       {'input_tokens': 30, 'output_tokens': 40}, p)
        self.assertEqual(cost['solver_tokens'], 100)
        self.assertEqual(cost['encoder_calls'], 4)
        self.assertAlmostEqual(cost['compute'], (4 * 100 + .4 * 26) / 3.09)

    def test_preserves_official_option_order(self):
        self.assertEqual(options(self.row()), ['red', 'blue', 'green', 'white'])
        with self.assertRaises(ValueError):
            options({**self.row(), 'question': 'Question\n\nOptions:\nB. blue\nA. red\nC. green\nD. white'})


if __name__ == '__main__':
    unittest.main()
