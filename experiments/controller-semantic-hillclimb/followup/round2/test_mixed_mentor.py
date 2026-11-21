import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import prepare_mixed_mentor as module
from development_mixed_mentor import select


class MixedMentorTest(unittest.TestCase):
    def test_plain_reader_uses_sealed_prefix_during_concurrent_append(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / 'splits').mkdir()
            (data / 'splits/train-questions.jsonl').write_text(json.dumps({'id': 'q'}) + '\n')
            raw = ''.join(json.dumps({'id': 'q', 'kind': 'solver', 'path': p, 'response': {'answer': '1'}}) + '\n'
                          for p in module.PLAIN).encode()
            (data / 'train-plain-complete.json').write_text(json.dumps({
                'base_records': 7, 'base_bank_sha256': hashlib.sha256(raw).hexdigest()}))
            bank = data / 'train-observations.jsonl'
            bank.write_bytes(raw + b'{"incomplete_append":')
            rows, trees = module.plain_data(data, 'train')
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(trees['q']), set(module.PLAIN))
            bank.write_bytes(raw.replace(b'"1"', b'"2"', 1))
            with self.assertRaises(ValueError):
                module.plain_data(data, 'train')

    def test_selection_requires_paid_reviews_and_strict_B_cost_win(self):
        def candidate(accuracy, cost, calls):
            return {'dev': {'accuracy': accuracy, 'mean_compute': cost, 'mean_judge_calls': calls}}
        expensive = candidate(.9, 1700., 1.)
        equal_cost = candidate(.89, 1600., 1.)
        cheap = candidate(.85, 1500., .7)
        no_review = candidate(.95, 1000., 0.)
        result = select([expensive, equal_cost, cheap, no_review], 1600.)
        self.assertIs(result['best'], no_review)
        self.assertIs(result['best_queried'], expensive)
        self.assertIs(result['selected_C'], cheap)
        self.assertIsNone(select([expensive, equal_cost, no_review], 1600.)['selected_C'])
        self.assertIsNone(select([cheap])['selected_C'])

    def fixture(self):
        row = {'id': 'synthetic', 'answer': '1', 'subject': 'math'}
        node = {'answer': '1', 'text': 'paid', 'prompt_sha256': 'prompt'}
        tree = {p: dict(node) for p in module.PLAIN}
        tree.update({'guided': {**node, 'text': 'OLD GUIDED'},
                     'guided/repair': {**node, 'text': 'OLD GUIDED CHILD'},
                     'retry/reviewed': {**node, 'text': 'ORDINARY REVIEW'}})
        records = [{'id': row['id'], 'kind': kind, 'path': path, 'dose': 1024,
                    'response': {**node, 'finish': 'stop', 'text': 'NEW ' + path}}
                   for kind, path in [('judge', ''), ('solver', 'guided'), ('solver', 'guided/retry'),
                                      ('solver', 'guided/reuse'), ('solver', 'retry/reuse')]]
        return row, tree, records

    def test_overlay_removes_incompatible_ordinary_descendants(self):
        row, tree, records = self.fixture()
        original = copy.deepcopy(tree)
        trees, judges = module.overlay([row], {row['id']: tree}, records, [row], validate=False)
        self.assertEqual(tree, original)
        self.assertEqual(trees[row['id']]['guided']['text'], 'NEW guided')
        self.assertNotIn('guided/repair', trees[row['id']])
        self.assertNotIn('retry/reviewed', trees[row['id']])
        self.assertEqual(len(judges), 1)
        for bad in (records[:-1], records + records[:1], [{**records[0], 'dose': 512}] + records[1:]):
            with self.assertRaises(ValueError):
                module.overlay([row], {row['id']: tree}, bad, [row], validate=False)

    def test_neural_inputs_only_see_paid_prefix_and_no_reference(self):
        row, tree, records = self.fixture()
        trees, judges = module.overlay([row], {row['id']: tree}, records, [row], validate=False)
        seen = []
        class Observer:
            def entry(self, safe, path, paid, judge=None):
                if safe['answer']:
                    raise AssertionError('reference reached encoder')
                expected = {'', path}
                if '/' in path:
                    expected.add(path.rsplit('/', 1)[0])
                if set(paid) != expected:
                    raise AssertionError('unpaid branch reached encoder')
                seen.append(path)
                return {'left': safe['id'], 'right': paid[path]['text'], 'numeric': [0.]}
        metadata, inputs = module.entries([row], trees, judges, {row['id']}, Observer())
        self.assertEqual(len(metadata), len(seen))
        self.assertTrue(all(r['fold'] in (0, 1, 2) for r in metadata))
        new_metadata, new_inputs = module.entries([row], trees, judges, set(), Observer())
        self.assertTrue(all(r['fold'] == 'full' for r in new_metadata))
        changed = {**row, 'answer': 'changed-secret-reference'}
        _, changed_inputs = module.entries([changed], trees, judges, set(), Observer())
        self.assertEqual(new_inputs, changed_inputs)


if __name__ == '__main__':
    unittest.main()
