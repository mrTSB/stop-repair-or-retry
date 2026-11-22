from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import grounded_root_review_pilot as run
from test_strong_late_review import fixture
from test_science_mentor_expansion import Remote


class GroundedRemote(Remote):
    def __call__(self, endpoint, payload):
        value = super().__call__(endpoint, payload)
        if self.two_phase and payload['model'] == run.protocol.old.JUDGE:
            value['response'] = 'Checking the public options.' if payload['options']['num_predict'] == 1024 else 'A", "p_correct":0.3,"action":"repair"}'
        return value


class GroundedReviewTests(unittest.TestCase):
    def setUp(self):
        self.context = patch.object(run.spec, 'context', return_value='Fixed question-only textbook evidence.'); self.context.start()
        self.addCleanup(self.context.stop)

    def test_only_question_grounding_changes_with_original_seed_and_cap(self):
        rows, trees, *_ = fixture(); row = rows[0]; tree = trees[row['id']]
        with self.assertRaises(run.spec.MissingPhysical) as error: run.spec.review(row, tree, {})
        payload = error.exception.payload
        self.assertEqual(payload['prompt'], run.previous.spec.reasoning_prompt(run.spec.grounded_row(row), tree['']))
        seed = int(hashlib.sha256(f'{run.protocol.old.SEED}:{row["id"]}:bounded-thought-v1'.encode()).hexdigest()[:8], 16) % 2**31
        self.assertEqual(payload['options'], {'temperature': 0, 'seed': seed, 'num_predict': 1024, 'num_ctx': 8192})
        changed = copy.deepcopy(tree); changed['guided']['text'] = 'UNPAID_FUTURE'
        with self.assertRaises(run.spec.MissingPhysical) as second: run.spec.review(row, changed, {})
        self.assertEqual(second.exception.payload, payload)
        with self.assertRaises(ValueError): run.spec.review({**row, 'answer': 'PRIVATE_REFERENCE'}, tree, {})

    def test_complete_cohort_bills_one_or_two_phases_and_never_regenerates(self):
        rows, trees, *_ = fixture()
        for two_phase, expected in ((False, 192), (True, 256)):
            with TemporaryDirectory() as temp:
                folder = Path(temp); remote = GroundedRemote(two_phase)
                with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw, (folder/'attempts').open('a') as attempt_stream:
                    collector = run.Collector(trees, {}, bank, run.Journal(remote, raw, attempt_stream))
                    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(collector.one, rows))
                records, raw = [run.protocol.strict_records(folder/n) for n in ('bank', 'raw')]
                attempts = run.protocol.strict_records(folder/'attempts')
                self.assertEqual(len(run.attempts_checked(attempts, raw)), expected)
                known, coverage = run.spec.restore(rows, trees, records, raw, True)
                self.assertEqual(len(remote.calls), expected); self.assertEqual(coverage['physical_calls'], expected)
                self.assertLessEqual(remote.peak, 2)
                self.assertTrue(all(not r['response']['answer'] for k, r in known.items() if k[1] == 'solver'))
                with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as stream, (folder/'attempts').open('a') as attempt_stream:
                    journal = run.Journal(lambda *a: self.fail('Repeated completed response'), stream, attempt_stream, raw, attempts)
                    run.Collector(trees, known, bank, journal).one(rows[0])
                with self.assertRaises(ValueError): run.spec.restore(rows, trees, records[:-1], raw, True)

    def test_all_prior_control_answers_reduce_new_path_novelty(self):
        rows, trees, reviews, ordinary, ordinary_reviews = fixture(); known = {}; late = {}; labels = {r['id']: 'A' for r in rows}
        for i, row in enumerate(rows):
            qid = row['id']
            for tree in (trees[qid], ordinary[qid]):
                for node in tree.values(): node['answer'] = 'A' if i >= 16 else 'B'
            for parent in ('repair', 'retry'):
                late[qid, 'judge', parent] = {'response': {**reviews[qid, ''], 'answer': 'B'}}
                late[qid, 'solver', parent+'/late-guided'] = {'response': {**trees[qid]['retry/retry'], 'answer': 'A' if i >= 16 else 'B'}}
            teacher = {**reviews[qid, ''], 'answer': 'A', 'physical_api_calls': 2}
            known[qid, 'judge', ''] = {'response': teacher}
            for path in run.spec.PATHS:
                known[qid, 'solver', path] = {'response': {**trees[qid]['retry/retry'], 'answer': 'A'}}
        extras = {r['id']: [] for r in rows}
        before = run.aggregate(rows, trees, reviews, ordinary, ordinary_reviews, late, known, labels, extras)[0]['all_science']
        extras[rows[0]['id']] = ['A']
        after = run.aggregate(rows, trees, reviews, ordinary, ordinary_reviews, late, known, labels, extras)[0]['all_science']
        self.assertEqual(after['novel_questions'], before['novel_questions']-1)

    def test_intent_is_durable_and_unresolved_call_blocks_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fail(endpoint, payload):
                self.assertEqual(len(run.protocol.strict_records(root/'attempts')), 1)
                raise RuntimeError('synthetic transport failure')
            with (root/'raw').open('a') as raw, (root/'attempts').open('a') as attempts:
                journal = run.Journal(fail, raw, attempts)
                with self.assertRaises(RuntimeError): journal('/api/generate', {'model': run.protocol.old.MODEL, 'prompt': 'synthetic'})
            saved = run.protocol.strict_records(root/'attempts')
            self.assertEqual(len(saved), 1)
            with self.assertRaises(ValueError): run.attempts_checked(saved, [])


if __name__ == '__main__': unittest.main()
