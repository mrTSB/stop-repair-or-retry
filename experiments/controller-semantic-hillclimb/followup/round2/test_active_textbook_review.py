from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import active_textbook_review_runner as run
from test_strong_late_review import fixture
from test_science_mentor_expansion import Remote


class Retriever:
    def retrieve(self, row):
        return [{'source': row['question']+str(i), 'title': 'Reference physics', 'text': 'PUBLIC FACT '+str(i), 'retrieval_score': .5}
            for i in range(2)]


class ActiveRemote(Remote):
    def __init__(self): super().__init__(two_phase=True)
    def __call__(self, endpoint, payload):
        result = super().__call__(endpoint, payload)
        if payload['options']['num_predict'] == 1024:
            result['response'] = '<search>harmonic oscillator</search><search>selection rule</search> Paid derivation'
            result['thinking'] = ''
        return result


class ActiveTextbookTests(unittest.TestCase):
    def test_queries_bounded_and_results_deduplicated(self):
        tool = run.spec.retrieval
        with patch.object(tool, 'retriever', return_value=Retriever()):
            results = tool.evaluate_text('<search>one</search><search>two</search><search>three</search>', 'physics')
            self.assertEqual(len(results), 2); self.assertEqual(sum(len(r['passages']) for r in results), 3)
            self.assertEqual(tool.queries('<search> one </search><search>one</search><search>two</search>'), ['one', 'two'])
            result = tool.evaluate_text('<search>'+'x'*201+'</search>', 'chemistry')
            self.assertFalse(result[0]['ok']); self.assertEqual(result[0]['passages'], [])
            self.assertEqual(tool.evaluate_text('No query', 'physics'), [])
            with self.assertRaises(ValueError): tool.evaluate_text('<search>one</search>', 'math')

    def test_stale_index_cannot_be_rebuilt(self):
        tool = run.spec.retrieval
        with TemporaryDirectory() as tmp, patch.object(tool, 'Retriever') as constructor:
            p = Path(tmp); (p/'passages.jsonl').write_text('changed')
            (p/'tfidf.pkl').write_bytes(pickle.dumps({'sha256': 'wrong'}))
            with patch.object(tool, 'CORPUS', p):
                tool.retriever.cache_clear()
                with self.assertRaises(ValueError): tool.retriever()
            constructor.assert_not_called(); tool.retriever.cache_clear()

    def collect(self, folder, rows, trees):
        remote = ActiveRemote()
        with run.configured_runner(), patch.object(run.spec.retrieval, 'retriever', return_value=Retriever()), \
                (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw, (folder/'attempts').open('a') as attempts:
            journal = run.original.Journal(remote, raw, attempts)
            collector = run.original.Collector(trees, {}, bank, journal)
            with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(collector.one, rows))
        return remote, run.protocol.strict_records(folder/'bank'), run.protocol.strict_records(folder/'raw')

    def test_reference_free_paid_evidence_and_frozen_compact_protocol(self):
        rows, trees, *_ = fixture(); row = rows[0]; tree = trees[row['id']]
        tree['']['text'] = 'PAID_ROOT Final: B'; tree['repair']['text'] = 'UNPAID_BRANCH'
        with TemporaryDirectory() as tmp: remote, records, raw = self.collect(Path(tmp), [row], trees)
        first, second, guided, reuse = remote.calls
        self.assertIn('PAID_ROOT', first['prompt']); self.assertIn('PUBLIC FACT', second['prompt'])
        self.assertNotIn('PUBLIC FACT', first['prompt'])
        self.assertFalse(any('UNPAID_BRANCH' in p['prompt'] for p in remote.calls))
        self.assertEqual(guided['prompt'], run.spec.previous.spec.short_repair(row, tree[''], records[0]['response']))
        self.assertEqual([p['options']['num_predict'] for p in remote.calls], [1024, 512, 128, 128])
        self.assertEqual([p['options']['seed'] for p in (guided, reuse)],
            [run.protocol.seed(row['id'], p) for p in ('repair', 'repair/repair')])
        self.assertEqual(records[0]['response']['retrieval_query_count'], 2)
        self.assertEqual(records[0]['response']['input_tokens'], 40)
        with self.assertRaises(ValueError): run.spec.reasoning_prompt({**row, 'answer': 'SECRET'}, tree[''])

    def test_full_grid_receipts_and_evidence_tamper_detection(self):
        rows, trees, *_ = fixture()
        with TemporaryDirectory() as tmp:
            folder = Path(tmp); remote, records, raw = self.collect(folder, rows, trees)
            with patch.object(run.spec.retrieval, 'retriever', return_value=Retriever()):
                known, coverage = run.spec.restore(rows, trees, records, raw, True)
                self.assertEqual(len(known), 192); self.assertEqual(coverage['physical_calls'], 256)
                altered = copy.deepcopy(records); altered[0]['response']['retrieval_results'][0]['passages'][0]['text'] = 'TAMPER'
                with self.assertRaises(ValueError): run.spec.restore(rows, trees, altered, raw, True)
                with self.assertRaises(ValueError): run.spec.restore(rows, trees, records[:-1], raw, True)
            with run.configured_runner(): run.original.attempts_checked(run.protocol.strict_records(folder/'attempts'), raw)
            self.assertEqual(len(remote.calls), 256)

    def test_scoped_runner_restores_predecessor(self):
        before = run.original.spec, run.original.NAMES, run.original.registration, run.original.require_priority
        with run.configured_runner():
            self.assertIs(run.original.spec, run.spec); self.assertIs(run.original.registration, run.registration)
        self.assertEqual(before, (run.original.spec, run.original.NAMES, run.original.registration, run.original.require_priority))


if __name__ == '__main__': unittest.main()
