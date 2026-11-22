from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import active_textbook_review_runner_v2 as run
from test_active_textbook_review import Retriever, ActiveRemote
from test_strong_late_review import fixture


class PrefilledRemote(ActiveRemote):
    def __call__(self, endpoint, payload):
        result = super().__call__(endpoint, payload)
        if payload['options']['num_predict'] == 1024:
            result['response'] = 'harmonic oscillator selection rules</search> Paid derivation'
        return result


class ActiveV2Tests(unittest.TestCase):
    def test_placeholders_do_not_consume_valid_query_budget(self):
        q = run.spec.retrieval.queries('<search>query</search><search>...</search><search>selection rule</search><search>harmonic oscillator</search>')
        self.assertEqual(q, ['selection rule', 'harmonic oscillator'])
        self.assertEqual(run.spec.retrieval.queries('<search>query</search><search>...</search>'), [])

    def test_missing_query_falls_back_only_to_public_question_stem(self):
        tool = run.spec.retrieval
        with patch.object(tool, 'retriever', return_value=Retriever()):
            results = tool.evaluate_text('<search>...</search>', 'physics', 'Public stem\n\nOptions:\nA. distractor')
            self.assertEqual(results[0]['query'], 'Public stem'); self.assertTrue(results[0]['public_question_fallback'])
            results = tool.evaluate_text('<search>quantum selection rule</search>', 'physics', 'Public stem')
            self.assertFalse(results[0]['public_question_fallback'])
            results = tool.evaluate_text('incomplete tag', 'physics', 'x'*1000)
            self.assertEqual(len(results[0]['query']), 800)
        with self.assertRaises(ValueError): tool.evaluate_text('text', 'physics', '')

    def test_prefilled_query_roundtrip_and_original_compact_payload(self):
        rows, trees, *_ = fixture(); row = rows[0]; remote = PrefilledRemote()
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with run.configured_runner(), patch.object(run.spec.retrieval, 'retriever', return_value=Retriever()), \
                    (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw, (folder/'attempts').open('a') as attempts:
                journal = run.original.Journal(remote, raw, attempts)
                run.original.Collector(trees, {}, bank, journal).one(row)
                records = run.protocol.strict_records(folder/'bank'); physical = run.protocol.strict_records(folder/'raw')
                self.assertTrue(remote.calls[0]['prompt'].endswith('<think>\n<search>'))
                teacher = records[0]['response']; query = teacher['retrieval_results'][0]['query']
                self.assertEqual(query, 'harmonic oscillator selection rules')
                self.assertFalse(teacher['public_question_fallback'])
                self.assertIn('PUBLIC FACT', remote.calls[1]['prompt'])
                self.assertEqual(remote.calls[2]['prompt'], run.spec.previous.spec.short_repair(row, trees[row['id']][''], teacher))
                self.assertEqual([p['options']['num_predict'] for p in remote.calls], [1024, 512, 128, 128])
                run.original.attempts_checked(run.protocol.strict_records(folder/'attempts'), physical)


if __name__ == '__main__': unittest.main()
