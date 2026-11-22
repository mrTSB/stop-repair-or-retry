import copy
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

from sklearn.feature_extraction.text import TfidfVectorizer

import training_example_pool as run


class ExamplePoolTests(unittest.TestCase):
    def test_excluded_references_do_not_enter_pool(self):
        row = {'id': 'train', 'subject': 'physics', 'question': 'What is radiation?',
            'options': ['first', 'second', 'third', 'fourth'], 'answer': 'B'}
        excluded = {**row, 'id': 'pilot', 'answer': 'PRIVATE_REFERENCE_NOT_AN_OPTION'}
        a = run.examples([row, excluded], {'pilot'})
        excluded['answer'] = 'CHANGED'
        self.assertEqual(a, run.examples([row, excluded], {'pilot'}))
        self.assertEqual(a[0]['correct_answer_content'], 'second')
        self.assertNotIn('answer', a[0])

    def test_retrieval_uses_public_subject_and_stem_and_excludes_near_duplicate(self):
        rows = [{'id': 'p1', 'subject': 'physics', 'stem': 'Explain photons emitted by a hot surface', 'correct_answer_content': 'Thermal radiation'},
            {'id': 'p2', 'subject': 'physics', 'stem': 'Thermal radiation from a hot star surface depends on temperature', 'correct_answer_content': 'DUPLICATE_ANSWER'},
            {'id': 'c1', 'subject': 'chemistry', 'stem': 'Thermal radiation from a hot star surface depends on temperature', 'correct_answer_content': 'WRONG_SUBJECT'}]
        vectorizer = TfidfVectorizer(); matrix = vectorizer.fit_transform([r['stem'] for r in rows])
        with TemporaryDirectory() as tmp:
            folder = Path(tmp); run.io.jsonl(folder/'pool.jsonl', rows)
            run.io.atomic_bytes(folder/'index.pkl', pickle.dumps((vectorizer, matrix)))
            run.io.write(folder/'manifest.json', {})
            run.io.write(folder/'complete.json', {'schema': run.SCHEMA, 'development_or_test_read': False, 'model_calls': 0,
                'digests': {p.name: run.io.sha(p) for p in folder.iterdir()}})
            retriever = run.Retriever(folder)
            row = {'id': 'pilot', 'subject': 'physics', 'question': rows[1]['stem'], 'answer': ''}
            text, receipt = retriever.context(row)
            self.assertEqual(receipt['example_id'], 'p1')
            self.assertIn('Thermal radiation', text); self.assertNotIn('DUPLICATE_ANSWER', text)
            self.assertEqual((text, receipt), retriever.context({**row, 'unused_solver_response': 'CHANGED'}))
            with self.assertRaises(ValueError): retriever.context({**row, 'answer': 'A'})
            with self.assertRaises(ValueError): retriever.context({**row, 'id': 'p1'})


if __name__ == '__main__': unittest.main()
