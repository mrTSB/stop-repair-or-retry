import unittest
from unittest.mock import patch

import numpy as np

import dense_textbook_retrieval as retrieval


class DenseRetrievalTests(unittest.TestCase):
    def test_subject_filter_source_dedup_and_exact_score_ties(self):
        passages = [{'id': key,'subject':subject,'source':source,'title':key,'text':' '.join(['fact']*100)}
                    for key,subject,source in [('a','chemistry','source1'),('b','chemistry','source1'),
                                              ('c','chemistry','source2'),('d','physics','source3')]]
        found = retrieval.rank(passages,[.5,.5,.5,1.],[.1,.1,.1,1.],'chemistry')
        self.assertEqual([r['title'] for r in found],['a','c'])
        self.assertTrue(all(len(r['text'].split())==85 for r in found))
        with self.assertRaises(ValueError): retrieval.rank(passages,[np.nan]*4,[0]*4,'chemistry')

    def test_public_fallback_charges_every_used_query_even_if_excerpts_duplicate(self):
        class Fake:
            def retrieve(self, query, subject):
                return [{'title':'fact','text':'scientific exposition','source':'source1','retrieval_score':.02}], {'charged_query_forwards':1,'compute':2.}
        with patch.object(retrieval,'retriever',return_value=Fake()):
            results, charges = retrieval.evaluate_text('<search>...</search>','physics','PUBLIC STEM\n\nOptions:\nA. option')
            self.assertEqual(results[0]['query'],'PUBLIC STEM')
            self.assertTrue(results[0]['public_question_fallback'])
            results, charges = retrieval.evaluate_text('<search>angular momentum</search><search>spin selection</search>','physics','PUBLIC STEM')
            self.assertEqual(len(charges),2)
            self.assertEqual(sum(r['compute'] for r in charges),4.)
            self.assertEqual(sum(len(r['passages']) for r in results),1)


if __name__ == '__main__': unittest.main()
