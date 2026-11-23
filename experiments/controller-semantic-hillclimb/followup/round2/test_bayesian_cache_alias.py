import unittest
import numpy as np
import development_bayesian_hybrid_v2 as adapter


class ExactAliasTests(unittest.TestCase):
    def test_same_input_reuses_but_different_question_or_hash_cannot(self):
        value = (np.zeros(4),np.zeros(8),np.zeros(8),17)
        entries = {('solver','qid','', 'a'*64):value}
        requested = [('solver','qid','retry','a'*64),('solver','other','retry','a'*64),('solver','qid','repair','b'*64)]
        result,receipt = adapter.alias_entries(entries,requested)
        self.assertIs(result[requested[0]],value)
        self.assertEqual(receipt['aliases'],[requested[0]])
        self.assertEqual(receipt['unavailable_inputs'],requested[1:])

    def test_conflicting_output_never_selected(self):
        def value(x):return (np.full(4,x),np.zeros(8),np.zeros(8),17)
        entries = {('solver','qid','','a'*64):value(0),('solver','qid','guided','a'*64):value(1)}
        request = ('solver','qid','retry','a'*64)
        result,receipt = adapter.alias_entries(entries,[request])
        self.assertNotIn(request,result)
        self.assertEqual(receipt['unavailable_inputs'],[request])


if __name__ == '__main__':unittest.main()
