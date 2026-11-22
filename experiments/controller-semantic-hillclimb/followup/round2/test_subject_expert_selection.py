from itertools import product
import random
import unittest

import subject_expert_selection as select


def candidate(key,values):
    return {'key':str(key),'subjects':{s:{'n':select.COUNTS[s],'correct':correct,'mean_compute':cost}
                                      for s,(correct,cost) in zip(select.SUBJECTS,values)}}


class SelectionTests(unittest.TestCase):
    def test_pareto_pruning_equals_exhaustive_cost_constrained_selection(self):
        rng = random.Random(238)
        candidates = [candidate(i,[(rng.randrange(30,select.COUNTS[s]+1),rng.randrange(100,1000)) for s in select.SUBJECTS]) for i in range(9)]
        result,_ = select.best_C(candidates,700)
        brute = []
        for choices in product(candidates,repeat=3):
            chosen = dict(zip(select.SUBJECTS,choices))
            value = select.combine(chosen)
            if value['mean_compute'] < 700:
                brute.append(((-value['correct'],value['mean_compute'],tuple(c['key'] for c in choices)),value))
        self.assertEqual(result[1],min(brute,key=lambda v:v[0])[1])

    def test_B_gets_same_subject_specialization_without_nerfing(self):
        candidates = [candidate('a',[(120,200),(20,600),(20,600)]),candidate('b',[(110,100),(30,800),(28,700)])]
        choices,result = select.best_B(candidates)
        self.assertEqual(result['correct'],178)
        self.assertEqual(choices['math']['key'],'a')
        self.assertEqual(choices['chemistry']['key'],'b')
        self.assertGreaterEqual(result['correct'],max(select.combine({s:c for s in select.SUBJECTS})['correct'] for c in candidates))

    def test_strict_compute_and_duplicate_records(self):
        candidates = [candidate('only',[(120,500),(30,500),(30,500)])]
        self.assertIsNone(select.best_C(candidates,500)[0])
        self.assertIsNotNone(select.best_C(candidates,501)[0])
        with self.assertRaises(ValueError): select.metrics([{'id':'same','correct':1,'compute':1.}]*2)


if __name__ == '__main__': unittest.main()
