import unittest
from types import SimpleNamespace
from unittest.mock import patch
import test_completion_mentor as fixtures
import continuous_completion_hybrid as hybrid

node = fixtures.node


class ContinuousHybridTests(unittest.TestCase):
    def run_route(self,query,route=(3,),semantic=False,answer='A'):
        row={'id':'q','subject':'physics','domain':'science','answer':answer}
        tree={p:node('A' if p.endswith('complete') or p.endswith('reuse') else 'B',20+i)
              for i,p in enumerate(['','complete','repair','repair/complete','guided','guided/reuse'])}
        calls=[{'input_tokens':30,'output_tokens':4},{'input_tokens':40,'output_tokens':5}]
        judge={**node('A'), 'input_tokens':70,'output_tokens':9,'calls':calls,'physical_api_calls':2,'parse_valid':True}
        model=fixtures.RuntimeTests().model(query,route,semantic)
        model.science.kernel=SimpleNamespace(signal=lambda p,v:0)
        model.science_settings={'posterior_weight':.5}
        with patch.object(hybrid.assets,'asset_id',return_value='recipe'),patch.object(hybrid.mentor_core,'signal',return_value=0):
            result=hybrid.evaluate(row,tree,{('q',''):judge},model,{'penalty':0.,'threshold':0.},'C')
        return result,model,tree

    def test_unqueried_complete_is_terminal_and_root_forward_charged_once(self):
        result,model,tree=self.run_route(False)
        self.assertEqual(result['path'],'complete');self.assertEqual(result['completion_calls'],1)
        self.assertEqual(result['judge_calls'],0);self.assertEqual(result['judge_physical_calls'],0)
        self.assertEqual(result['solver_calls'],2);self.assertEqual(result['verifier_calls'],1)
        expected=(4*sum(tree[p]['input_tokens']+1 for p in ('','complete'))+.4*10)/3.09
        self.assertAlmostEqual(result['compute'],expected)
        self.assertFalse(model.mentor.classifier.seen)

    def test_plain_repair_then_complete_never_observes_terminal(self):
        result,model,_=self.run_route(False,(1,3))
        self.assertEqual(result['path'],'repair/complete');self.assertEqual(result['solver_calls'],3)
        self.assertEqual(result['verifier_calls'],2)
        self.assertEqual({p for p,_ in model.science.classifier.seen},{'','repair'})

    def test_query_retains_old_paths_and_bills_both_review_phases(self):
        result,model,tree=self.run_route(True,semantic=True)
        self.assertEqual(result['path'],'guided/reuse');self.assertEqual(result['completion_calls'],0)
        self.assertEqual(result['judge_calls'],1);self.assertEqual(result['judge_physical_calls'],2)
        self.assertEqual(result['solver_calls'],3);self.assertEqual(result['verifier_calls'],3)
        self.assertEqual(result['judge_tokens'],79)
        expected=(4*sum(tree[p]['input_tokens']+1 for p in ('','guided','guided/reuse'))+3*79+.4*30)/3.09
        self.assertAlmostEqual(result['compute'],expected)
        self.assertEqual({p for p,_ in model.mentor.classifier.seen},{'','guided'})
        altered,_,_=self.run_route(True,semantic=True,answer='D')
        self.assertEqual(result['decisions'],altered['decisions']);self.assertEqual(result['compute'],altered['compute'])
        self.assertNotEqual(result['correct'],altered['correct'])


if __name__ == '__main__': unittest.main()
