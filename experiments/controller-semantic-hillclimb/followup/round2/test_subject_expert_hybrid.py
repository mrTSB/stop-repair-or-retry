import unittest
from unittest.mock import patch

import subject_expert_hybrid as policy


class SubjectRuntimeTests(unittest.TestCase):
    def test_only_observed_subject_selects_expert_and_preserves_charge(self):
        experts = {s:{'model':s,'policy_module':'stronger_base_hybrid','settings':{'penalty':.02},
                      'original_candidate_index':i} for i,s in enumerate(policy.SUBJECTS)}
        model = policy.Planner({s:object() for s in policy.SUBJECTS},experts,{})
        item = {'path':'guided','compute':1234.,'solver_calls':2,'judge_calls':1}
        with patch.object(policy.runtime,'dispatch',return_value=item) as dispatch:
            first = policy.evaluate({'subject':'physics','answer':''},{},{},model,{},'C')
            second = policy.evaluate({'subject':'physics','answer':'PRIVATE_LABEL'},{'UNPAID':{}},{},model,{},'C')
            self.assertEqual(first,second)
            self.assertEqual(first['compute'],1234.)
            self.assertEqual(first['subject_expert_model'],'physics')
            self.assertEqual(dispatch.call_count,2)
            self.assertIs(dispatch.call_args.args[0],model.models['physics'])
        with self.assertRaises(ValueError): policy.evaluate({'subject':'physics'},{},{},model,{},'A')

    def test_matched_B_observed_subject_rule_has_no_POMDP(self):
        settings = {'by_subject':{s:{'root_stop':'agreement','second_stop':'never','second_action':'reuse'} for s in policy.SUBJECTS}}
        with patch.object(policy.llm,'policy',return_value={'compute':100.}) as llm:
            policy.evaluate_B({'subject':'chemistry'},{},{},settings)
            self.assertEqual(llm.call_args.args[-1],settings['by_subject']['chemistry'])
        with self.assertRaises(ValueError): policy.evaluate_B({'subject':'chemistry'},{},{},{'by_subject':{}})


if __name__ == '__main__': unittest.main()
