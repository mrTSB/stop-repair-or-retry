import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import test_difficulty_pomdp as fixtures
from difficulty_pomdp import evaluate,MissingObservation
from late_pairs import llm_policy,merged_records


class LateReviewedRepair(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.RolesAndHiddenEvidence('test_a_cannot_read_an_llm_observation')
        self.fixture.setUp()
        for path in ('repair','retry','guided'):
            self.fixture.tree[path+'/reviewed']={**self.fixture.tree[path],
                                               'prompt_sha256':path+'/reviewed'}

    def test_llm_controller_uses_paid_review_at_the_second_checkpoint(self):
        f=self.fixture
        result=llm_policy(f.row,f.tree,f.judges,{'judge_stop':'action','reuse_review':False})
        self.assertEqual(result['path'],'guided/reviewed')
        self.assertEqual(result['solver_calls'],3)
        self.assertEqual(result['judge_calls'],2)
        self.assertAlmostEqual(result['compute'],(360*4+500*3)/3.09)

    def test_hybrid_can_take_the_same_second_checkpoint_action(self):
        f=self.fixture
        result=evaluate(f.row,f.tree,f.judges,f.planner(),{'penalty':.1,'late_feedback':True},'always')
        self.assertEqual(result['path'],'guided/reviewed')
        self.assertEqual(result['solver_calls'],3)
        self.assertEqual(result['judge_calls'],2)

    def test_missing_late_branch_is_requested_instead_of_silently_using_plain_repair(self):
        f=self.fixture;del f.tree['guided/reviewed']
        with self.assertRaises(MissingObservation) as caught:
            llm_policy(f.row,f.tree,f.judges,{'judge_stop':'action','reuse_review':False})
        self.assertEqual((caught.exception.kind,caught.exception.path),('solver','guided/reviewed'))

    def test_old_review_does_not_substitute_for_missing_new_checkpoint_review(self):
        f=self.fixture
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder);(source/'test-late-observations.jsonl').write_text('')
            with patch('late_pairs.records',return_value=([f.row],{'unit':f.tree},f.judges)):
                _,_,late=merged_records(source,'test',False)
        self.assertEqual(set(late),{('unit','')})
        with self.assertRaises(MissingObservation) as caught:
            llm_policy(f.row,f.tree,late,{'judge_stop':'action','reuse_review':False})
        self.assertEqual((caught.exception.kind,caught.exception.path),('judge','guided'))


if __name__=='__main__':unittest.main()
