import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import dense_grounded_runner as run
from test_science_mentor_expansion import Remote
from test_strong_late_review import fixture


class FakeRetriever:
    def retrieve(self,query,subject):
        bill = {'model':run.spec.retrieval.index.MODEL,'revision':run.spec.retrieval.index.REVISION,
            'input_sha256':hashlib.sha256((run.spec.retrieval.index.INSTRUCTION+query).encode()).hexdigest(),
            'model_parameters':33360000,'actual_forward_token_slots':20,'nonpadding_tokens':20,
            'charged_query_forwards':1,'compute':33360000*20/3.09e9}
        return [{'title':'PUBLIC SCIENCE','text':'PUBLIC_FACT_FROM_FROZEN_CORPUS','source':'source1','retrieval_score':.02}],bill


class DenseGroundedTests(unittest.TestCase):
    def test_full64_replay_and_query_ledger_tamper(self):
        rows,trees,*_ = fixture()
        remote = Remote(two_phase=True)
        with TemporaryDirectory() as tmp,run.configured_runner(),patch.object(run.spec.retrieval,'retriever',return_value=FakeRetriever()):
            folder = Path(tmp)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw,(folder/'attempts').open('a') as attempts:
                journal = run.original.Journal(remote,raw,attempts)
                collector = run.original.Collector(trees,{},bank,journal)
                for row in rows: collector.one(row)
            records = run.protocol.strict_records(folder/'bank')
            physical = run.protocol.strict_records(folder/'raw')
            known,coverage = run.spec.restore(rows,trees,records,physical,complete=True)
            self.assertEqual(coverage['physical_calls'],256)
            self.assertEqual(coverage['physical_teacher_phases'],128)
            self.assertEqual([r['options']['num_predict'] for r in remote.calls[:4]],[1024,512,128,128])
            self.assertIn('PUBLIC_FACT_FROM_FROZEN_CORPUS',remote.calls[0]['prompt'])
            teacher = known[rows[0]['id'],'judge','']['response']
            self.assertEqual(teacher['retrieval_encoder_charge']['charged_query_forwards'],1)
            self.assertEqual(remote.calls[2]['prompt'],run.spec.previous.spec.short_repair(rows[0],trees[rows[0]['id']][''],teacher))
            run.original.attempts_checked(run.protocol.strict_records(folder/'attempts'),physical)
            changed = copy.deepcopy(records)
            changed[0]['response']['retrieval_encoder_charge']['compute'] = 0.
            with self.assertRaises(ValueError): run.spec.restore(rows,trees,changed,physical,complete=True)
            with self.assertRaises(ValueError): run.spec.restore(rows,trees,records[:-1],physical,complete=True)

    def test_query_visibility_and_reference_rejection(self):
        row,trees = fixture()[0][0],fixture()[1]
        seen = []
        class Recording(FakeRetriever):
            def retrieve(self,query,subject):
                seen.append(query)
                return super().retrieve(query,subject)
        with patch.object(run.spec.retrieval,'retriever',return_value=Recording()):
            first = run.spec.evidence(row)
            changed = copy.deepcopy(row)
            changed['question'] = row['question'].partition('\n\nOptions:\n')[0]+'\n\nOptions:\nA. UNPAID_ALTERNATIVE'
            self.assertEqual(first,run.spec.evidence(changed))
            self.assertNotIn('UNPAID_ALTERNATIVE',seen[-1])
            with self.assertRaises(ValueError): run.spec.evidence({**row,'answer':'PRIVATE_REFERENCE'})
        with patch.object(run.spec.retrieval,'retriever',return_value=FakeRetriever()):
            with self.assertRaises(run.spec.MissingPhysical) as first: run.spec.review(row,trees[row['id']],{})
            changed = copy.deepcopy(trees[row['id']])
            for path,node in changed.items():
                if path: node['text'] = 'UNPAID_FUTURE'
            with self.assertRaises(run.spec.MissingPhysical) as second: run.spec.review(row,changed,{})
            self.assertEqual(first.exception.payload,second.exception.payload)

    def test_query_cost_added_to_both_proxy_sensitivities_and_screen(self):
        private = [{'id':str(i),'subject':subject,'methods':{name:{'compute':100.,'total_parameter_compute':1000.}
                    for name in run.NAMES}} for i,subject in enumerate(('chemistry','physics'))]
        reports = {subject:{'novel_questions':6,'methods':{**{name:{} for name in run.NAMES},
                    'Strong retained second repair':{'mean_compute':100.}}}
                   for subject in ('all_science','chemistry','physics')}
        charges = {r['id']:{'compute':30.,'actual_forward_token_slots':20,'charged_query_forwards':1} for r in private}
        screen = run.add_query_costs(reports,private,charges)
        self.assertEqual(reports['all_science']['methods'][run.NAMES[1]]['mean_compute'],130.)
        self.assertEqual(reports['all_science']['methods'][run.NAMES[1]]['mean_total_parameter_compute'],1030.)
        self.assertFalse(screen['checks']['second_trajectory_compute_at_most1p25'])
        with self.assertRaises(ValueError): run.add_query_costs(reports,private,{'0':charges['0']})


if __name__ == '__main__': unittest.main()
