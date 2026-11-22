from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import history_fusion_runner as run
from test_science_mentor_expansion import Remote
from test_strong_late_review import fixture as old_fixture


def fixture():
    result = old_fixture()
    for review in result[2].values():
        inputs,outputs = review['input_tokens'],review['output_tokens']
        review['calls'] = [{'phase':name,'text':'PAID REVIEW REASONING','thinking':'','output_cap':cap,
            'input_tokens':n,'output_tokens':m} for name,cap,n,m in (
                ('bounded-thought-v1',1024,inputs//2,outputs//2),
                ('bounded-answer-v1',512,inputs-inputs//2,outputs-outputs//2))]
        review['physical_api_calls'] = 2
        review.setdefault('verdict',{'check':'PAID REVIEW CHECK'})
    return result


class FusionTests(unittest.TestCase):
    def test_full128_compact_only_receipts_replay_and_paid_prefix_cost(self):
        rows,trees,reviews,*_ = fixture()
        remote = Remote(two_phase=True)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw,(folder/'attempts').open('a') as attempts:
                journal = run.Journal(remote,raw,attempts)
                collector = run.Collector(trees,reviews,{},bank,journal)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(collector.one,[(r,p) for r in rows for p in run.spec.PREFIXES]))
            records = run.protocol.strict_records(folder/'bank')
            physical = run.protocol.strict_records(folder/'raw')
            known,coverage = run.spec.restore(rows,trees,reviews,records,physical,True)
            self.assertEqual(coverage,{'questions':64,'physical_calls':128,'logical_endpoints':128})
            self.assertTrue(all(p['model'] == run.protocol.old.MODEL for p in remote.calls))
            self.assertTrue(all(p['options']['num_predict'] == 128 for p in remote.calls))
            self.assertLessEqual(remote.peak,4)
            run.attempts_checked(run.protocol.strict_records(folder/'attempts'),physical)
            row = rows[0]
            for parent in run.spec.PREFIXES:
                node = known[row['id'],parent]['response']
                trajectory = run.spec.previous.trajectory([trees[row['id']][''],trees[row['id']][parent],node],reviews[row['id'],''])
                self.assertEqual(trajectory['solver_calls'],3)
                self.assertEqual(trajectory['logical_review_calls'],1)
                self.assertEqual(trajectory['review_input_tokens'],reviews[row['id'],'']['input_tokens'])
                self.assertEqual(trajectory['physical_review_calls'],reviews[row['id'],'']['physical_api_calls'])
            changed = copy.deepcopy(records)
            changed[0]['paid_prefix']['root_sha256'] = 'UNPAID_ROOT'
            with self.assertRaises(ValueError): run.spec.restore(rows,trees,reviews,changed,physical,True)
            with self.assertRaises(ValueError): run.spec.restore(rows,trees,reviews,records[:-1],physical,True)

    def test_fixed_excerpt_and_no_reference_or_unpaid_future_context(self):
        row,trees,reviews = fixture()[0][0],fixture()[1],fixture()[2]
        tree,review = trees[row['id']],copy.deepcopy(reviews[row['id'],''])
        review['calls'][0]['text'] = ' '.join(f'word{i}' for i in range(200))
        prompt = run.spec.prompt(row,tree,review,'guided')
        self.assertIn('word40 word41',prompt)
        self.assertNotIn('word39 ',prompt)
        self.assertIn('word199',prompt)
        changed = copy.deepcopy(tree)
        for path,node in changed.items():
            if path not in ('','guided'): node['text'] = 'UNPAID_FUTURE'
        self.assertEqual(prompt,run.spec.prompt(row,changed,review,'guided'))
        with self.assertRaises(ValueError): run.spec.prompt({**row,'answer':'PRIVATE_REFERENCE'},tree,review,'guided')
        with self.assertRaises(ValueError): run.spec.payload(row,tree,review,'guided/reuse')

    def test_unresolved_physical_attempt_cannot_be_retried(self):
        row,trees,reviews = fixture()[0][0],fixture()[1],fixture()[2]
        payload = run.spec.payload(row,trees[row['id']],reviews[row['id'],''],'guided')
        key = run.old.digest({'endpoint':'/api/generate','payload':payload})
        attempt = {'schema':run.spec.SCHEMA,'endpoint':'/api/generate','payload':payload,
                   'request_sha256':key,'payload_sha256':run.old.digest(payload)}
        with self.assertRaises(ValueError): run.attempts_checked([attempt],[])

    def test_novel_questions_count_once_and_each_endpoint_is_a_legal_trajectory(self):
        rows,trees,reviews,ordinary,ordinary_reviews = fixture()
        private = []
        known = {}
        for i,row in enumerate(rows):
            control = {'correct':False,'review_correct':None,**run.spec.previous.trajectory([trees[row['id']][''],trees[row['id']]['guided'],trees[row['id']]['guided/reuse']],reviews[row['id'],''])}
            private.append({'id':row['id'],'subject':row['subject'],'all_collected_oracle':i < 50,
                            'late_union_correct':False,'methods':{'Strong retained second repair':control}})
            for parent in run.spec.PREFIXES:
                node = {**trees[row['id']][parent],'answer':'A' if 50 <= i < 56 else 'B'}
                known[row['id'],parent] = {'response':node}
        with patch.object(run.spec.previous,'aggregate',return_value=({},private,{})):
            reports,items,screen = run.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,{},known,
                {r['id']:'A' for r in rows},{r['id']:[] for r in rows})
        self.assertEqual(reports['all_science']['novel_questions'],6)
        self.assertEqual(reports['all_science']['new_two_endpoint_union'],6)
        self.assertTrue(screen['checks']['at_least6_novel_questions'])
        for name in run.NAMES: self.assertEqual(reports['all_science']['methods'][name]['mean_solver_calls'],3)


if __name__ == '__main__': unittest.main()
