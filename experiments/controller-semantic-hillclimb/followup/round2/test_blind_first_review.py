from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import blind_first_review_pilot as run
from test_strong_late_review import fixture
from test_science_mentor_expansion import Remote


class BlindRemote(Remote):
    def __init__(self):super().__init__(two_phase=True)
    def __call__(self,endpoint,payload):
        result=super().__call__(endpoint,payload)
        if payload['options']['num_predict']==1024:
            # A complete answer in phase1 must never suppress paid phase2.
            result['response']='{"answer":"A","p_correct":0.8,"action":"stop"}'
            result['thinking']='PAID_THINKING_SEGMENT'
        return result


class BlindFirstTests(unittest.TestCase):
    def collect(self,folder,rows,trees,remote=None):
        remote=BlindRemote() if remote is None else remote
        with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw:
            collector=run.Collector(trees,{},bank,run.old.RequestJournal(remote,raw,slots=2))
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(collector.one,rows))
        return remote,run.protocol.strict_records(folder/'bank'),run.protocol.strict_records(folder/'raw')

    def test_phase_visibility_mandatory_two_phases_and_original_paired_seeds(self):
        rows,trees,*_=fixture();row=rows[0];tree=trees[row['id']]
        tree['']['text']='ORIGINAL_CURRENT_ONLY Final: B';tree['repair']['text']='UNPAID_BRANCH'
        with TemporaryDirectory() as tmp:
            remote,records,raw=self.collect(Path(tmp),[row],trees)
        self.assertEqual(len(remote.calls),4);first,second,guided,reuse=remote.calls
        self.assertNotIn('ORIGINAL_CURRENT_ONLY',first['prompt']);self.assertNotIn('PRIVATE_MENTOR_FEEDBACK',first['prompt'])
        self.assertIn('ORIGINAL_CURRENT_ONLY',second['prompt']);self.assertIn('PAID_THINKING_SEGMENT',second['prompt'])
        self.assertIn(raw[0]['response']['response'],second['prompt'])
        self.assertFalse(any('UNPAID_BRANCH' in r['prompt'] for r in remote.calls))
        for payload,name,cap in zip((first,second),run.spec.PHASES,(1024,512)):
            seed=int(hashlib.sha256(f'{run.protocol.old.SEED}:{row["id"]}:{name}'.encode()).hexdigest()[:8],16)%2**31
            self.assertEqual(payload['options'],{'temperature':0,'seed':seed,'num_predict':cap,'num_ctx':8192})
        for payload,path in zip((guided,reuse),('repair','repair/repair')):
            self.assertEqual(payload['options'],{'temperature':.65,'seed':run.protocol.seed(row['id'],path),
                'num_predict':128,'num_ctx':4096})
        teacher=records[0]['response'];self.assertEqual(teacher['physical_api_calls'],2)
        self.assertEqual(teacher['input_tokens'],40);self.assertEqual(teacher['output_tokens'],16)
        self.assertIn('PAID_THINKING_SEGMENT',teacher['verdict']['solution'])
        with self.assertRaises(ValueError):run.spec.reasoning_prompt({**row,'answer':'PRIVATE_REFERENCE'})

    def test_complete64_grid_and_repeated_invalid_answers_never_regenerate(self):
        rows,trees,*_=fixture()
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);remote,records,raw=self.collect(folder,rows,trees)
            known,coverage=run.spec.restore(rows,trees,records,raw,True)
            self.assertEqual(len(remote.calls),256);self.assertEqual(len(known),192)
            self.assertEqual(coverage['physical_teacher_phases'],128);self.assertEqual(coverage['second_repairs'],64)
            self.assertTrue(all(not v['response']['answer'] for k,v in known.items() if k[1]=='solver'))
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as stream:
                journal=run.old.RequestJournal(lambda *a:self.fail('Repeated completed response'),stream,raw)
                run.Collector(trees,known,bank,journal).one(rows[0])
            with self.assertRaises(ValueError):run.spec.restore(rows[:-1],trees,records,raw,True)
            with self.assertRaises(ValueError):run.spec.restore(rows,trees,records[:-1],raw,True)
            altered=copy.deepcopy(records);altered[0]['paid_prefix']['root_sha256']='tampered'
            with self.assertRaises(ValueError):run.spec.restore(rows,trees,altered,raw)
            altered=copy.deepcopy(raw);altered[0]['response']['thinking']='tampered'
            with self.assertRaises(ValueError):run.spec.restore(rows,trees,records,altered)

    def test_partial_first_phase_resume_only_missing_calls(self):
        rows,trees,*_=fixture();row=rows[0];remote=BlindRemote()
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            def stopped(endpoint,payload):
                if payload['options']['num_predict']==512:raise RuntimeError('transport failure')
                return remote(endpoint,payload)
            with self.assertRaises(RuntimeError):self.collect(folder,[row],trees,stopped)
            raw=run.protocol.strict_records(folder/'raw');self.assertEqual(len(raw),1)
            self.assertEqual(run.protocol.strict_records(folder/'bank'),[])
            known,_=run.spec.restore(rows,trees,[],raw)
            def resume(endpoint,payload):
                self.assertNotEqual(payload['options']['num_predict'],1024)
                return remote(endpoint,payload)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as stream:
                run.Collector(trees,known,bank,run.old.RequestJournal(resume,stream,raw)).one(row)
            self.assertEqual(len(remote.calls),4)

    def test_completed_anomalous_phase_is_preserved_and_fails_closed(self):
        rows,trees,*_=fixture();remote=BlindRemote()
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            def bad(endpoint,payload):
                value=remote(endpoint,payload);value['eval_count']=1025;return value
            with self.assertRaises(ValueError):self.collect(folder,[rows[0]],trees,bad)
            raw=run.protocol.strict_records(folder/'raw');self.assertEqual(len(raw),1)
            self.assertEqual(run.protocol.strict_records(folder/'bank'),[])
            with self.assertRaises(ValueError):run.spec.restore(rows,trees,[],raw)
            with patch.object(run.old.fixed,'lines',side_effect=AssertionError('Labels opened before seal')):
                with self.assertRaises(FileNotFoundError):run.grade(SimpleNamespace(output=folder))

    def test_raw_survives_logical_failure_and_paid_node_order_is_enforced(self):
        rows,trees,*_=fixture();row=rows[0];remote=BlindRemote()
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with patch.object(run.old,'durable_line',side_effect=RuntimeError('logical append failed')):
                with self.assertRaises(RuntimeError):self.collect(folder,[row],trees,remote)
            raw=run.protocol.strict_records(folder/'raw');self.assertEqual(len(raw),2)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as stream:
                run.Collector(trees,{},bank,run.old.RequestJournal(remote,stream,raw)).one(row)
            records=run.protocol.strict_records(folder/'bank');raw=run.protocol.strict_records(folder/'raw')
            self.assertEqual(len(remote.calls),4)
            with self.assertRaises(ValueError):run.spec.restore(rows,trees,list(reversed(records)),raw)

    def test_fixed_full48_novelty_and_new_trajectory_does_not_pay_old_review(self):
        rows,trees,reviews,ordinary,ordinary_reviews=fixture();known={};late={};labels={r['id']:'A' for r in rows}
        for i,row in enumerate(rows):
            qid=row['id']
            for tree in (trees[qid],ordinary[qid]):
                for node in tree.values():node['answer']='A' if i>=16 else 'B'
            for parent in ('repair','retry'):
                late[qid,'judge',parent]={'response':{**reviews[qid,''],'answer':'B'}}
                late[qid,'solver',parent+'/late-guided']={'response':{**trees[qid]['retry/retry']}}
            known[qid,'judge','']={'response':{**reviews[qid,''],'answer':'A'}}
            for path in run.spec.PATHS:
                known[qid,'solver',path]={'response':{**trees[qid]['retry/retry'],'answer':'A' if i<6 or i>=16 else 'B'}}
        reports,_,screen=run.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,known,labels)
        main=reports['all_science'];self.assertEqual(main['registered_known_union'],48)
        self.assertEqual(main['novel_questions'],6);self.assertEqual(main['combined_hindsight_union'],54)
        self.assertTrue(screen['passed']);self.assertFalse(screen['controller_promoted'])
        first,second=(main['methods'][n] for n in run.NAMES)
        self.assertEqual(first['mean_solver_calls'],2);self.assertEqual(second['mean_solver_calls'],3)
        self.assertEqual(second['mean_physical_review_calls'],2)
        self.assertIn('Repair then Complete if invalid',main['methods'])
        self.assertIn('Strong late after retry',main['methods'])
        self.assertEqual(main['paired'][run.NAMES[1]]['Strong retained second repair'],{'wins':6,'losses':0})
        old_cost=second['mean_compute']
        for teacher in reviews.values():teacher['input_tokens']+=10000
        other=run.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,known,labels)[0]
        self.assertEqual(other['all_science']['methods'][run.NAMES[1]]['mean_compute'],old_cost)
        with self.assertRaises(ValueError):run.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,known,labels,47)
        for key,value in known.items():
            if key[1]=='solver':value['response']['input_tokens']=100000
        self.assertFalse(run.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,late,known,labels)[2]['passed'])

    def test_registration_binds_prior_full_grid_and_no_calls(self):
        rows,trees,reviews,ordinary,ordinary_reviews=fixture()
        prior={'selection':[{'id':r['id'],'subject':r['subject'],'bank':'new'} for r in rows]}
        inputs=rows,trees,reviews,ordinary,ordinary_reviews,{},prior,{'sealed-source':'synthetic-sha'}
        with TemporaryDirectory() as tmp,patch.object(run.spec,'public_inputs',return_value=inputs):
            args=SimpleNamespace(**{k:Path(tmp)/k for k in ('data','mentor','complete','late','output')})
            value=run.registration(args)
        self.assertEqual(value['maximum_total_physical_calls'],256);self.assertEqual(len(value['first_requests']),64)
        self.assertEqual(value['selection'],prior['selection']);self.assertEqual(value['screen']['registered_known_union_questions'],48)
        self.assertFalse(value['old_root_review_is_paid_prefix']);self.assertTrue(value['teacher']['always_two_phases'])
        self.assertEqual(value['input_evidence'],{'sealed-source':'synthetic-sha'})


if __name__=='__main__':unittest.main()
