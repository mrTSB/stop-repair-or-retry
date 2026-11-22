from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import strong_late_review_pilot as run
from test_drop_review_repair import fixture as original_fixture
from test_science_mentor_expansion import Remote


def fixture():
    rows,trees,reviews=original_fixture()
    ordinary={};ordinary_reviews={}
    for row in rows:
        qid=row['id'];tree=trees[qid]
        for p in ('complete','repair/complete','retry/complete'):tree[p]=copy.deepcopy(tree['retry/retry'])
        ordinary[qid]=copy.deepcopy(tree)
        for parent in run.spec.PREFIXES:
            ordinary[qid][parent+'/reviewed']=copy.deepcopy(tree['retry/retry'])
            ordinary_reviews[qid,parent]={'input_tokens':20,'output_tokens':8,'answer':'B'}
    return rows,trees,reviews,ordinary,ordinary_reviews


class StrongLateTests(unittest.TestCase):
    def test_exact_current_prefix_template_seed_caps_and_no_root_review(self):
        rows,trees,*_=fixture();row=rows[0];tree=trees[row['id']]
        payloads=[]
        for parent in run.spec.PREFIXES:
            with self.assertRaises(run.spec.MissingPhysical) as error:run.spec.review(row,tree,parent,{})
            payload=error.exception.payload;payloads.append(payload)
            self.assertEqual(payload['prompt'],run.spec.reasoning_prompt(row,tree[parent]))
            self.assertNotIn('PRIVATE_MENTOR_FEEDBACK',payload['prompt'])
            self.assertEqual(payload['options']['num_predict'],1024);self.assertEqual(payload['options']['num_ctx'],8192)
            tagged=run.spec.seeded_row(row,parent)
            expected=int(hashlib.sha256(f'{run.protocol.old.SEED}:{tagged["id"]}:bounded-thought-v1'.encode()).hexdigest()[:8],16)%2**31
            self.assertEqual(payload['options']['seed'],expected)
            changed=copy.deepcopy(tree);changed['guided']['text']='UNPAID ROOT TEACHER';changed['guided/reuse']['text']='UNPAID FUTURE'
            with self.assertRaises(run.spec.MissingPhysical) as again:run.spec.review(row,changed,parent,{})
            self.assertEqual(payload,again.exception.payload)
            with self.assertRaises(ValueError):run.spec.review({**row,'answer':'PRIVATE_REFERENCE'},tree,parent,{})
        self.assertNotEqual(payloads[0]['options']['seed'],payloads[1]['options']['seed'])
        teacher={'answer':'A','verdict':{'check':'paid review'},'text':'paid review'}
        payload=run.spec.solver_payload(row,tree,'repair',teacher)
        self.assertEqual(payload['prompt'],run.spec.short_repair(row,tree['repair'],teacher))
        self.assertEqual(payload['options'],{'temperature':.65,'seed':run.old.fixed.logical_seed(row['id'],'repair/repair'),
                                           'num_predict':128,'num_ctx':4096})

    def test_complete128_grid_both_phases_singleflight_and_bad_answer_retention(self):
        rows,trees,*_=fixture();remote=Remote(two_phase=True)
        with TemporaryDirectory() as directory:
            folder=Path(directory)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw:
                journal=run.old.RequestJournal(remote,raw,slots=2)
                collector=run.Collector(trees,{},bank,journal)
                pairs=[(r,p) for r in rows for p in run.spec.PREFIXES]
                with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(collector.one,[*pairs,pairs[0]]))
            records=run.protocol.strict_records(folder/'bank');raw=run.protocol.strict_records(folder/'raw')
            known,coverage=run.spec.restore(rows,trees,records,raw,True)
            self.assertEqual(coverage,{'questions':64,'logical_reviews':128,'compact_repairs':128,
                                      'physical_review_phases':256,'physical_calls':384})
            self.assertEqual(len(remote.calls),384);self.assertLessEqual(remote.peak,2)
            self.assertTrue(all(v['response']['answer']=='' for v in known.values() if v['kind']=='solver'))
            with self.assertRaises(ValueError):run.spec.restore(rows,trees,records[:-1],raw,True)
            with self.assertRaises(ValueError):run.spec.restore(rows,trees,records,raw[:-1],True)
            bad=copy.deepcopy(records);bad[0]['paid_prefix']['current_sha256']='tamper'
            with self.assertRaises(ValueError):run.spec.restore(rows,trees,bad,raw)

    def test_one_phase_review_is_charged_once_and_resume_has_no_regeneration(self):
        rows,trees,*_=fixture();remote=Remote();row=rows[0]
        with TemporaryDirectory() as directory:
            folder=Path(directory)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw:
                collector=run.Collector(trees,{},bank,run.old.RequestJournal(remote,raw))
                with patch.object(run.old,'durable_line',side_effect=RuntimeError('logical append failed')):
                    with self.assertRaises(RuntimeError):collector.one((row,'repair'))
            saved=run.protocol.strict_records(folder/'raw');self.assertEqual(len(saved),1)
            def request(endpoint,payload):
                self.assertEqual(payload['model'],run.protocol.old.MODEL)
                return remote(endpoint,payload)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw:
                run.Collector(trees,{},bank,run.old.RequestJournal(request,raw,saved)).one((row,'repair'))
            known,coverage=run.spec.restore(rows,trees,run.protocol.strict_records(folder/'bank'),run.protocol.strict_records(folder/'raw'))
            self.assertEqual(coverage['physical_review_phases'],1);self.assertEqual(coverage['physical_calls'],2)
            self.assertEqual(len(remote.calls),2)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw:
                journal=run.old.RequestJournal(lambda *a:self.fail('Repeated a completed generation'),raw,run.protocol.strict_records(folder/'raw'))
                run.Collector(trees,known,bank,journal).one((row,'repair'))

    def test_partial_two_phase_review_resumes_only_missing_phase(self):
        rows,trees,*_=fixture();row=rows[0];remote=Remote(two_phase=True)
        with TemporaryDirectory() as directory:
            folder=Path(directory)
            def interrupted(endpoint,payload):
                if payload['options']['num_predict']==512:raise RuntimeError('transport stopped')
                return remote(endpoint,payload)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw:
                with self.assertRaises(RuntimeError):run.Collector(trees,{},bank,run.old.RequestJournal(interrupted,raw)).one((row,'retry'))
            saved=run.protocol.strict_records(folder/'raw');self.assertEqual(len(saved),1)
            def resume(endpoint,payload):
                self.assertNotEqual(payload['options']['num_predict'],1024)
                return remote(endpoint,payload)
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw:
                run.Collector(trees,{},bank,run.old.RequestJournal(resume,raw,saved)).one((row,'retry'))
            _,coverage=run.spec.restore(rows,trees,run.protocol.strict_records(folder/'bank'),run.protocol.strict_records(folder/'raw'))
            self.assertEqual(coverage['physical_review_phases'],2);self.assertEqual(coverage['physical_calls'],3)

    def test_bad_completed_phase_fails_closed_before_logical_and_grades(self):
        rows,trees,*_=fixture();remote=Remote(two_phase=True)
        with TemporaryDirectory() as directory:
            folder=Path(directory)
            def bad(endpoint,payload):
                value=remote(endpoint,payload);value['eval_count']=1025;return value
            with (folder/'bank').open('a') as bank,(folder/'raw').open('a') as raw:
                with self.assertRaises(ValueError):run.Collector(trees,{},bank,run.old.RequestJournal(bad,raw)).one((rows[0],'repair'))
            raw=run.protocol.strict_records(folder/'raw');self.assertEqual(len(raw),1)
            self.assertEqual(run.protocol.strict_records(folder/'bank'),[])
            with self.assertRaises(ValueError):run.spec.restore(rows,trees,[],raw)
            with patch.object(run.old.fixed,'lines',side_effect=AssertionError('Opened labels before full seal')):
                with self.assertRaises(FileNotFoundError):run.grade(SimpleNamespace(output=folder))

    def test_fixed_novelty_and_all128_full_trajectory_cost_screen(self):
        rows,trees,reviews,ordinary,ordinary_reviews=fixture();known={};references={r['id']:'A' for r in rows}
        for i,row in enumerate(rows):
            qid=row['id'];tree=trees[qid]
            for node in tree.values():node['answer']='A' if i>=8 else 'B'
            for parent in run.spec.PREFIXES:
                teacher={**reviews[qid,''],'input_tokens':100,'output_tokens':30,'physical_api_calls':2}
                answer={**tree['retry/retry'],'answer':'A' if i<6 or i>=8 else 'B'}
                known[qid,'judge',parent]={'response':teacher}
                known[qid,'solver',parent+'/late-guided']={'response':answer}
        reports,_,screen=run.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,known,references)
        all_rows=reports['all_science'];self.assertEqual(all_rows['novel_questions'],6)
        self.assertEqual(all_rows['known_plain_Complete_rootstrong_oracle'],56)
        self.assertTrue(screen['passed']);self.assertFalse(screen['controller_promoted'])
        self.assertEqual(all_rows['methods']['Strong late after repair']['mean_solver_calls'],3)
        self.assertEqual(all_rows['methods']['Strong late after repair']['mean_logical_review_calls'],1)
        self.assertEqual(all_rows['methods']['Strong late after repair']['mean_physical_review_calls'],2)
        self.assertEqual(all_rows['methods']['Repair then Complete if invalid']['mean_solver_calls'],2)
        self.assertEqual(all_rows['methods']['Stop if valid else Complete']['mean_solver_calls'],1)
        self.assertEqual(all_rows['paired']['Strong late after retry']['Strong retained second repair'],{'wins':6,'losses':0})
        for key,value in known.items():
            if key[1]=='solver':value['response']['input_tokens']=10000
        self.assertFalse(run.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,known,references)[2]['passed'])

    def test_broader_ordinary_union_and_teacher_transfer_audit_do_not_change_screen(self):
        rows,trees,reviews,ordinary,ordinary_reviews=fixture();known={};references={r['id']:'A' for r in rows}
        for i,row in enumerate(rows):
            qid=row['id']
            for node in trees[qid].values():node['answer']='A' if i>=8 else 'B'
            if i<3:ordinary[qid]['repair/reviewed']['answer']='A'
            for parent in run.spec.PREFIXES:
                known[qid,'judge',parent]={'response':{**reviews[qid,''],'answer':'A'}}
                known[qid,'solver',parent+'/late-guided']={'response':{**trees[qid]['retry/retry'],
                    'answer':'A' if i<6 or i>=8 else 'B'}}
        reports,_,screen=run.aggregate(rows,trees,reviews,ordinary,ordinary_reviews,known,references)
        report=reports['all_science']
        self.assertEqual(report['novel_questions'],6)
        self.assertEqual(report['all_collected_ordinary_late_Complete_rootstrong_oracle'],59)
        self.assertEqual(report['novel_questions_beyond_all_collected_paths'],3)
        self.assertTrue(screen['passed'])
        self.assertEqual(report['methods']['Strong late after repair']['review_correct'],64)
        self.assertEqual(report['methods']['Strong late after repair']['correct_review_wrong_compact'],2)


if __name__=='__main__':unittest.main()
