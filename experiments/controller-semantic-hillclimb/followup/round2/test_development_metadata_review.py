import hashlib
from contextlib import ExitStack
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import development_metadata_review as run


def cache():
    return {('solver', 'q', '', 'a'*64, 'paid text'): (np.zeros(4), np.zeros(8), np.zeros(8), 12)}


def receipt(cache_bytes, raw=b''):
    return {'schema': run.SCHEMA, 'neural_cache_sha256': hashlib.sha256(cache_bytes).hexdigest(),
            'raw_import_sha256': hashlib.sha256(raw).hexdigest()}


class MetadataDevelopmentTests(unittest.TestCase):
    def test_import_retains_all_wrong_invalid_and_added_receipts(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary); payload = pickle.dumps(cache())
            prefix = b'{"answer":"wrong"}\n'; raw = b'{"paid":"invalid"}\n'
            run.write(output/'registration.json', {'fixture': True})
            run.import_prefix(output, prefix, payload, raw, receipt(payload, raw))
            (output/'dev-observations.jsonl').write_bytes(prefix+b'{"answer":""}\n')
            (output/'raw-generations.jsonl').write_bytes(raw+b'{"paid":"wrong"}\n')
            extended = cache(); extended[('solver','q','repair','b'*64,'paid text')] = (np.ones(4),np.ones(8),np.ones(8),20)
            run.atomic_pickle(output/'private-neural-cache.pkl', extended)
            run.import_prefix(output, prefix, payload, raw, receipt(payload, raw))
            self.assertEqual((output/'dev-observations.jsonl').read_bytes(), prefix+b'{"answer":""}\n')
            self.assertEqual(len(pickle.loads((output/'private-neural-cache.pkl').read_bytes())), 2)

    def test_partial_tails_and_changed_cache_fail_before_any_request(self):
        for target in ('dev-observations.jsonl','raw-generations.jsonl'):
            with TemporaryDirectory() as temporary:
                output = Path(temporary); payload = pickle.dumps(cache())
                run.import_prefix(output,b'{}\n',payload,b'{}\n',receipt(payload,b'{}\n'))
                (output/target).write_bytes(b'{}\n{"partial":')
                with self.assertRaisesRegex(ValueError, 'partial'):
                    run.import_prefix(output,b'{}\n',payload,b'{}\n',receipt(payload,b'{}\n'))
        with TemporaryDirectory() as temporary:
            output=Path(temporary); payload=pickle.dumps(cache())
            run.import_prefix(output,b'{}\n',payload,b'',receipt(payload))
            wrong=cache(); key=next(iter(wrong));wrong[key]=(np.ones(4),np.zeros(8),np.zeros(8),12)
            run.atomic_pickle(output/'private-neural-cache.pkl',wrong)
            with self.assertRaisesRegex(ValueError, 'changed a prior'):
                run.import_prefix(output,b'{}\n',payload,b'',receipt(payload))

    def test_unregistered_output_is_not_overwritten(self):
        with TemporaryDirectory() as temporary:
            output=Path(temporary);(output/'keep').write_text('original')
            payload=pickle.dumps(cache())
            with self.assertRaisesRegex(ValueError,'unregistered'):
                run.import_prefix(output,b'{}\n',payload,b'',receipt(payload))
            self.assertEqual((output/'keep').read_text(),'original')

    def test_source_seal_covers_final_bank_cache_search_and_raw_receipts(self):
        with TemporaryDirectory() as temporary:
            output=Path(temporary);bank=b'{}\n';(output/'dev-observations.jsonl').write_bytes(bank)
            run.write(output/'summary.json',{'test_used':False,'development_questions':256,'bank_sha256':hashlib.sha256(bank).hexdigest()})
            run.write(output/'identity.json',{'test_used':False,'dev_questions_sha256':'panel'})
            run.write(output/'cache-import.json',{'schema':run.SCHEMA})
            for filename in ('registration.json','search.json','private-neural-cache.pkl','raw-generations.jsonl'):
                (output/filename).write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'final'):
                run.completed_bank(output,'panel')
            (output/'transport-errors.jsonl').write_bytes(b'{"attempt":"timeout"}\n')
            run.finalize(output); run.completed_bank(output,'panel')
            self.assertIn('transport-errors.jsonl',run.read(output/'metadata-replay-complete.json')['files'])
            before=(output/'metadata-replay-complete.json').read_bytes();run.finalize(output)
            self.assertEqual(before,(output/'metadata-replay-complete.json').read_bytes())
            (output/'private-neural-cache.pkl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'seal'):
                run.completed_bank(output,'panel')

    def test_registration_precedes_wait_and_source_recipe_is_immutable(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';(data/'splits').mkdir(parents=True)
            (data/'splits/dev-questions.jsonl').write_bytes(b'{"synthetic":"only"}\n')
            args=SimpleNamespace(data=data,models=root/'models',responses=root/'responses',cache=root/'cache',
                cache_models=root/'cache_models',b_reference=root/'B-summary.json',output=root/'output',device='cpu')
            def waiting(folder):
                self.assertTrue((args.output/'registration.json').exists())
                self.assertEqual(run.read(args.output/'registration.json')['policy_module'],'metadata_review_hybrid')
                raise RuntimeError('synthetic wait stop')
            with patch.object(run,'wait_completed',side_effect=waiting):
                with self.assertRaisesRegex(RuntimeError,'synthetic'):
                    run.run(args)
            with patch.object(run,'PENALTIES',[.99]):
                with self.assertRaisesRegex(ValueError,'Changed registered'):
                    run.register(args)

    def test_explicit_runtime_dispatch_and_policy_lock_released_for_acquisition(self):
        lock=threading.Lock();rows=[{'id':'q'}];tree={'q':{}}
        class Acquire:
            calls=0
            def acquire(self,row,missing):
                self.assertions = (not lock.locked(), missing.path == 'guided')
                self.calls+=1; tree['q']['guided']={}
        acquire=Acquire()
        def evaluate(row,paid,judges,model,settings,family):
            self.assertTrue(lock.locked());self.assertEqual(family,'C')
            if 'guided' not in paid: raise run.study.MissingObservation('solver','guided')
            return {'id':row['id'],'metadata_calls':1}
        with patch.object(run.policy,'evaluate',side_effect=evaluate) as function:
            result=run.evaluate(object(),{'penalty':.08},rows,tree,{},acquire,lock)
        self.assertEqual(result,[{'id':'q','metadata_calls':1}]);self.assertEqual(function.call_count,2)
        self.assertEqual(acquire.assertions,(True,True));self.assertEqual(acquire.calls,1)

    def test_exact_retry_reuse_never_calls_solver_or_regenerates_bad_answer(self):
        with TemporaryDirectory() as temporary:
            row={'id':'q','subject':'physics','domain':'science','answer':'SECRET','question':'Public?'}
            node={'answer':'WRONG','text':'Final: WRONG','input_tokens':10,'output_tokens':2,
                  'temperature':.65,'context_length':4096,'seed':run.study.mentor_pairs.seed_for('q','guided/retry')}
            trees={'q':{'repair/retry':node}};path=Path(temporary)/'bank.jsonl'
            with path.open('a') as stream:
                acquire=run.Acquisition([row],trees,{},stream,lambda *_: self.fail('No cloud request'),None)
                missing=run.study.MissingObservation('solver','guided/retry')
                acquire.acquire(row,missing);acquire.acquire(row,missing)
            records=run.strict_records(path)
            self.assertEqual(len(records),1);self.assertIs(trees['q']['guided/retry'],node)
            self.assertTrue(records[0]['exact_control_reused'])
            self.assertEqual(records[0]['response']['answer'],'WRONG')
            self.assertEqual(run.coverage(records,[])['logical_response_nodes'],0)

    def test_new_generation_without_raw_receipt_is_rejected(self):
        record={'id':'q','kind':'solver','path':'guided','metadata_acquired':True,
                'response':{'text':'Final: A','model':run.study.old.MODEL,'prompt_sha256':'a'*64,
                    'seed':1,'temperature':.65,'output_cap':128,'context_length':4096,
                    'input_tokens':10,'output_tokens':3}}
        with self.assertRaisesRegex(ValueError,'raw receipt'):
            run.coverage([record],[])
        self.assertEqual(run.coverage([{**record,'metadata_acquired':False}],[])['logical_response_nodes'],0)

    def test_complete_synthetic_grid_seals_and_resume_does_no_work(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';(data/'splits').mkdir(parents=True)
            (data/'splits/dev-questions.jsonl').write_bytes(b'synthetic panel only\n')
            run.write(data/'dev-plain-complete.json',{'synthetic':True})
            models=root/'models';models.mkdir();source=root/'source';source.mkdir()
            cache_models=root/'cache-models';cache_models.mkdir()
            run.write(cache_models/'complete.json',{'synthetic':True})
            rows=[{'id':f'q{i}','subject':('math','chemistry','physics')[i%3]} for i in range(256)]
            trees={r['id']:{'':{}} for r in rows}
            def stats(items):
                return {'n':len(items),'correct':sum(r['correct'] for r in items),
                    'accuracy':sum(r['correct'] for r in items)/len(items),
                    'mean_compute':float(np.mean([r['compute'] for r in items])),
                    'mean_judge_calls':float(np.mean([r['judge_calls'] for r in items]))}
            specs=[]
            for smoothing in (8,32):
                observer=SimpleNamespace(device='cpu',_cache={},_model=None,_tokenizer=None,
                    _recipe={},_retriever=None,_backgrounds={})
                model=SimpleNamespace(base=SimpleNamespace(classifier=observer),
                    mentor=SimpleNamespace(classifier=SimpleNamespace(**vars(observer))))
                path=models/f'metadata-smooth{smoothing}.pkl';path.write_bytes(pickle.dumps(model))
                specs.append({'key':path.stem,'sha256':run.sha(path),'smoothing':smoothing})
            fitted={'models':specs,'base_A':{'dev':{'accuracy':.5,'mean_compute':10.}}}
            run.write(models/'complete.json',fitted)
            panel=run.sha(data/'splits/dev-questions.jsonl')
            run.write(source/'identity.json',{'test_used':False,'dev_questions_sha256':panel,
                'models_sha256':run.sha(cache_models/'complete.json'),'source_sha256':run.sha(run.study.__file__)})
            (source/'dev-observations.jsonl').write_bytes(b'')
            run.atomic_pickle(source/'private-neural-cache.pkl',cache())
            run.write(source/'summary.json',{'test_used':False,'development_questions':256,
                'bank_sha256':hashlib.sha256(b'').hexdigest()})
            bdir=root/'b';bdir.mkdir();bpath=bdir/'summary.json'
            baseline=stats([{'correct':0,'compute':10.,'judge_calls':0}]*256)
            b={'dev':{'accuracy':.7,'mean_compute':100.}}
            run.write(bpath,{'B':b,'ordinary_B':b,'matched_B':b,
                            'baselines':{name:baseline for name in run.study.BASELINES}})
            run.write(bdir/'identity.json',{})
            args=SimpleNamespace(data=data,models=models,responses=source,cache=source,cache_models=cache_models,
                b_reference=bpath,output=root/'output',device='cpu')
            calls=[]
            def evaluated(model,settings,rows,trees,judges,acquisition,lock):
                calls.append(settings)
                return [{'correct':1,'compute':50.,'judge_calls':1,'judge_physical_calls':2,
                    'metadata_calls':1,'metadata_cpu_seconds':.001,'review_neural_calls':0} for _ in rows]
            with ExitStack() as stack:
                stack.enter_context(patch.object(run,'model_assets',side_effect=lambda path:(fitted,'recipe')))
                stack.enter_context(patch.object(run.inherited,'validate_b'))
                stack.enter_context(patch.object(run.study,'development_view',return_value=(rows,trees,{})))
                stack.enter_context(patch.object(run.study,'statistics',side_effect=stats))
                stack.enter_context(patch.object(run.study,'fixed',return_value={'correct':0,'compute':10.,'judge_calls':0}))
                stack.enter_context(patch.object(run,'evaluate',side_effect=evaluated))
                run.run(args)
                self.assertEqual(len(calls),52)
                summary=run.read(args.output/'summary.json')
                self.assertEqual(summary['new_physical_generations'],0)
                self.assertFalse(summary['final_gap_claim'])
                self.assertEqual(summary['selected_C']['dev']['mean_judge_physical_calls'],2)
                self.assertEqual(len(run.read(args.output/'search.json')),52)
                run.run(args)
                self.assertEqual(len(calls),52)
                run.completed_bank(args.output,panel)

    def test_strong_review_physical_accounting_and_strict_B_ceiling(self):
        items=[{'metadata_calls':1,'metadata_cpu_seconds':.002,'judge_physical_calls':2,'review_neural_calls':0}]
        with patch.object(run.study,'statistics',return_value={'mean_judge_calls':1}):
            value=run.measured(items)
        self.assertEqual(value['mean_judge_calls'],1)
        self.assertEqual(value['mean_judge_physical_calls'],2)
        self.assertEqual(value['mean_metadata_cpu_ms'],2)
        candidates=[{'dev':{'accuracy':.9,'mean_compute':101.,'mean_judge_calls':1}},
                    {'dev':{'accuracy':.8,'mean_compute':90.,'mean_judge_calls':1}},
                    {'dev':{'accuracy':.95,'mean_compute':80.,'mean_judge_calls':0}}]
        result=run.study.select(candidates,100.)
        self.assertEqual(result['selected_C'],candidates[1]);self.assertEqual(result['best'],candidates[2])
        self.assertIsNone(run.study.select(candidates,80.)['selected_C'])


if __name__=='__main__':
    unittest.main()
