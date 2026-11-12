import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stop_repair_retry import llm_experiment as e

class LLMExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.c=e.load_config(e.ROOT/'experiments/llm-controller/config.example.json')
        self.prompt=e.DEFAULT_PROMPT.read_text();self.data=e.DEFAULT_DATA
        self.row=e.load_dataset(self.data,'dev')[0]
        self.guard=patch.object(e.HTTPClient,'__call__',side_effect=AssertionError('Tests must never use the network'));self.guard.start()
    def tearDown(self):self.guard.stop();self.tmp.cleanup()
    def config(self,**kw):
        c={**self.c,'endpoint':'https://router.invalid/v1/chat/completions','model':'test-model','billing_mode':'token','input_usd_per_million':0,'output_usd_per_million':0,'spend_limit_usd':1,'request_reserve_usd':.01};c.update(kw);return c
    def response(self,action='stop',inp=100,out=8,model='test-model'):
        return {'model':model,'id':'fixture','choices':[{'message':{'content':json.dumps({'action':action})},'finish_reason':'stop'}],'usage':{'prompt_tokens':inp,'completion_tokens':out,'total_tokens':inp+out}}
    def run_live(self,client=None,limit=1,c=None,out=None,split='dev',freeze=None):
        return e.run(c or self.config(),self.prompt,self.data,split,limit,out or self.root/'run','live',freeze,client or (lambda body:self.response()))
    def test_allowlist_excludes_gold_and_unvisited_branches(self):
        for path in ('','repair','retry'):
            original=e.request_body(self.row,path,self.c,self.prompt);row=copy.deepcopy(self.row)
            row['answer']='SECRET_GOLD';row['reference_solution']='SECRET_REFERENCE'
            for p,n in row['tree'].items():
                n['correct']='SECRET_CORRECT'
                if p not in ['',path]:n['text']='SECRET_FUTURE'
            self.assertEqual(original,e.request_body(row,path,self.c,self.prompt))
            self.assertNotIn('SECRET_',json.dumps(original))
        with self.assertRaises(e.ExperimentError):e.observable(self.row,'repair/retry',.01)
    def test_prompt_describes_budget_actions_cost_and_untrusted_data(self):
        for term in ['stop','repair','retry','two additional','controller input/output','untrusted','latest']:
            self.assertIn(term,self.prompt)
    def test_only_five_fixed_baselines(self):
        self.assertEqual(set(e.FIXED),{'stop','repair_once','repair_twice','retry_once','retry_twice'})
    def test_api_cap_and_schema_configuration(self):
        c={**self.c,'response_format':'json_schema','output_limit_parameter':'max_completion_tokens','temperature':None}
        b=e.request_body(self.row,'',c,self.prompt)
        self.assertEqual(b['max_completion_tokens'],128);self.assertNotIn('temperature',b)
        self.assertEqual(b['response_format']['json_schema']['schema']['properties']['action']['enum'],list(e.ACTIONS))
        self.assertFalse(b['stream'])
    def test_redirects_rejected(self):self.assertIsNone(e.NoRedirect().redirect_request(None,None,302,'redirect',{},'https://other.invalid'))
    def test_http_errors_do_not_expose_key_or_provider_body(self):
        self.guard.stop()
        c=self.config();secret='fixture-key-not-real'
        error=e.urllib.error.HTTPError(c['endpoint'],401,'unauthorized',{},io.BytesIO(secret.encode()))
        with patch.dict(os.environ,{c['api_key_env']:secret}), patch.object(e.urllib.request,'build_opener') as opener:
            opener.return_value.open.side_effect=error
            with self.assertRaises(e.ExperimentError) as caught:e.HTTPClient(c)({'messages':[]})
            self.assertEqual(caught.exception.http_status,401)
            self.assertNotIn(secret,str(caught.exception));self.assertNotIn('Authorization',str(caught.exception))
            self.assertEqual(opener.return_value.open.call_count,1)
    def test_live_requires_configuration_and_key(self):
        with self.assertRaises(e.ExperimentError):e.check_live(self.c)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(e.ExperimentError):e.check_live(self.config())
        for url in ['http://router.invalid/v1/chat/completions','https://user:secret@router.invalid/x','https://router.invalid/x?key=abc']:
            with self.assertRaises(e.ExperimentError):e.check_live(self.config(endpoint=url),need_key=False)
    def test_compute_budget_explicit_no_fabricated_dollars(self):
        c={**self.c,'endpoint':'https://example.modal.direct/v1/chat/completions','model':'endpoint-model','billing_mode':'compute'}
        with self.assertRaises(e.ExperimentError):e.check_live(c,need_key=False)
        c['compute_budget_confirmed']=True;e.check_live(c,need_key=False)
        self.assertIsNone(e.decode(self.response(),c)['estimated_usd'])
    def test_stop_charges_controller_and_initial_solver(self):
        d=self.run_live();m,rows,rr=e.completed_records(d);x=rr[0]
        self.assertEqual(x['tokens'],self.row['tree']['']['input_tokens']+self.row['tree']['']['output_tokens']+108)
        self.assertEqual(x['controller_calls'],1);self.assertEqual(x['path'],'');self.assertEqual(x['actions'],['stop'])
    def test_two_actions_force_latest_submission_no_third_call(self):
        bodies=[]
        def client(body):
            bodies.append(body);return self.response('repair' if len(bodies)==1 else 'retry')
        d=self.run_live(client);_,_,rr=e.completed_records(d)
        self.assertEqual(len(bodies),2);self.assertEqual(rr[0]['path'],'repair/retry')
        self.assertEqual(rr[0]['answer'],self.row['tree']['repair/retry']['answer'])
        self.assertEqual(len(json.loads(bodies[1]['messages'][1]['content'])['visited_responses']),2)
    def test_strict_action_invalid_is_counted_stop(self):
        r=self.response();r['choices'][0]['message']['content']='```json\n{"action":"retry"}\n```'
        d=self.run_live(lambda body:r);_,_,rr=e.completed_records(d)
        self.assertEqual(rr[0]['actions'],['stop']);self.assertEqual(rr[0]['invalid_actions'],1)
        self.assertEqual(rr[0]['controller_output_tokens'],8)
        for text in ['{"action":"repair","answer":"4"}','{"action":null}','{"action":[]}']:
            r['choices'][0]['message']['content']=text;self.assertFalse(e.decode(r,self.config())['valid_action'])
    def test_missing_usage_never_zero_cost(self):
        r=self.response();r.pop('usage')
        with self.assertRaises(e.ExperimentError):self.run_live(lambda body:r)
        req,res,pending,_=e.journal(self.root/'run',self.config());self.assertEqual(len(pending),1);self.assertEqual(len(res),0)
        self.assertTrue(any(v['type']=='raw_response' for v in e.events(self.root/'run')))
        with self.assertRaises(e.ExperimentError):e.evaluate(self.root/'run')
    def test_bad_usage_totals_and_missing_model_rejected(self):
        for change in ['total','negative','bool','model']:
            r=self.response()
            if change=='total':r['usage']['total_tokens']=999
            if change=='negative':r['usage']['completion_tokens']=-1
            if change=='bool':r['usage']['prompt_tokens']=True
            if change=='model':r.pop('model')
            with self.assertRaises(e.ExperimentError):e.decode(r,self.config())
    def test_reasoning_tokens_not_double_counted(self):
        r=self.response(out=40);r['usage']['completion_tokens_details']={'reasoning_tokens':30}
        self.assertEqual(e.decode(r,self.config())['output_tokens'],40)
    def test_request_persisted_before_send_and_timeout_blocks_resume(self):
        counter=[]
        def client(body):
            self.assertEqual(e.events(self.root/'run')[-1]['type'],'request');counter.append(1);raise TimeoutError()
        with self.assertRaises(e.ExperimentError):self.run_live(client)
        with self.assertRaises(e.ExperimentError):self.run_live(client)
        self.assertEqual(len(counter),1)
    def test_provider_receipt_recovery_does_not_send_again(self):
        with self.assertRaises(e.ExperimentError):self.run_live(lambda b:(_ for _ in ()).throw(TimeoutError()))
        d=self.root/'run';rid=next(iter(e.journal(d,self.config())[2]));receipt=self.root/'receipt.json';receipt.write_text(json.dumps(self.response()))
        e.recover(d,rid,receipt,.4)
        self.run_live(lambda b:self.fail('Recovered response must not be requested again'))
        self.assertEqual(e.completed_records(d)[2][0]['controller_latency_seconds'],.4)
    def test_resume_makes_no_duplicate_calls(self):
        self.run_live();self.run_live(lambda b:self.fail('Unexpected duplicate request'))
        self.assertEqual(len(e.journal(self.root/'run',self.config())[0]),1)
    def test_changed_config_or_prompt_rejected_on_resume(self):
        self.run_live()
        with self.assertRaises(e.ExperimentError):self.run_live(c=self.config(penalty=.02))
        with self.assertRaises(e.ExperimentError):e.run(self.config(),self.prompt+' changed',self.data,'dev',1,self.root/'run','live',client=lambda b:self.response())
    def test_budget_checked_before_next_call(self):
        c=self.config(input_usd_per_million=10,output_usd_per_million=0,spend_limit_usd=.0105)
        calls=[]
        def client(b):calls.append(1);return self.response('repair')
        with self.assertRaises(e.ExperimentError):self.run_live(client,c=c)
        self.assertEqual(len(calls),1)
    def test_reserve_overrun_halts_and_cannot_silently_resume(self):
        c=self.config(input_usd_per_million=200);calls=[]
        def client(b):calls.append(1);return self.response('repair')
        with self.assertRaises(e.ExperimentError):self.run_live(client,c=c)
        with self.assertRaises(e.ExperimentError):self.run_live(client,c=c)
        self.assertEqual(len(calls),1)
    def test_request_cap_stops_before_next_call(self):
        with self.assertRaises(e.ExperimentError):self.run_live(lambda b:self.response('retry'),c=self.config(max_requests=1))
        self.assertEqual(len(e.journal(self.root/'run',self.config(max_requests=1))[0]),1)
    def test_lock_prevents_concurrent_runs(self):
        d=self.root/'run';d.mkdir();(d/'.lock').write_text('fixture')
        with self.assertRaises(e.ExperimentError):self.run_live()
    def test_test_gate_rejects_unfrozen_and_mock(self):
        with self.assertRaises(e.ExperimentError):self.run_live(split='test')
        d=e.run(self.c,self.prompt,self.data,'dev',1,self.root/'mock','mock')
        with self.assertRaises(e.ExperimentError):e.freeze_run(d,self.root/'freeze.json')
        with self.assertRaises(e.ExperimentError):e.evaluate(self.root/'missing')
    def test_full_dev_freeze_evidence_and_reported_model_gate(self):
        dev=self.run_live(limit=None,out=self.root/'dev');f=self.root/'freeze.json';e.freeze_run(dev,f)
        self.run_live(out=self.root/'test',split='test',freeze=f)
        with self.assertRaises(e.ExperimentError):self.run_live(lambda b:self.response(model='changed'),out=self.root/'other-test',split='test',freeze=f)
        with self.assertRaises(e.ExperimentError):e.evaluate(self.root/'other-test')
        with (dev/'events.jsonl').open('a') as out:out.write(json.dumps({'type':'note'})+'\n')
        with self.assertRaises(e.ExperimentError):self.run_live(out=self.root/'changed-dev-test',split='test',freeze=f)
    def test_incomplete_dev_cannot_freeze(self):
        d=self.run_live()
        with self.assertRaises(e.ExperimentError):e.freeze_run(d,self.root/'freeze.json')
    def test_landscape_uses_training_only_and_no_fit(self):
        r=e.landscape(self.data,self.root/'landscape')
        self.assertEqual(r['split'],'train');self.assertEqual(r['questions'],100)
        self.assertEqual(sum(x['edges'] for x in r['transitions']),600)
        self.assertEqual(sum(x['nodes'] for x in r['confidence_counts']),700)
        self.assertEqual(len(r['fixed_baseline_reward_grid']),25)
        self.assertEqual({x['baseline'] for x in r['fixed_baseline_reward_grid']},set(e.FIXED))
    def test_mock_evaluation_clearly_labeled_and_baselines_match(self):
        d=e.run(self.c,self.prompt,self.data,'dev',2,self.root/'mock','mock');s=e.evaluate(d)
        self.assertEqual(s['mode'],'mock');self.assertIsNone(s['estimated_controller_usd'])
        self.assertIn('NOT LLM PERFORMANCE',(d/'report.md').read_text())
        self.assertEqual({x['policy'] for x in s['summaries']},{'llm_controller',*e.FIXED})
        rr=next(x for x in s['summaries'] if x['domain']=='all' and x['policy']=='repair_once')
        rows=e.load_dataset(self.data,'dev')[:2]
        self.assertEqual(rr['accuracy'],sum(r['tree']['repair']['correct'] for r in rows)/2)

if __name__=='__main__':unittest.main()
