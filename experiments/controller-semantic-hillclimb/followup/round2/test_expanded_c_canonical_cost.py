"""Synthetic exact-request canonicalization, zero-forward and cost proofs."""
from contextlib import ExitStack, nullcontext
import copy
import hashlib
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import expanded_c_canonical_cost as run
from prospective_parallel import serial


def raw(text,tokens,start=1.,returned=2.):
    payload={'model':run.previous.study.old.MODEL,'prompt':'Exact paid prompt','options':{'seed':42,'temperature':.65,'num_predict':128,'num_ctx':4096}}
    response={'model':run.previous.study.old.MODEL,'response':text,'prompt_eval_count':100,'eval_count':tokens,'done_reason':'stop','done':True}
    return {'endpoint':'/api/generate','payload':payload,'response':response,'started_unix':start,'returned_unix':returned,
        'request_sha256':serial.fingerprint({'endpoint':'/api/generate','payload':payload}),
        'response_sha256':serial.fingerprint(response)}


def logical(record,answer='A'):
    p,r=record['payload'],record['response'];o=p['options']
    return {'id':'q','kind':'solver','path':'retry/reuse','response':{'model':p['model'],
        'model_digest':run.previous.study.old.MODEL_DIGEST,
        'prompt_sha256':hashlib.sha256(p['prompt'].encode()).hexdigest(),'seed':o['seed'],'temperature':o['temperature'],
        'output_cap':o['num_predict'],'context_length':o['num_ctx'],'text':r['response'],'answer':answer,
        'input_tokens':r['prompt_eval_count'],'output_tokens':r['eval_count'],'finish':r['done_reason']}}


def outcome():
    return {'id':'q','path':'retry/reuse','correct':1,'decisions':[{'path':'','queried':True},{'path':'retry','queried':False}],
        'solver_calls':3,'judge_calls':1,'judge_tokens':100,'judge_physical_calls':2,
        'verifier_components':[{'kind':'solver','path':'','slots':32}],'verifier_calls':1,'verifier_token_slots':32,
        'solver_tokens':300,'compute':500.,'total_parameter_compute':600.,'metadata_calls':0,'metadata_cpu_seconds':0.}


class CanonicalCostTests(unittest.TestCase):
    def test_canonical_collection_requires_exact_source_panel_registration(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';data.mkdir();canonical=root/'canonical';canonical.mkdir()
            run.io.write(data/'dev-plain-complete.json',{})
            registration={'schema':run.canonical_protocol.SCHEMA,'paths':{'data':str(data.resolve()),'output':str(canonical.resolve())},
                'evidence':{'dev_questions_sha256':'panel','plain_completion_sha256':run.io.sha(data/'dev-plain-complete.json')},
                'model_digest':run.previous.study.old.MODEL_DIGEST,'source_files':{},'test_used':False}
            run.io.write(canonical/'registration.json',registration)
            marker={'schema':run.canonical_protocol.SCHEMA,'scope':run.canonical_protocol.SCOPE,
                'source_sha256':run.io.sha(run.canonical_protocol.__file__),'registration_sha256':run.io.sha(canonical/'registration.json'),
                'digests':{'registration.json':run.io.sha(canonical/'registration.json')}}
            run.io.write(canonical/'collection-complete.json',marker);args=SimpleNamespace(data=data,canonical=canonical)
            run.validate_canonical(args,'panel')
            with self.assertRaisesRegex(ValueError,'exact source/registration/panel'):run.validate_canonical(args,'other-panel')
            marker['source_sha256']='other-code';run.io.write(canonical/'collection-complete.json',marker)
            with self.assertRaisesRegex(ValueError,'exact source/registration/panel'):run.validate_canonical(args,'panel')

    def test_uniform_earlier_exact_request_not_shorter_or_favourable_selection(self):
        earlier=raw('Earlier longer explanation',40);current=raw('Later short',10,3.,4.)
        mapping,receipt=run.replacements([logical(current)],[current],[logical(earlier)],[earlier])
        self.assertEqual(mapping['q','retry/reuse']['text'],earlier['response']['response'])
        self.assertEqual(receipt[0]['canonical_tokens']-receipt[0]['current_tokens'],30)
        bad=logical(earlier,'B')
        with self.assertRaisesRegex(ValueError,'answer-identical'):run.replacements([logical(current)],[current],[bad],[earlier])
        bad=logical(earlier);bad['response']['model_digest']='changed'
        with self.assertRaisesRegex(ValueError,'frozen compact'):run.replacements([logical(current)],[current],[bad],[earlier])
        late=raw('Late canonical',40,5.,6.)
        with self.assertRaisesRegex(ValueError,'earlier'):run.replacements([logical(current)],[current],[logical(late)],[late])

    def test_nonterminal_or_inexact_request_cannot_enter_derivative(self):
        earlier=raw('Prior',20);current=raw('New',10,3.,4.)
        node=logical(current);node['path']='guided'
        with self.assertRaisesRegex(ValueError,'Only terminal'):run.replacements([node],[current],[logical(earlier)],[earlier])
        earlier['payload']['options']['seed']=99
        earlier['request_sha256']=serial.fingerprint({'endpoint':earlier['endpoint'],'payload':earlier['payload']})
        with self.assertRaisesRegex(ValueError,'exact earlier'):run.replacements([logical(current)],[current],[logical(earlier)],[earlier])

    def test_identical_policy_and_exact_four_B_terminal_token_delta_required(self):
        original=outcome();canonical=copy.deepcopy(original)
        old={'retry/reuse':{'input_tokens':100,'output_tokens':10}}
        new={'retry/reuse':{'input_tokens':100,'output_tokens':17}}
        canonical['solver_tokens']+=7
        for field in ('compute','total_parameter_compute'):canonical[field]+=4*7/3.09
        self.assertEqual(run.assert_same_policy(original,canonical,old,new),7)
        canonical['decisions'][1]['path']='guided'
        with self.assertRaisesRegex(ValueError,'policy/answer/observation'):run.assert_same_policy(original,canonical,old,new)

    def test_actual_observer_uses_cache_and_blocks_missing_forward_and_terminal(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary);run.io.write(folder/'recipe.json',{'parameters':435_000_000,'max_length':512})
            observer=run.perception.Observer(str(folder),1.)
            model=SimpleNamespace(classifier=observer)
            row={'id':'q','answer':''};tree={'':{'prompt_sha256':'a'*64,'text':'paid'}}
            key=observer.key(row,'',tree[''],'solver');cache={key:(np.zeros(4),np.zeros(8),np.zeros(8),32)}
            with run.cached_model(model,{run.previous.previous.share_key(observer):cache}):
                self.assertEqual(observer.values(row,'',tree)[3],32)
                with self.assertRaisesRegex(ValueError,'terminal'):observer.values(row,'retry/reuse',tree)
                with self.assertRaisesRegex(ValueError,'Reference'):observer.values({**row,'answer':'private'},'',tree)
                with patch.object(observer,'entry',return_value={}):
                    with self.assertRaisesRegex(ValueError,'no neural'):observer.values(row,'',{'':{**tree[''],'text':'changed'}})
            self.assertIsNone(observer._model)

    def test_complete156_paired_replay_keeps_original_stats_and_seals(self):
        with TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary);source=root/'source';(source/'models').mkdir(parents=True);output=root/'result'
            args=SimpleNamespace(data=root/'data',source=source,canonical=root/'canonical',output=output,mode='register')
            rows=[{'id':f'q{i}'} for i in range(256)];trees={r['id']:{'retry/reuse':{'input_tokens':100,'output_tokens':10}} for r in rows}
            canonical={qid:{'retry/reuse':{'input_tokens':100,'output_tokens':17}} for qid in trees}
            base=outcome()
            statistic=lambda items:{'n':len(items),'accuracy':float(np.mean([r['correct'] for r in items])),
                'mean_compute':float(np.mean([r['compute'] for r in items])),'mean_judge_calls':1.}
            stack.enter_context(patch.object(run.previous.study,'statistics',side_effect=statistic))
            stats=run.previous.measured([base]*256)
            candidates=[];specs=[]
            for i in range(6):
                name=f'model{i}';(source/'models'/(name+'.pkl')).write_bytes(pickle.dumps({'model':name}));specs.append({'key':name})
                for j in range(26):candidates.append({'model':name,'settings':{'index':j},'policy_module':'synthetic','dev':stats})
            run.io.write(source/'summary.json',{'B':{'dev':{'mean_compute':520.}},'selected_C':candidates[0],'prior_C':[]})
            bundle=(rows,trees,canonical,{},candidates,{'models':specs},{},[{'synthetic':True}],{})
            stack.enter_context(patch.object(run,'inputs',return_value=bundle))
            stack.enter_context(patch.object(run,'cached_model',return_value=nullcontext()))
            calls=[]
            def dispatch(model,module,row,tree,judges,settings):
                calls.append(row['id']);result=copy.deepcopy(base);result['id']=row['id']
                delta=tree['retry/reuse']['output_tokens']-10;result['solver_tokens']+=delta
                for name in ('compute','total_parameter_compute'):result[name]+=4*delta/3.09
                return result
            stack.enter_context(patch.object(run.previous,'dispatch',side_effect=dispatch))
            run.run(args);args.mode='run';args.registration_sha256=run.io.sha(output/'registration.json');run.run(args)
            self.assertEqual(len(calls),156*256*2)
            summary=run.io.read(output/'summary.json')
            self.assertTrue(summary['all_decisions_answers_and_encoder_ledgers_identical'])
            self.assertTrue(summary['original_selected_still_cheaper_than_B']);self.assertFalse(summary['selection_changed'])
            marker=(output/'complete.json').read_bytes();run.run(args)
            self.assertEqual(len(calls),156*256*2);self.assertEqual((output/'complete.json').read_bytes(),marker)


if __name__=='__main__':unittest.main()
