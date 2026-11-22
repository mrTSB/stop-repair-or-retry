from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import auxiliary_paired_contract as contract
import collect_auxiliary_pairs as run
from test_scoring_completion_availability import canonical
from test_scoring_observation import response


class Remote:
    def __init__(self,invalid=False):self.calls=[];self.invalid=invalid
    def __call__(self,endpoint,payload):
        if endpoint!='/api/generate':raise AssertionError('Unexpected endpoint')
        assert payload['model']==run.old.MODEL
        assert 'PRIVATE_REFERENCE' not in payload['prompt']
        self.calls.append(copy.deepcopy(payload))
        value=response(' B' if self.invalid else 'B');value['done_reason']='stop'
        return value


def read_all(root):
    return tuple(run.pilot.protocol.strict_records(root/name) for name in ('bank','raw','attempts'))


def journal(root,remote,raw=(),attempts=()):
    stream=(root/'raw').open('a');attempt=(root/'attempts').open('a')
    return run.Journal(remote,stream,attempt,raw,attempts),stream,attempt


def source_fixture(root):
    aux=root/'auxiliary';cache=root/'cache';data=root/'data';observer=root/'observer'
    for p in (aux/'provenance',cache,data/'splits',observer):p.mkdir(parents=True)
    train=[f'{subject}-{i}' for subject in ('chemistry','physics') for i in range(720)]
    cal=[f'{subject}-cal{i}' for subject in ('chemistry','physics') for i in range(80)]
    source=[{'uuid':qid,'discipline':'Science','field':qid.partition('-')[0].title(),'question':'Which option follows?',
        'options':['Carbon','Oxygen','Hydrogen','Helium'],'answer':'PRIVATE_REFERENCE','answer_letter':'PRIVATE_REFERENCE'} for qid in train+cal]
    run.io.jsonl(cache/'SuperGPQA-all.jsonl',source);run.io.write(cache/'pin.json',{'repo':'official','revision':'pin'})
    protected={s:s+'-sha' for s in ('train','dev','test')}
    metadata={'train_ids':train,'calibration_ids':cal,'repository':'official','revision':'pin','source_sha256':run.io.sha(cache/'SuperGPQA-all.jsonl'),
        'protected_split_sha256':protected,'source_code_sha256':run.io.sha(contract.reserve_science_aux.__file__)}
    run.io.write(aux/'metadata.json',metadata)
    exclusions={'ids':['supergpqa-'+qid for qid in train+cal],'metadata_sha256':run.io.sha(aux/'metadata.json')}
    run.io.write(aux/'provenance/historical-exclusions.json',exclusions)
    run.io.write(aux/'provenance/seal-receipt.json',{'metadata_sha256':run.io.sha(aux/'metadata.json'),
        'historical_exclusions_sha256':run.io.sha(aux/'provenance/historical-exclusions.json'),
        'sealer_sha256':run.io.sha(contract.auxiliary.__file__),'protected_overlap':0,'train_questions':1440,'calibration_questions':160,
        'source_sha256':metadata['source_sha256']})
    run.io.write(data/'splits/selection.json',{'science':{'sha256':metadata['source_sha256']},'splits':{s:{'sha256':v} for s,v in protected.items()}})
    legacy=root/'legacy.jsonl';run.io.jsonl(legacy,[{'id':'old-only','answer':'PRIVATE_REFERENCE'}])
    run.io.write(observer/'recipe.json',{'data_sha256':run.io.sha(legacy),'fold':'full','input_profile':'full-options-openstax-v1'})
    return SimpleNamespace(auxiliary=aux,cache=cache,data=data,observer=observer,legacy_examples=legacy)


class AuxiliaryContractTests(unittest.TestCase):
    def test_projection_never_decodes_reference_values(self):
        raw=b'{"uuid":"ok","question":"stem\\\"withescape","answer":{"deep":["PRIVATE_REFERENCE"]},"options":["A","B"]}'
        original=json.loads
        def checked(value,*args,**kwargs):
            if isinstance(value,(str,bytes)) and ('PRIVATE_REFERENCE' in value if isinstance(value,str) else b'PRIVATE_REFERENCE' in value):
                self.fail('Reference value decoded')
            return original(value,*args,**kwargs)
        with patch.object(contract.json,'loads',side_effect=checked):
            self.assertEqual(contract.project(raw,{'uuid','question'}),{'uuid':'ok','question':'stem"withescape'})
        with self.assertRaises(ValueError):contract.project(raw,{'answer'})
        with self.assertRaises(ValueError):contract.project(b'{"uuid":"one","uuid":"two"}',{'uuid'})

    def test_fixed_balanced_selection_all1440_and_no_calibration(self):
        subjects={f'{s}{i}':s for s in contract.COUNTS for i in range(720)}
        selected=contract.choose(subjects)
        self.assertEqual(Counter(v['subject'] for v in selected),contract.COUNTS)
        self.assertEqual(selected,contract.choose(dict(reversed(list(subjects.items())))))
        self.assertEqual(len({v['id'] for v in selected}),512)
        with self.assertRaises(ValueError):contract.choose(dict(list(subjects.items())[:-1]))

    def test_public_preflight_skips_labels_and_protected_files(self):
        with TemporaryDirectory() as temporary:
            args=source_fixture(Path(temporary));original=json.loads
            def checked(value,*a,**kw):
                if isinstance(value,bytes) and b'PRIVATE_REFERENCE' in value:self.fail('Reference text decoded')
                return original(value,*a,**kw)
            with run.guarded.training_only_reads(),patch.object(contract.json,'loads',side_effect=checked):
                rows,selected,receipt=contract.public_inputs(args)
            self.assertEqual(len(rows),512);self.assertTrue(all(r['answer']=='' for r in rows))
            self.assertFalse(any('-cal' in r['id'] for r in rows));self.assertFalse(receipt['reserved_question_files_opened'])
            self.assertEqual(receipt['legacy_encoder_overlap'],0);self.assertEqual(receipt['project_split_overlap'],0)
            altered=run.io.read(args.data/'splits/selection.json');altered['splits']['test']['sha256']='tampered'
            run.io.write(args.data/'splits/selection.json',altered)
            with self.assertRaises(ValueError):contract.public_inputs(args)

    def test_exact_plain_and_Complete_prompts_seeds_and_horizon(self):
        row=canonical();tree={}
        for path in contract.PATHS:
            payload=run.payload(row,path,tree)
            if path in contract.PLAIN:
                self.assertEqual(payload['prompt'],run.old.solver_prompt(row,path,tree,{}))
                self.assertEqual(payload['options']['num_predict'],128 if not path else 256)
                self.assertEqual(payload['options']['temperature'],0 if not path else .65)
            else:
                parent=path.removesuffix('/complete') if '/' in path else ''
                self.assertEqual(payload,run.pilot.request_payload(row,parent,tree)[0]);self.assertEqual(payload['options']['num_predict'],1)
            answer=response();answer['done_reason']='stop'
            raw={'endpoint':'/api/generate','payload':payload,'request_sha256':contract.fingerprint({'endpoint':'/api/generate','payload':payload}),
                'response':answer,'response_sha256':contract.fingerprint(answer),'started_unix':1.,'returned_unix':2.}
            item=run.logical(row,path,tree,raw,'registered');tree[path]=item['response']
            self.assertEqual(item['kind'],'solver');self.assertNotIn('features',item['response'])
            if path in contract.complete.PATHS:self.assertLessEqual(item['response']['total_answer_changing_continuations'],2)
        before=run.payload(row,'repair',tree);changed=copy.deepcopy(tree);changed['retry']['text']='FUTURE_SECRET'
        self.assertEqual(before,run.payload(row,'repair',changed))
        with self.assertRaises(ValueError):run.payload({**row,'answer':'PRIVATE_REFERENCE'},'',{})
        with self.assertRaises(ValueError):run.payload(row,'complete/repair',tree)


class AuxiliaryJournalTests(unittest.TestCase):
    def test_full5120_grid_fixed_membership_and_all_cost_receipts(self):
        rows=[canonical(i,s) for s in ('chemistry','physics') for i in range(256)]
        with TemporaryDirectory() as temporary:
            root=Path(temporary);remote=Remote();j,raw_stream,attempt_stream=journal(root,remote)
            with (root/'bank').open('a') as bank,raw_stream,attempt_stream:
                collector=run.Collector({},bank,j,'registered')
                with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(collector.one,rows))
            saved,raw,attempts=read_all(root);known,coverage=run.restore(rows,saved,raw,attempts,'registered',True)
            self.assertEqual(len(remote.calls),5120);self.assertEqual(coverage['physical_calls'],5120)
            self.assertEqual(len(known),5120);self.assertEqual(sum(v['response']['input_tokens'] for v in known.values()),512000)
            self.assertEqual(sum(v['response']['output_tokens'] for v in known.values()),5120)
            self.assertEqual(Counter(v['path'] for v in saved),dict.fromkeys(contract.PATHS,512))

    def test_parallel_all_paths_once_and_complete_resume(self):
        rows=[canonical(i,s) for s in ('chemistry','physics') for i in range(2)]
        with TemporaryDirectory() as temporary:
            root=Path(temporary);remote=Remote();j,raw_stream,attempt_stream=journal(root,remote)
            with (root/'bank').open('a') as bank,raw_stream,attempt_stream:
                collector=run.Collector({},bank,j,'registered')
                with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(collector.one,[*rows,rows[0]]))
            saved,raw,attempts=read_all(root);known,coverage=run.restore(rows,saved,raw,attempts,'registered',True)
            self.assertEqual(len(remote.calls),40);self.assertEqual(len(known),40);self.assertEqual(coverage['physical_attempts'],40)
            self.assertEqual(coverage['review_calls'],0);self.assertEqual(coverage['neural_calls'],0)
            j,raw_stream,attempt_stream=journal(root,lambda *a:self.fail('regenerated'),raw,attempts)
            with (root/'bank').open('a') as bank,raw_stream,attempt_stream:
                collector=run.Collector(known,bank,j,'registered')
                for row in rows:collector.one(row)
            self.assertEqual(read_all(root),(saved,raw,attempts))

    def test_orphan_return_recovers_without_duplicate_physical_call(self):
        row=canonical()
        with TemporaryDirectory() as temporary:
            root=Path(temporary);remote=Remote();j,raw_stream,attempt_stream=journal(root,remote)
            with raw_stream,attempt_stream:j('/api/generate',run.payload(row,'',{}))
            saved,raw,attempts=read_all(root);known,_=run.restore([row],saved,raw,attempts,'registered')
            self.assertFalse(known);self.assertEqual(len(remote.calls),1)
            j,raw_stream,attempt_stream=journal(root,remote,raw,attempts)
            with (root/'bank').open('a') as bank,raw_stream,attempt_stream:run.Collector(known,bank,j,'registered').one(row)
            saved,raw,attempts=read_all(root);run.restore([row],saved,raw,attempts,'registered',True)
            self.assertEqual(len(remote.calls),10)

    def test_unresolved_attempt_and_partial_tail_fail_closed(self):
        row=canonical()
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            def failed(*args):raise RuntimeError('transport ambiguity')
            j,raw_stream,attempt_stream=journal(root,failed)
            with raw_stream,attempt_stream:
                with self.assertRaisesRegex(RuntimeError,'transport'):j('/api/generate',run.payload(row,'',{}))
            _,raw,attempts=read_all(root);self.assertEqual(len(attempts),1);self.assertFalse(raw)
            with self.assertRaisesRegex(ValueError,'Unresolved'):run.attempts_checked(attempts,raw)
            (root/'raw').write_bytes(b'{"partial":')
            with self.assertRaises(ValueError):run.pilot.protocol.strict_records(root/'raw')

    def test_invalid_answers_retained_and_tampering_rejected(self):
        row=canonical()
        with TemporaryDirectory() as temporary:
            root=Path(temporary);remote=Remote(invalid=True);j,raw_stream,attempt_stream=journal(root,remote)
            with (root/'bank').open('a') as bank,raw_stream,attempt_stream:run.Collector({},bank,j,'registered').one(row)
            saved,raw,attempts=read_all(root);known,_=run.restore([row],saved,raw,attempts,'registered',True)
            self.assertEqual(len(known),10);self.assertEqual(len(remote.calls),10)
            self.assertEqual(known[row['id'],'']['response']['answer'],'')
            bad=copy.deepcopy(saved);bad[0]['response']['input_tokens']+=1
            with self.assertRaises(ValueError):run.restore([row],bad,raw,attempts,'registered')
            with self.assertRaises(ValueError):run.restore([row],saved,raw,attempts[:-1],'registered')
            with self.assertRaises(ValueError):run.restore([row],saved[:-1],raw,attempts,'registered',True)

    def test_registration_binds_public_rows_source_and_priority(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);args=SimpleNamespace(output=root/'new',registration_sha256=None)
            rows=[canonical()];record={'schema':run.SCHEMA,'source_files':{},'input_evidence':{'files':{}},'selection':'fixed'}
            with patch.object(run,'specification',return_value=(rows,record)):
                run.register(args);args.registration_sha256=run.io.sha(args.output/'registration.json');run.registered(args)
                changed=copy.deepcopy(rows);changed[0]['question']='other';run.io.jsonl(args.output/'public-questions.jsonl',changed)
                with self.assertRaises(ValueError):run.registered(args)
        with self.assertRaises(ValueError):contract.priority_receipt(Path('/tmp/not-the-priority/complete.json'))

    def test_blind_first_priority_gate_requires_exact_complete_coverage(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'runs/controller-semantic-followup/round2/blind-first-root-review-pilot-v1';folder.mkdir(parents=True)
            with patch.object(contract.io,'PROJECT',root):
                with self.assertRaises(FileNotFoundError):contract.priority_receipt(folder/'complete.json',True)
                registered={'schema':'blind-first-root-review-TRAIN-v1','source_files':{}}
                run.io.write(folder/'registration.json',registered)
                coverage={'questions':64,'logical_reviews':64,'first_guided':64,'second_repairs':64,
                    'physical_teacher_phases':128,'physical_compact_calls':128,'physical_calls':256}
                done={'schema':registered['schema'],'registration_sha256':run.io.sha(folder/'registration.json'),
                    'coverage':coverage,'training_only':True,'test_used':False,'new_neural_calls':0,'scored':False,
                    'digests':{'registration.json':run.io.sha(folder/'registration.json')}}
                run.io.write(folder/'complete.json',done);receipt=contract.priority_receipt(folder/'complete.json',True)
                self.assertEqual(receipt['sha256'],run.io.sha(folder/'complete.json'))
                done['coverage']['physical_calls']=255;run.io.write(folder/'complete.json',done)
                with self.assertRaisesRegex(ValueError,'256-call'):contract.priority_receipt(folder/'complete.json',True)


if __name__=='__main__':unittest.main()
