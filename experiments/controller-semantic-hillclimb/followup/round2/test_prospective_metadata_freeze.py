"""Synthetic provenance only; no project question banks or model inference."""
import copy
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import prospective_metadata_freeze as run
from test_metadata_review import make_adapter


def write(path, value):
    Path(path).write_text(json.dumps(value)+'\n')


def adapter_evidence():
    adapter = make_adapter()
    adapter.receipt.update(training_questions=192, training_records=320, registration_sha256='registered',
                           root_tables_use_nested_oof=True, neural_calls=0, runtime_temperature=1.)
    linear = {'state': adapter.state(), 'receipt': adapter.receipt,
              'final_calibration_training_only': True, 'root_emissions_do_not_use_final_temperature': True,
              'temperature_search': [{'temperature': t, 'loss': (t-1.)**2} for t in run.spec.TEMPERATURES]}
    return adapter, linear, {'adapter': adapter.receipt}


def diagnostic_fixture(directory):
    records = []
    for i in range(192):
        for protocol in (('root','critical','independent') if i < 64 else ('root',)):
            x = np.zeros(len(run.spec.FEATURES)); x[0]=1; x[3+('root','critical','independent').index(protocol)]=1
            x[14]=i/192; x[15]=(i%7)/7
            records.append({'id': f'synthetic-q{i}', 'protocol': protocol, 'features':x.tolist(),
                            'target':i%2, 'answer_present':True, 'subject':'math', 'bank':'new'})
    records = sorted(records, key=lambda r:(r['protocol'],r['id']))
    (directory/'private-features.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    ids={r['id'] for r in records}; nested=np.full(len(records),np.nan); folds=[]
    for fold in (0,1,2):
        held={qid for qid in ids if run.diagnostic.fold(qid)==fold}
        training=[r for r in records if r['id'] not in held]
        model=run.diagnostic.train_model('metadata_logistic',training)
        path=directory/f'metadata_logistic-outer{fold}.pkl';path.write_bytes(pickle.dumps(model))
        indexes=[i for i,r in enumerate(records) if r['id'] in held]
        nested[indexes]=run.diagnostic.predict(model,[records[i] for i in indexes])
        inner=[]
        for number in (0,1,2):
            local={qid for qid in ids-held if run.diagnostic.fold(qid,True)==number}
            inner.append({'fold':number,'held_ids':sorted(local),'training_ids':sorted(ids-held-local)})
        folds.append({'outer_fold':fold,'held_ids':sorted(held),'training_ids':sorted(ids-held),
                      'inner':inner,'temperature':1.,'model_sha256':run.sha(path)})
    np.savez(directory/'private-predictions.npz',metadata_logistic=nested)
    write(directory/'fold-receipts.json',{'metadata_logistic':folds})
    write(directory/'registration.json',{'specification_sha256':run.sha(run.spec.__file__),
                                       'features':run.spec.FEATURES,'seed':run.spec.SEED})
    write(directory/'implementation.json',{'source_files':{},'input_evidence':{},'records_sha256':run.fingerprint(records)})
    write(directory/'numerical-audit.json',{'synthetic':True})
    checks={'registration.json':'registration_sha256','implementation.json':'implementation_sha256',
            'private-features.jsonl':'private_features_sha256','private-predictions.npz':'predictions_sha256',
            'fold-receipts.json':'fold_receipts_sha256'}
    summary={key:run.sha(directory/name) for name,key in checks.items()}
    summary.update(strict_nested_question_calibration=True,development_or_test_used=False)
    write(directory/'summary.json',summary)
    return records,folds


def metadata_fit_fixture(root):
    diagnostic=root/'diagnostic';diagnostic.mkdir();records,_=diagnostic_fixture(diagnostic)
    _,raw,nested=run.diagnostic_evidence(diagnostic,set())
    folder=root/'fit';folder.mkdir();parent=root/'parent';parent.mkdir()
    write(parent/'identity.json',{'synthetic_parent':True})
    observer=SimpleNamespace(directory=str(root/'observer'),_model=None)
    subjects=('math','chemistry','physics')
    aligned=[{'id':r['id'],'subject':r['subject'],'state':i%8,'signal':i%32,
              'nested_probability':float(p),'outer_fold':run.diagnostic.fold(r['id']),
              'first_states':[(i+1)%8,(i+2)%8],'review_cost':100.+i}
             for i,(r,p) in enumerate(zip(records,nested)) if r['protocol']=='root']
    (folder/'root-oof-alignment.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in aligned))
    parent_specs=[];parents={}
    for smoothing in (8.,32.):
        dynamics={s:SimpleNamespace(review=np.ones((8,32))/32,reviewed_transition=np.ones((32,2,8,8))/8,
                    review_costs=np.full(8,200.),emission=np.ones((2,8,3))/3,initial=np.ones(8)/8)
                  for s in subjects}
        model=SimpleNamespace(base=SimpleNamespace(classifier=observer,dynamics={s:np.eye(8) for s in subjects}),
            transfer={s:np.eye(8) for s in subjects},mentor=SimpleNamespace(classifier=observer,
                review_observer=observer,regular=SimpleNamespace(calibration=np.eye(4)),dynamics=dynamics,
                fit_receipt={'semantic_review':True,'smoothing':smoothing}))
        key=f'smooth{smoothing:g}-review1';path=parent/(key+'.pkl');path.write_bytes(pickle.dumps(model))
        parent_specs.append({'key':key,'sha256':run.sha(path),'smoothing':smoothing,'semantic_review':True})
        parents[smoothing]=model
    base_candidate={'settings':{'penalty':.08},'dev':{'accuracy':.5}}
    ancestor={'models':parent_specs,'base_A':base_candidate,'fit_questions':512}
    write(parent/'complete.json',ancestor)
    source_files={str(Path(m.__file__).resolve()):run.sha(m.__file__) for m in (run.fitting,run.policy,run.paid,run.diagnostic,run.spec)}
    contract={'recipe':run.fitting.RECIPE,'source_files':source_files,'input_evidence':{},'diagnostic_directory':str(diagnostic)}
    write(folder/'registration.json',contract)
    adapter=make_adapter()
    search=[{'temperature':t,'loss':run.diagnostic.binary_loss([r['target'] for r in records],
        run.diagnostic.calibrated(raw,t,records),run.diagnostic.weights(records))} for t in run.spec.TEMPERATURES]
    adapter.temperature=min(search,key=lambda r:(r['loss'],r['temperature']))['temperature']
    adapter.receipt={'learned_state_sha256':run.paid.digest(adapter.state()),'training_questions':192,'training_records':320,
        'registration_sha256':run.sha(folder/'registration.json'),'runtime_temperature':adapter.temperature,
        'root_tables_use_nested_oof':True,'neural_calls':0}
    identity={'source_sha256':run.sha(run.fitting.__file__),'policy_module':run.MODULE,'cohort':'mixed',
        'registration_sha256':run.sha(folder/'registration.json'),'parent_models_directory':str(parent),
        'parent_complete_sha256':run.sha(parent/'complete.json'),'parent_identity_sha256':run.sha(parent/'identity.json'),
        'root_records':192,'adapter':adapter.receipt,'neural_calls':0,'cloud_calls':0,'development_or_test_used':False}
    write(folder/'identity.json',identity)
    write(folder/'trained-linear.json',{'state':adapter.state(),'receipt':adapter.receipt,'temperature_search':search,
        'final_calibration_training_only':True,'root_emissions_do_not_use_final_temperature':True})
    (folder/'adapter.pkl').write_bytes(pickle.dumps(adapter));specs=[]
    for original in parent_specs:
        smoothing=original['smoothing'];old=parents[smoothing];mentor=copy.deepcopy(old.mentor)
        updated=run.fitting.rebuild(aligned,mentor.dynamics,smoothing)
        for subject,dynamics in mentor.dynamics.items():
            dynamics.review,dynamics.reviewed_transition,dynamics.review_costs=updated[subject]
        mentor.review_observer=None;mentor.fit_receipt={'semantic_review':False,
            'root_oof_alignment_sha256':run.sha(folder/'root-oof-alignment.jsonl')}
        receipt={'policy_module':run.MODULE,'state_count':16,'maximum_reviews':1,'include_reviews':False,
                 'parent_transfer_unchanged':True,'metadata_adapter':adapter.receipt}
        model=run.policy.Planner(old.base,mentor,old.transfer,'repairability_pomdp',{'penalty':.08},receipt,metadata_adapter=adapter)
        key=f'metadata-smooth{smoothing:g}';path=folder/(key+'.pkl');path.write_bytes(pickle.dumps(model))
        specs.append({'key':key,'sha256':run.sha(path),'parent_sha256':original['sha256'],'smoothing':smoothing,
            'semantic_review':False,'metadata_review':True,'policy_module':run.MODULE,
            'preserved_components_sha256':run.fitting.preserved_digest(old)})
    complete={'models':specs,'base_A':base_candidate,'fit_questions':512,'policy_module':run.MODULE,
        'identity_sha256':run.sha(folder/'identity.json'),'registration_sha256':run.sha(folder/'registration.json'),
        'artifacts':{name:run.sha(folder/name) for name in ('trained-linear.json','adapter.pkl','root-oof-alignment.jsonl')},
        'paired_review_questions':192,'neural_calls':0,'cloud_calls':0,'development_or_test_used':False}
    write(folder/'complete.json',complete)
    return folder,parent,ancestor


class MetadataProvenanceTests(unittest.TestCase):
    def test_structural_identity_survives_pickle_stride_aliases_but_detects_values(self):
        value=SimpleNamespace(matrix=np.arange(12.).reshape(3,4),_cache={'runtime':'ignored'},_asset_pool={'loaded':True})
        restored=pickle.loads(pickle.dumps(value));restored.matrix=np.asfortranarray(restored.matrix)
        restored._cache['runtime']='different'
        self.assertEqual(run.structural_digest(value),run.structural_digest(restored))
        restored.matrix[0,0]+=1
        self.assertNotEqual(run.structural_digest(value),run.structural_digest(restored))
        value.matrix[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):run.structural_digest(value)

    def test_full_adapter_state_temperature_and_receipt_are_bound(self):
        adapter,linear,identity=adapter_evidence()
        run.validate_adapter(adapter,linear,identity,'registered')
        changed=copy.deepcopy(linear);changed['state']['scale'][0]*=2
        with self.assertRaises(ValueError):run.validate_adapter(adapter,changed,identity,'registered')
        changed=copy.deepcopy(linear);changed['root_emissions_do_not_use_final_temperature']=False
        with self.assertRaises(ValueError):run.validate_adapter(adapter,changed,identity,'registered')
        with self.assertRaises(ValueError):run.validate_adapter(adapter,linear,identity,'other-registration')

    def test_training_nested_forecasts_and_model_hashes_are_verified(self):
        with TemporaryDirectory() as temporary:
            directory=Path(temporary);records,_=diagnostic_fixture(directory);files=set()
            actual,raw,nested=run.diagnostic_evidence(directory,files)
            self.assertEqual(actual,records);self.assertEqual(raw.shape,(320,));np.testing.assert_allclose(raw,nested,atol=1e-15)
            self.assertTrue(all((directory/f'metadata_logistic-outer{i}.pkl').resolve() in files for i in (0,1,2)))
            (directory/'metadata_logistic-outer0.pkl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):run.diagnostic_evidence(directory,set())

    def test_nested_question_leak_fails_even_if_new_receipt_hashes_are_consistent(self):
        with TemporaryDirectory() as temporary:
            directory=Path(temporary);_,folds=diagnostic_fixture(directory)
            folds[0]['inner'][0]['training_ids'].append(folds[0]['held_ids'][0])
            write(directory/'fold-receipts.json',{'metadata_logistic':folds})
            summary=run.read(directory/'summary.json');summary['fold_receipts_sha256']=run.sha(directory/'fold-receipts.json')
            write(directory/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'leaks'):run.diagnostic_evidence(directory,set())

    def test_scope_restores_legacy_dispatch_and_keeps_original_target_function(self):
        args=SimpleNamespace();original=(run.legacy.select_inputs,run.legacy.fitting_chain,run.legacy.validate_role,dict(run.legacy.C_MODULES))
        targets=run.legacy.check_targets
        with run.Provenance(args).installed():
            self.assertIn(run.MODULE,run.legacy.C_MODULES)
            self.assertIs(run.legacy.check_targets,targets)
        self.assertEqual((run.legacy.select_inputs,run.legacy.fitting_chain,run.legacy.validate_role,run.legacy.C_MODULES),original)

    def test_missing_metadata_provenance_fails_before_model_loading(self):
        frozen={'selected':{'C':{'module':run.MODULE}},'completed_A_B_registry':{}}
        with patch.object(run.legacy,'verify',return_value=frozen):
            with self.assertRaisesRegex(ValueError,'Missing explicit'):run.verify(Path('/synthetic'))

    def test_saved_metadata_chain_rebuilds_tables_and_preserves_parent_components(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);folder,parent,ancestor=metadata_fit_fixture(root)
            provenance=run.Provenance(SimpleNamespace())
            def inherited_chain(directory,*args):
                self.assertEqual(directory,parent.resolve())
                return ancestor,[parent/'complete.json',parent/'identity.json']
            with patch.object(run.inherited.Provenance,'chain',side_effect=inherited_chain),patch.object(run,'ORIGINAL_ROLE'):
                complete,files=provenance.chain(folder,root)
                self.assertEqual(len(complete['models']),2)
                self.assertEqual(provenance.metadata_receipts[str(folder.resolve())]['root_records'],192)
                self.assertIn((folder/'adapter.pkl').resolve(),files)
                # Even self-consistently edited manifest/file hashes cannot hide
                # a changed root transition table from independent reconstruction.
                path=folder/(complete['models'][0]['key']+'.pkl');model=pickle.loads(path.read_bytes())
                model.mentor.dynamics['math'].reviewed_transition[0,0,0,0]+=.01
                path.write_bytes(pickle.dumps(model));complete['models'][0]['sha256']=run.sha(path)
                write(folder/'complete.json',complete)
                with self.assertRaises(AssertionError):run.Provenance(SimpleNamespace()).chain(folder,root)


if __name__=='__main__':unittest.main()
