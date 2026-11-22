import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import completion_v4_diagnostic as diagnostic


def numeric_fixture():
    rng=np.random.default_rng(238)
    rows=[]
    for i in range(256):
        for path in ('','repair','retry'):
            rows.append({'id':f'q{i}','subject':'chemistry' if i<128 else 'physics',
                'path':path,'fold':diagnostic.refined.fold_for(f'q{i}')})
    old=rng.normal(size=(768,25));new=old.copy();new[:,:20]+=rng.normal(size=(768,20))*.1
    return {'records':rows,'features':dict(zip(diagnostic.VARIANTS,(old,new))),
        'labels':rng.integers(0,16,size=768),'valid':np.ones(768,dtype=bool)}


class CompletionV4Tests(unittest.TestCase):
    def test_whole_question_fold_and_held_labels_cannot_change_predictions(self):
        data=numeric_fixture()
        held,p,model,receipt=diagnostic.outer_prediction(data,diagnostic.VARIANTS[0],0)
        changed=copy.deepcopy(data);changed['labels'][held]=(changed['labels'][held]+5)%16
        held2,p2,model2,_=diagnostic.outer_prediction(changed,diagnostic.VARIANTS[0],0)
        np.testing.assert_array_equal(held,held2);np.testing.assert_array_equal(p,p2)
        training={r['id'] for r,flag in zip(data['records'],held) if not flag}
        excluded={r['id'] for r,flag in zip(data['records'],held) if flag}
        self.assertFalse(training&excluded);self.assertEqual(len(training)+len(excluded),256)
        self.assertEqual(receipt['held_questions']*3,int(held.sum()))
        np.testing.assert_allclose(model[0].mean_,data['features'][diagnostic.VARIANTS[0]][~held].mean(0))

    def test_invalid_support_is_zero_without_removing_rows(self):
        data=numeric_fixture();data['valid'][::2]=False
        data['labels'][::2]%=8
        held,p,_,_=diagnostic.outer_prediction(data,diagnostic.VARIANTS[0],1)
        invalid=~data['valid'][held]
        self.assertGreater(invalid.sum(),0)
        np.testing.assert_array_equal(p[invalid][:,diagnostic.core.CURRENT==1],0.)
        np.testing.assert_allclose(p.sum(1),1.)

    def test_table_keeps_exact_cohort_and_only_raw20_can_differ(self):
        rows=[{'id':f'q{i}','subject':'chemistry' if i<128 else 'physics','answer':'A'} for i in range(256)]
        paths=('','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry','complete','repair/complete','retry/complete')
        trees={r['id']:{p:{'answer':'A' if p.endswith('complete') else 'B','text':p,'prompt_sha256':p} for p in paths} for r in rows}
        old={(r['id'],p):(np.zeros(4),np.zeros(8),np.zeros(8),512) for r in rows for p in ('','repair','retry')}
        new={key:(a+1,b+1,c+1,512) for key,(a,b,c,_) in old.items()}
        def observations(row,path,paid,neural):
            self.assertEqual(row['answer'],'');self.assertEqual(set(paid),{'',path})
            return [*np.concatenate(neural[:3]),float(bool(path)),len(paid[path]['text'])]
        with patch.object(diagnostic.core.plain,'observables',side_effect=observations):
            data=diagnostic.table(rows,trees,old,new)
            changed=copy.deepcopy(trees);changed['q0']['repair/complete']['answer']='B'
            other=diagnostic.table(rows,changed,old,new)
            for name in diagnostic.VARIANTS:np.testing.assert_array_equal(data['features'][name],other['features'][name])
            self.assertNotEqual(data['labels'][1],other['labels'][1])
            with self.assertRaises(ValueError):diagnostic.table(rows[:-1],trees,old,new)
        self.assertEqual(data['features'][diagnostic.VARIANTS[0]].shape,(768,22))
        np.testing.assert_array_equal(data['features'][diagnostic.VARIANTS[0]][:,20:],data['features'][diagnostic.VARIANTS[1]][:,20:])

    def test_alignment_rejects_metadata_and_fold_changes(self):
        metadata=[{'id':qid,'kind':'solver','path':p,'fold':diagnostic.refined.fold_for(qid),
            'left':qid,'right':p,'numeric':[1.]} for qid in ('q0','q1') for p in ('','repair','retry')]
        arrays={name:np.ones((6,n)) for name,n in (('state',4),('plain',8),('guided',8))}
        prompts={(r['id'],r['kind'],r['path']):'prompt' for r in metadata}
        lookup,_=diagnostic.refined.aligned_lookup(metadata,arrays,copy.deepcopy(metadata),prompts,{'q0','q1'})
        self.assertEqual(len(lookup),6)
        changed=copy.deepcopy(metadata);changed[0]['fold']=(changed[0]['fold']+1)%3
        with self.assertRaises(ValueError):diagnostic.refined.aligned_lookup(changed,arrays,metadata,prompts,{'q0','q1'})
        changed=copy.deepcopy(metadata);changed[0]['numeric']=[2.]
        with self.assertRaises(ValueError):diagnostic.refined.aligned_lookup(changed,arrays,metadata,prompts,{'q0','q1'})

    def test_no_neural_load_and_no_protected_split_open(self):
        with diagnostic.guarded.no_model_calls():
            with self.assertRaises(AssertionError):diagnostic.guarded.perception.Observer.load(None)
        with TemporaryDirectory() as temporary,diagnostic.training_only_reads():
            for name in ('dev-questions.jsonl','test-questions.jsonl'):
                with self.assertRaisesRegex(AssertionError,'DEV/TEST'):Path(temporary,name).read_text()
                with self.assertRaisesRegex(AssertionError,'DEV/TEST'):open(Path(temporary,name))
            path=Path(temporary,'train-questions.jsonl');path.write_text('allowed')
            self.assertEqual(path.read_text(),'allowed')

    def test_joint_and_marginal_metrics_are_exact(self):
        y=np.array([0,15]);p=np.ones((2,16))/16
        result=diagnostic.losses(y,p)
        np.testing.assert_allclose(result['joint_log_loss'],np.log(16))
        np.testing.assert_allclose(result['joint_Brier'],15/16)
        np.testing.assert_allclose(result['mean_four_marginal_Brier'],.25)
        perfect=np.eye(16)[y];ideal=diagnostic.losses(y,perfect)
        for values in ideal.values():np.testing.assert_array_equal(values,0.)

    def test_fixed_screen_question_bootstrap_and_no_runtime_claim(self):
        data=numeric_fixture();n=len(data['labels'])
        predictions={diagnostic.VARIANTS[0]:np.full((n,16),1/16),diagnostic.VARIANTS[1]:np.eye(16)[data['labels']]}
        result=diagnostic.diagnostics(data,predictions)
        self.assertTrue(result['screen_passed']);self.assertFalse(result['nested_crossfit'])
        self.assertFalse(result['controller_accuracy_claim']);self.assertFalse(result['fresh_validation'])
        self.assertEqual(result['panels']['all/all']['nodes'],768)
        self.assertEqual(result['panels']['physics/retry']['nodes'],128)
        for metric in result['comparisons']:
            self.assertLessEqual(result['comparisons'][metric]['paired_question_CI95'][1],0)
        same={name:predictions[diagnostic.VARIANTS[0]] for name in diagnostic.VARIANTS}
        self.assertFalse(diagnostic.diagnostics(data,same)['screen_passed'])


if __name__=='__main__':unittest.main()
