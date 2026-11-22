import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import auxiliary_second_completion_features as features
import fit_auxiliary_completion as fit
from test_dense_completion import arrays,bundle


class SecondAuxiliaryCompletionTests(unittest.TestCase):
    def test_all_chunk_receipts_resume_without_inference(self):
        data=bundle(64);calls=[]
        def forward(entries):calls.append(len(entries));return arrays(len(entries))
        with TemporaryDirectory() as tmp,patch.object(features,'NODES',64):
            path=Path(tmp);values,digests=features.chunks(path,data,'registration',forward,lambda:None)
            self.assertEqual(calls,[32,32]);self.assertEqual(len(digests),4)
            def forbidden(*args):self.fail('Repeated inference')
            replay,seals=features.chunks(path,data,'registration',forbidden,forbidden)
            self.assertEqual(seals,digests)
            for key in values:np.testing.assert_array_equal(values[key],replay[key])
            changed=copy.deepcopy(data);changed['entries'][0]['left']='wrong paid input'
            with self.assertRaises(ValueError):features.chunks(path,changed,'registration',forbidden,forbidden)

    def test_unresolved_neural_attempt_fails_closed(self):
        with TemporaryDirectory() as tmp,patch.object(features,'NODES',64):
            path=Path(tmp)
            def interrupted(*args):raise RuntimeError('lost forward')
            with self.assertRaises(RuntimeError):features.chunks(path,bundle(64),'r',interrupted,lambda:None)
            self.assertEqual(len(features.pairs.pilot.protocol.strict_records(path/'attempts.jsonl')),1)
            with self.assertRaisesRegex(ValueError,'Unresolved'):features.chunks(path,bundle(64),'r',interrupted,lambda:None)

    def test_only_selected_train_references_are_decoded(self):
        selected={'uuid':'selected','field':'Physics','question':'Which one?','options':['one','two','three','four'],
            'answer_letter':'B','answer':'two'}
        protected={**selected,'uuid':'protected','answer_letter':'PROTECTED_SECRET','answer':'PROTECTED_SECRET'}
        public=[{'id':'supergpqa-selected','subject':'physics','answer':'',
            'question':features.pairs.contract.auxiliary.question(selected)}]
        original=json.loads
        def checked(raw,*a,**kw):
            if isinstance(raw,(bytes,str)) and ('PROTECTED_SECRET' in raw if isinstance(raw,str) else b'PROTECTED_SECRET' in raw):
                self.fail('Decoded an unselected source reference')
            return original(raw,*a,**kw)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.jsonl';features.io.jsonl(path,[protected,selected])
            with patch.object(json,'loads',side_effect=checked),features.fixed.training_only_reads():
                labeled=fit.reference_join(public,path)
            self.assertEqual(labeled[0]['answer'],'B');self.assertEqual(public[0]['answer'],'')

    def test_expanded_table_keeps_reference_out_of_features_and_groups_prefixes(self):
        rows=[{'id':f'q{i}','subject':'chemistry' if i%2 else 'physics','answer':'A'} for i in range(8)]
        trees={r['id']:{p:{'answer':'A' if p.endswith('complete') else 'B','text':p,'prompt_sha256':p}
            for p in features.pairs.contract.PATHS} for r in rows}
        metadata=[{'id':r['id'],'path':p,'prompt_sha256':p} for r in rows for p in ('','repair','retry')]
        values=arrays(24)
        def observable(public,path,paid,neural):
            self.assertEqual(public['answer'],'');self.assertEqual(set(paid),{'',path});return np.zeros(130)
        with patch.object(fit.runtime.core.plain,'observables',side_effect=observable):
            data=fit.table(rows,trees,metadata,values)
            changed=copy.deepcopy(trees);changed['q0']['repair/complete']['answer']='B'
            other=fit.table(rows,changed,metadata,values)
            self.assertNotEqual(data['labels'][1],other['labels'][1])
            for key in data['features']:np.testing.assert_array_equal(data['features'][key],other['features'][key])
            with self.assertRaises(ValueError):fit.table(rows,trees,metadata[::-1],values)
        for i in range(8):self.assertEqual(len({r['fold'] for r in data['records'][i*3:i*3+3]}),1)


if __name__=='__main__':unittest.main()
