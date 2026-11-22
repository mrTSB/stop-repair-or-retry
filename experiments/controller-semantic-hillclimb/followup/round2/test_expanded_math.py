from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import expanded_math_diagnostic as study


class ExpandedMathTests(unittest.TestCase):
    def test_held_question_labels_cannot_affect_forecasts_or_selection(self):
        rng = np.random.default_rng(238);n=180
        data = {'records':[{'id':str(i//3),'path':('','repair','retry')[i%3],'subject':'math','fold':i//3%3} for i in range(n)],
            'x':rng.normal(size=(n,6)),'y':np.arange(n)%8,'valid':np.ones(n,dtype=bool)}
        with patch.object(study.math,'SETTINGS',({'kind':'linear','C':.01},)):
            held,p,_,receipt = study.outer(data,1)
            changed = {**data,'y':data['y'].copy()};changed['y'][held]=7-changed['y'][held]
            held2,q,_,receipt2 = study.outer(changed,1)
        np.testing.assert_array_equal(held,held2);np.testing.assert_array_equal(p,q);self.assertEqual(receipt,receipt2)

    def test_reference_join_uses_only_selected_original_indices(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.arrow'
            table = pa.table({'question':['unselected0','selected1','unselected2','selected3'],
                              'answer':['MALFORMED_UNSELECTED','Reasoning #### 8',None,'Reasoning #### 12']})
            with pa.OSFile(str(source),'wb') as sink:
                with pa.ipc.new_stream(sink,table.schema) as writer:writer.write_table(table)
            public = [{'id':'gsm8k-train-00003','question':'selected3','answer':''},
                      {'id':'gsm8k-train-00001','question':'selected1','answer':''}]
            result = study.reference_join(public,source)
            self.assertEqual([r['answer'] for r in result],['12','8'])
            with self.assertRaises(ValueError): study.reference_join([{**public[0],'question':'Changed'}],source)
            with self.assertRaises(ValueError): study.reference_join([{**public[0],'answer':'12'}],source)


if __name__=='__main__':unittest.main()
