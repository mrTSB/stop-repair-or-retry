import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import completion_development_inputs as inputs


def fixture():
    rows=[{'id':f'q{i}','question':'visible question','domain':'math' if i<128 else 'science',
           'subject':'math' if i<128 else ('chemistry' if i<192 else 'physics'),'answer':'A'} for i in range(256)]
    plain={r['id']:{p:{'answer':'B','text':p} for p in inputs.retained.PLAIN} for r in rows}
    strong={qid:{**copy.deepcopy(tree),**{p:{'answer':'B','text':p} for p in inputs.retained.ADDITIONAL}}
            for qid,tree in plain.items()}
    science=[r for r in rows if r['subject']!='math']
    complete={r['id']:{**copy.deepcopy(plain[r['id']]),**{p:{'answer':'A','text':p}
              for p in ('complete','repair/complete','retry/complete')}} for r in science}
    return rows,plain,strong,science,complete


class JoinedDevelopmentTests(unittest.TestCase):
    def test_both_grid_seals_precede_reference_access(self):
        with patch.object(inputs,'strong_data',side_effect=ValueError('unsealed')), \
             patch.object(inputs.complete,'development_data') as completion,patch.object(inputs,'plain_data') as labels:
            with self.assertRaises(ValueError): inputs.load(Path('data'),Path('strong'),Path('complete'))
            completion.assert_not_called();labels.assert_not_called()
        with patch.object(inputs,'strong_data',return_value=([],{}, {},{})), \
             patch.object(inputs.complete,'development_data',side_effect=ValueError('unsealed')),patch.object(inputs,'plain_data') as labels:
            with self.assertRaises(ValueError): inputs.load(Path('data'),Path('strong'),Path('complete'))
            labels.assert_not_called()

    def test_full_join_preserves_plain_nodes_and_all_questions(self):
        rows,plain,strong,science,completed=fixture()
        with patch.object(inputs,'strong_data',return_value=([{**r,'answer':''} for r in rows],strong,{},{})), \
             patch.object(inputs.complete,'development_data',return_value=(science,completed,{})), \
             patch.object(inputs,'plain_data',return_value=(rows,plain)),patch.object(inputs.io,'sha',return_value='panel'):
            restored,trees,_,receipt=inputs.load(Path('data'),Path('strong'),Path('complete'))
        self.assertEqual(restored,rows);self.assertEqual(receipt['science_questions'],128)
        for row in rows:
            for path,node in plain[row['id']].items(): self.assertEqual(trees[row['id']][path],node)
        result=inputs.availability(rows,trees,177)
        self.assertEqual(result['science_panels']['all_science']['Complete_novel_beyond_strong'],128)
        self.assertEqual(result['correct_required_for_10_point_gap'],203)
        self.assertFalse(result['reserved_test_used'])

    def test_changed_reference_or_plain_control_fails_closed(self):
        for kind in ('reference','plain'):
            rows,plain,strong,science,completed=fixture()
            if kind=='reference': science=copy.deepcopy(science);science[0]['answer']='C'
            else: completed[science[0]['id']]['repair']['text']='tampered'
            with patch.object(inputs,'strong_data',return_value=([{**r,'answer':''} for r in rows],strong,{},{})), \
                 patch.object(inputs.complete,'development_data',return_value=(science,completed,{})), \
                 patch.object(inputs,'plain_data',return_value=(rows,plain)):
                with self.assertRaises(ValueError): inputs.load(Path('data'),Path('strong'),Path('complete'))


if __name__=='__main__': unittest.main()
