import copy
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import development_expanded1280_completion as run
from test_dense_completion import arrays


class Expanded1280DevelopmentTests(unittest.TestCase):
    def test_synthetic_full_replay_preserves_stronger_previous_and_all_controls(self):
        rows=[{'id':s,'subject':s} for s in ('math','chemistry','physics')]
        old_items=[{'correct':True,'compute':2.}]*3
        def stats(items):return {'n':len(items),'correct':sum(v['correct'] for v in items),
            'accuracy':sum(v['correct'] for v in items)/len(items),'mean_compute':sum(v['compute'] for v in items)/len(items)}
        observer=SimpleNamespace(directory='synthetic',temperature=1.,_recipe={},_retriever=None,_backgrounds={})
        parent=run.routed.Planner(SimpleNamespace(classifier=None),SimpleNamespace(classifier=None),{}, {},{})
        metadata=[{'id':r['id'],'path':p,'input_sha256':'paid'} for r in rows if r['subject']!='math' for p in ('','repair','retry')]
        summary={'completion_A':{'key':'old'},'math_metrics':stats(old_items[:1])}
        data=(rows,{r['id']:{} for r in rows},parent,observer,summary,{}, {'metadata':metadata})
        prior={'key':'stronger-previous','policy_module':'continuous_completion_pomdp','dev':{'accuracy':1.,'mean_compute':1.,'correct':3,'n':3}}
        retained_items=[{'correct':True,'compute':1.},*old_items[1:]]
        baselines={f'b{i}':stats(old_items) for i in range(5)}
        science={s:stats(old_items[1:] if s=='all' else old_items[1:2]) for s in ('all','chemistry','physics')}
        matched={f'm{i}':{'full256_with_retained_math_A':stats(old_items),'science':science} for i in range(6)}
        new_matched={f'm{i}':{'full256_with_retained_math_A':stats(retained_items),'science':science} for i in range(6)}
        with TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output';output.mkdir();models=root/'models';models.mkdir();parent_path=root/'parent';parent_path.mkdir()
            run.io.write(parent_path/'private-candidate-items.json',{'old':old_items})
            for f in run.nested.FAMILIES:(models/(f+'.pkl')).write_bytes(b'training-model')
            retained=root/'retained';retained.mkdir();model_path=retained/'model.pkl';model_path.write_bytes(pickle.dumps(parent))
            selected={**prior,'model_path':str(model_path),'model_sha256':run.io.sha(model_path)}
            run.io.write(retained/'summary.json',{'selected_A':selected,'math_metrics':stats(retained_items[:1]),'matched_complete_controls':new_matched})
            run.io.write(retained/'private-candidate-items.json',{prior['key']:retained_items})
            args=SimpleNamespace(output=output,models=models,retain_a=retained,registration_sha256='registered')
            record={'previous_A_candidates':[prior],'dev_questions_sha256':'panel','source_files':{},'input_evidence':{}}
            fitted={f:SimpleNamespace(classifier=None) for f in run.nested.FAMILIES}
            with patch.object(run.routed,'evaluate',side_effect=lambda *a:copy.deepcopy(old_items[0])), \
                    patch.object(run.variants,'wrap',side_effect=copy.deepcopy), \
                    patch.object(run.variants,'evaluate_science',return_value={'correct':False,'compute':3.}), \
                    patch.object(run.retained_math.policy,'evaluate',side_effect=lambda r,*a:copy.deepcopy(retained_items[0 if r['subject']=='math' else 1])), \
                    patch.object(run.original,'MATCHED',[(f'm{i}','',False) for i in range(6)]), \
                    patch.object(run.original,'matched_item',return_value=old_items[1]), \
                    patch.object(run.original,'statistics',side_effect=stats), \
                    patch.object(run.previous,'controls',return_value=(baselines,matched)) as controls:
                run.replay(args,data,fitted,arrays(6),parent_path,record)
            result=run.io.read(output/'summary.json')
            self.assertEqual(result['selected_A'],prior);self.assertEqual(len(result['candidates']),32)
            self.assertEqual(result['baselines'],baselines);self.assertEqual(result['matched_complete_controls'],new_matched)
            self.assertEqual(result['previous_matched_complete_controls'],matched)
            controls.assert_called_once();self.assertFalse(result['test_used']);self.assertEqual(result['new_neural_calls'],0)
            items=run.io.read(output/'private-candidate-items.json')
            self.assertTrue(all(v[0]==retained_items[0] for v in items.values()))
            with self.assertRaises(ValueError):run.previous.preserve_partial(output)


if __name__=='__main__':unittest.main()
