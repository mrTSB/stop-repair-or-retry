import copy
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
        prior={'key':'stronger-previous','dev':{'accuracy':1.,'mean_compute':1.,'correct':3,'n':3}}
        baselines={f'b{i}':stats(old_items) for i in range(5)};matched={f'm{i}':{'retained':True} for i in range(6)}
        with TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output';output.mkdir();models=root/'models';models.mkdir();parent_path=root/'parent';parent_path.mkdir()
            run.io.write(parent_path/'private-candidate-items.json',{'old':old_items})
            for f in run.nested.FAMILIES:(models/(f+'.pkl')).write_bytes(b'training-model')
            args=SimpleNamespace(output=output,models=models,registration_sha256='registered')
            record={'previous_A_candidates':[prior],'dev_questions_sha256':'panel','source_files':{},'input_evidence':{}}
            fitted={f:SimpleNamespace(classifier=None) for f in run.nested.FAMILIES}
            with patch.object(run.routed,'evaluate',side_effect=lambda *a:copy.deepcopy(old_items[0])), \
                    patch.object(run.runtime,'evaluate',return_value={'correct':False,'compute':3.}), \
                    patch.object(run.original,'statistics',side_effect=stats), \
                    patch.object(run.previous,'controls',return_value=(baselines,matched)) as controls:
                run.replay(args,data,fitted,arrays(6),parent_path,record)
            result=run.io.read(output/'summary.json')
            self.assertEqual(result['selected_A'],prior);self.assertEqual(len(result['candidates']),16)
            self.assertEqual(result['baselines'],baselines);self.assertEqual(result['matched_complete_controls'],matched)
            controls.assert_called_once();self.assertFalse(result['test_used']);self.assertEqual(result['new_neural_calls'],0)
            items=run.io.read(output/'private-candidate-items.json')
            self.assertTrue(all(v[0]==old_items[0] for v in items.values()))
            with self.assertRaises(ValueError):run.previous.preserve_partial(output)


if __name__=='__main__':unittest.main()
