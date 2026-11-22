import copy
import importlib
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import completion_subject_pomdp as routed
import development_completion_pomdp as dev


def node(answer, tokens=100):
    return {'answer':answer,'text':answer,'input_tokens':tokens,'output_tokens':1}


class DevelopmentTests(unittest.TestCase):
    def test_collector_exact_field_names_and_model_binding(self):
        import development_complete_actions as actions
        collection={'schema':actions.SCHEMA,'evaluation_penalties':list(actions.PENALTIES),
                    'model_sha256':'model','paths':{'model':'/tmp/model'},'source_files':{},'test_used':False}
        complete={'schema':actions.SCHEMA,'registration_sha256':'registered','test_used':False,
                  'coverage':{'expected_cells':384,'physical_calls':384,'logical_actions':384,
                              'reused_physical_calls':0,'new_physical_calls':384}}
        dev.validate_collection_binding(collection,complete,{'model_sha256':'model'},Path('/tmp/model'),'registered')
        changed=copy.deepcopy(collection);changed['evaluation_penalties'].append(.3)
        with self.assertRaises(ValueError):
            dev.validate_collection_binding(changed,complete,{'model_sha256':'model'},Path('/tmp/model'),'registered')
        with self.assertRaises(ValueError):
            dev.validate_collection_binding(collection,complete,{'model_sha256':'different'},Path('/tmp/model'),'registered')

    def test_matched_controls_pay_full_prefix_and_only_use_validity(self):
        row={'id':'x','subject':'physics','domain':'science','answer':'A','question':'Q\n\nOptions:\nA. yes\nB. no\nC. maybe\nD. none'}
        tree={'':node('B',100),'repair':node('',200),'repair/complete':node('A',300),'complete':node('A',400)}
        item=dev.matched_item(row,tree,'repair','valid')
        self.assertEqual(item['path'],'repair/complete')
        self.assertEqual(item['solver_tokens'],603)
        self.assertEqual(item['solver_calls'],3)
        self.assertEqual(item['verifier_calls'],0)
        self.assertEqual(dev.matched_item(row,tree,'','valid')['path'],'')
        self.assertEqual(dev.matched_item(row,tree,'','valid')['correct'],0)
        self.assertEqual(dev.matched_item(row,tree,'','')['path'],'complete')

    def test_subject_route_executes_exactly_one_policy(self):
        model=routed.Planner('math-object','science-object',{'penalty':0.,'posterior_weight':.5},{'penalty':.02},{})
        with patch.object(routed.continuous_repairability,'evaluate',return_value={'math':True}) as math, \
             patch.object(routed.completion_pomdp,'evaluate',return_value={'science':True}) as science:
            self.assertEqual(routed.evaluate({'subject':'math'},{},{},model,{},'A'),{'math':True})
            self.assertEqual(routed.evaluate({'subject':'physics'},{},{},model,{},'A'),{'science':True})
            self.assertEqual(math.call_count,1);self.assertEqual(science.call_count,1)
            self.assertEqual(math.call_args.args[2],{});self.assertEqual(science.call_args.args[2],{})
        with self.assertRaises(ValueError):routed.evaluate({'subject':'math'},{},{},model,{},'C')

    def test_full_panel_controls_and_stronger_prior_retention(self):
        # A synthetic integration fixture: these scores are not project results.
        import development_complete_actions as actions
        with TemporaryDirectory() as temporary:
            root=Path(temporary); output=root/'out';output.mkdir();models=root/'models';models.mkdir()
            encoder=root/'encoder';encoder.mkdir();(encoder/'recipe.json').write_text('{}')
            (output/'registration.json').write_text('{}')
            science_model=SimpleNamespace(classifier=SimpleNamespace(directory=str(encoder)))
            (models/'completion-A.pkl').write_bytes(pickle.dumps(science_model))
            math_model=SimpleNamespace(classifier=SimpleNamespace(directory=str(encoder)))
            math_path=root/'math.pkl';math_path.write_bytes(pickle.dumps(math_model))
            rows=[{'id':str(i),'subject':'math' if i<128 else 'chemistry' if i<192 else 'physics',
                   'domain':'math' if i<128 else 'science','answer':'A',
                   'question':'Q\n\nOptions:\nA. yes\nB. no\nC. maybe\nD. none'} for i in range(256)]
            paths=('','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry')
            plain={r['id']:{p:node('A' if r['subject']=='math' else 'B') for p in paths} for r in rows}
            science_rows=rows[128:]; science_trees={r['id']:copy.deepcopy(plain[r['id']]) for r in science_rows}
            for row in science_rows:
                science_trees[row['id']].update({p:node('A' if int(row['id'])%2==0 else 'B',200)
                    for p in ('complete','repair/complete','retry/complete')})
            old_summary=root/'old-summary.json';old_summary.write_text('{}')
            old={'candidate':{'dev':{'n':256,'correct':195,'accuracy':195/256,'mean_compute':200.},'settings':{}},
                 'model_path':str(math_path),'module':'repairability_pomdp','summary':str(old_summary)}
            identity={'base_selection':old,'fitted_model_sha256':'synthetic','dev_questions_sha256':'panel',
                      'matched_full_panel_math':'same retained math A','source_files':{},'input_evidence':{}}
            args=SimpleNamespace(data=root/'data',complete=root/'complete',models=models,output=output,
                                 registry=root,features=root/'features')
            visited=[]
            def math_evaluate(row,tree,*unused):
                out=dev.core.plain.outcome(row,tree,{},'',[''],[],[])
                out.update(verifier_tokens=10,verifier_token_slots=10,verifier_calls=1)
                out['compute']+=1;out['total_parameter_compute']+=1
                return out
            def science_evaluate(row,tree,judges,model,setting,family):
                visited.append(row['id'])
                penalty=setting.get('by_subject',{}).get(row['subject'],setting)['penalty']
                return dev.matched_item(row,tree,'','' if penalty==0 else 'valid')
            with patch.object(actions,'development_data',return_value=(science_rows,science_trees,{})), \
                 patch.object(dev,'plain_data',return_value=(rows,plain)), \
                 patch.object(dev.base_policy,'resolve_base',return_value=(math_model,{'penalty':0.})), \
                 patch.object(dev.base_policy,'asset_id',return_value='same-asset'), \
                 patch.object(dev,'cached_observer',return_value=SimpleNamespace(directory=str(encoder))), \
                 patch.object(dev.core,'evaluate',side_effect=science_evaluate), \
                 patch.object(dev.core.plain,'evaluate',side_effect=math_evaluate), \
                 patch.object(dev.reference,'references',return_value=(old,[],[])):
                dev.evaluate(args,identity)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(len(visited),8*128)
            self.assertEqual(set(visited),{r['id'] for r in science_rows})
            self.assertEqual(len(summary['baselines']),5)
            self.assertEqual(len(summary['matched_complete_controls']),6)
            self.assertEqual(summary['completion_A']['dev']['correct'],192)
            self.assertEqual(summary['selected_A']['dev']['correct'],195)
            self.assertEqual(summary['selected_A']['source'],'previous-A')
            self.assertEqual(summary['selected_A']['source_summary'],str(old_summary))
            self.assertEqual(summary['completion_A']['policy_module'],'completion_subject_pomdp')
            self.assertEqual(summary['baseline_science_panels']['Stop']['all']['n'],128)
            self.assertEqual(summary['completion_A_above_best_matched_control'],0.)
            for control in summary['matched_complete_controls'].values():
                self.assertEqual(control['full256_with_retained_math_A']['n'],256)
                self.assertEqual(control['science']['all']['n'],128)
                self.assertEqual(control['science']['chemistry']['n'],64)
                self.assertEqual(control['science']['physics']['n'],64)
            self.assertFalse(summary['test_used']);self.assertFalse(summary['fresh_validation'])
            deployed=pickle.loads((output/'completion-subject-A.pkl').read_bytes())
            self.assertFalse(hasattr(deployed.science.classifier,'saved'))
            self.assertFalse(hasattr(deployed.math.classifier,'saved'))


if __name__=='__main__':unittest.main()
