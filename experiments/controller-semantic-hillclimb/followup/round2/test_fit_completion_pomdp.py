import copy
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import fit_completion_pomdp as fit
import science_complete_actions as actions


class FittingGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.args = SimpleNamespace(data=root/'data', complete=root/'complete', base=root/'base',
                                    legacy_examples=root/'legacy.jsonl', output=root/'output')
        self.corpus, self.encoder = root/'corpus', root/'encoder'
        for path in (self.args.data/'splits', self.args.complete, self.args.base, self.corpus, self.encoder):
            path.mkdir(parents=True)
        self.rows = [{'id':f'q{i}', 'subject':('math' if i<256 else 'chemistry' if i<384 else 'physics'),
                     'domain':'math' if i<256 else 'science', 'question':'synthetic','answer':'A'} for i in range(512)]
        self.science = self.rows[256:]
        paths = ('','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry')
        node = {'answer':'B','input_tokens':2,'output_tokens':1,'text':'B'}
        self.plain = {r['id']:{p:dict(node) for p in paths} for r in self.rows}
        self.trees = {r['id']:copy.deepcopy(self.plain[r['id']]) for r in self.science}
        for i,row in enumerate(self.science):
            self.trees[row['id']].update({p:{**node,'answer':'A' if i<16 else 'B'}
                                        for p in ('complete','repair/complete','retry/complete')})
        (self.args.data/'splits/train-questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        (self.args.legacy_examples).write_text('{"id":"old-only"}\n')
        for p in (self.args.data/'train-plain-complete.json', self.args.complete/'complete.json'):
            p.write_text('{}')
        (self.args.data/'train-observations.jsonl').write_text('synthetic sealed full bank')
        (self.args.base/'C1-smooth8.pkl').write_bytes(b'synthetic observer placeholder')
        (self.args.base/'training-neural.jsonl').write_text('')
        np.savez(self.args.base/'training-neural.npz', state=np.zeros((0,4)))
        (self.corpus/'passages.jsonl').write_text('synthetic corpus')
        (self.corpus/'tfidf.pkl').write_bytes(b'synthetic retrieval index')
        self.recipe = {'data_sha256':fit.io.sha(self.args.legacy_examples),
                       'corpus_sha256':fit.io.sha(self.corpus/'passages.jsonl')}
        (self.encoder/'recipe.json').write_text(json.dumps(self.recipe))
        self.identity = {'new_train_questions_sha256':fit.io.sha(self.args.data/'splits/train-questions.jsonl'),
            'encoder_trained_on_new_fit_questions':False, 'source_sha256':fit.io.sha(fit.original.__file__),
            'pomdp_sha256':fit.io.sha(fit.policy.plain.__file__),
            'old_only_encoder_inputs_sha256':fit.io.sha(self.args.legacy_examples)}
        self.save_identity()
        self.receipt = {'public_input_evidence':{'files':{}},'source_files':{},'digests':{},
                        'complete_sha256':fit.io.sha(self.args.complete/'complete.json')}
        self.observer = SimpleNamespace(_recipe=self.recipe,directory=str(self.encoder))

    def save_identity(self):
        (self.args.base/'identity.json').write_text(json.dumps(self.identity))

    def load(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(actions,'training_data',return_value=(self.science,self.trees,self.receipt)))
            stack.enter_context(patch.object(fit,'plain_data',return_value=(self.rows,self.plain)))
            stack.enter_context(patch.object(fit.cached,'observer_from',return_value=self.observer))
            stack.enter_context(patch.object(fit.cached,'verify_features',return_value=({},[])))
            stack.enter_context(patch.object(fit.guarded.perception,'CORPUS',str(self.corpus)))
            return fit.load_inputs(self.args)

    def test_binds_complete_grid_and_parent_bank_before_cpu_fit(self):
        rows,trees,observer,lookup,identity = self.load()
        self.assertEqual(len(rows),256)
        self.assertEqual(identity['training_headroom']['novel'],16)
        self.assertIn(str((self.args.complete/'complete.json').resolve()),identity['input_evidence'])
        self.assertIn(str((self.args.data/'train-observations.jsonl').resolve()),identity['input_evidence'])
        self.assertFalse(identity['development_or_test_records_read'])
        self.assertEqual(identity['models'],1)

    def test_fifteen_novel_cases_fail_fixed_training_screen(self):
        for p in ('complete','repair/complete','retry/complete'):
            self.trees['q271'][p]['answer']='B'
        with self.assertRaisesRegex(ValueError,'fewer than16'): self.load()

    def test_changed_plain_control_is_rejected(self):
        self.trees['q256']['repair']['answer']='A'
        with self.assertRaisesRegex(ValueError,'plain control changed'): self.load()

    def test_changed_training_reference_is_rejected(self):
        self.science=copy.deepcopy(self.science); self.science[0]['answer']='B'
        with self.assertRaisesRegex(ValueError,'training reference changed'): self.load()

    def test_encoder_supervised_on_current_questions_is_rejected(self):
        self.identity['encoder_trained_on_new_fit_questions']=True; self.save_identity()
        with self.assertRaisesRegex(ValueError,'unseen-training neural provenance'): self.load()

    def test_parent_receipt_tamper_is_rejected(self):
        extra=self.args.complete/'train.jsonl'; extra.write_text('original')
        self.receipt['digests']['train.jsonl']=fit.io.sha(extra)
        extra.write_text('tampered')
        with self.assertRaisesRegex(ValueError,'evidence/source changed'): self.load()


if __name__ == '__main__': unittest.main()
