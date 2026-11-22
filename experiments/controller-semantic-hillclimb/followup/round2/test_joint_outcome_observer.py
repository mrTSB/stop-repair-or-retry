import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import DebertaV2Config,AutoModel

import train_joint_outcome_observer as run
import joint_outcome_observer as network


def encoder():
    return AutoModel.from_config(DebertaV2Config(vocab_size=24,hidden_size=8,num_hidden_layers=4,
        num_attention_heads=2,intermediate_size=12,max_position_embeddings=32))


class JointObserverTests(unittest.TestCase):
    def test_invalid_support_loss_and_frozen_initial_layers(self):
        old=network.initial.Network(encoder(),3);prior=np.ones(16)/16
        with patch.object(network.initial,'load',return_value=(old,None,{})):
            model,_,_=network.from_initial('synthetic',prior)
        self.assertFalse(model.encoder.embeddings.word_embeddings.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.encoder.encoder.layer[-1].parameters()))
        logits=model(input_ids=torch.ones((2,5),dtype=torch.long),attention_mask=torch.ones((2,5)),numeric=torch.zeros((2,3)))
        loss=network.loss(logits,torch.tensor([1,9]),torch.tensor([False,True]));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(model.joint.weight.grad).all())
        raw=logits.detach().numpy();before=raw.copy();p=network.probabilities(raw,[False,True],prior)
        np.testing.assert_array_equal(raw,before);np.testing.assert_array_equal(p[0,8:],0.)
        np.testing.assert_allclose(p.sum(1),1.)
        with self.assertRaises(ValueError):network.loss(logits,torch.tensor([8,9]),torch.tensor([False,True]))

    def test_archived_half_rounded_model_reloads_with_same_architecture(self):
        class Tokenizer:
            def save_pretrained(self,path):(Path(path)/'synthetic-tokenizer.json').write_text('{}')
        with TemporaryDirectory() as tmp:
            path=Path(tmp);model=network.Network(encoder(),3).eval();tokenizer=Tokenizer()
            values={'input_ids':torch.ones((2,5),dtype=torch.long),'attention_mask':torch.ones((2,5)), 'numeric':torch.zeros((2,3))}
            before=model(**values).detach().numpy();recipe=network.save(model,tokenizer,path,{})
            with patch.object(network.initial.AutoTokenizer,'from_pretrained',return_value=tokenizer):restored,_,_=network.load(path)
            np.testing.assert_allclose(restored(**values).detach().numpy(),before,atol=1e-3)
            self.assertEqual(recipe['parameters'],sum(p.numel() for p in restored.parameters()))

    def test_question_folds_and_costs_exclude_held_outcomes(self):
        rows=[{'id':f'q{i}','subject':'chemistry' if i%2 else 'physics','answer':'A'} for i in range(30)]
        paths=('', 'repair','retry','complete','repair/repair','repair/retry','repair/complete','retry/repair','retry/retry','retry/complete')
        trees={r['id']:{p:{'answer':'A','input_tokens':100,'output_tokens':5} for p in paths} for r in rows}
        records=[{'id':r['id'],'path':p,'fold':run.fixed.refined.fold_for(r['id'])} for r in rows for p in ('','repair','retry')]
        train,held,_=run.fold_masks(records,0)
        self.assertFalse({r['id'] for r,v in zip(records,train) if v}&{r['id'] for r,v in zip(records,held) if v})
        original=run.conditional_costs(rows,trees,0);changed=copy.deepcopy(trees)
        for row in rows:
            if run.fixed.refined.fold_for(row['id'])==0:
                for node in changed[row['id']].values():node.update(answer='B',input_tokens=10**9)
        after=run.conditional_costs(rows,changed,0)
        for subject in original:np.testing.assert_array_equal(original[subject],after[subject])


if __name__=='__main__':unittest.main()
