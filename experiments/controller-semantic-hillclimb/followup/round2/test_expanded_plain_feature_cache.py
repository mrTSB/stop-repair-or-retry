"""Synthetic complete-cache conversion, exact ancestors and zero-forward tests."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import expanded_plain_feature_cache as conversion


class Observer:
    def __init__(self,directory,corpus_sha):
        self.directory=str(directory);self._model=None;self._recipe={};self._retriever=SimpleNamespace(sha256=corpus_sha)
    def entry(self,row,path,tree):
        if row['answer']:raise AssertionError('Reference leaked')
        return {'question':row['question'],'root':tree['']['text'],'current':tree[path]['text'],'path':path}
    def key(self,row,path,node,kind):return kind,row['id'],path,node['prompt_sha256'],node['text']
    def load(self):raise AssertionError('No loading')
    def values(self,*args):raise AssertionError('No forward')


def fixture(root):
    source=root/'features';source.mkdir();model=root/'model';model.mkdir();corpus=root/'corpus';corpus.mkdir()
    (corpus/'passages.jsonl').write_bytes(b'synthetic-public-evidence')
    recipe={'max_length':512,'corpus_sha256':conversion.io.sha(corpus/'passages.jsonl')}
    conversion.io.write(model/'recipe.json',recipe)
    observer=Observer(model,recipe['corpus_sha256'])
    rows=[];trees={};metadata=[];inherited={}
    arrays={name:np.arange(768*width,dtype=np.float32).reshape(768,width)/100 for name,width in [('state',4),('plain',8),('guided',8)]}
    for i in range(256):
        row={'id':f'q{i}','question':'Public question '+str(i),'answer':'PrivateReference'};rows.append(row)
        tree={p:{'text':'Paid '+p+str(i),'prompt_sha256':hashlib.sha256((str(i)+p).encode()).hexdigest()} for p in ('','repair','retry')}
        trees[row['id']]=tree
        for path in ('','repair','retry'):
            index=len(metadata);entry=observer.entry({**row,'answer':''},path,{p:tree[p] for p in ('',path)})
            metadata.append({'id':row['id'],'path':path,'token_slots':32,'prompt_sha256':tree[path]['prompt_sha256'],
                'input_sha256':conversion.original.fingerprint(entry)})
            if index<12:inherited[observer.key(row,path,tree[path],'solver')]=(*[arrays[n][index] for n in ('state','plain','guided')],32)
    conversion.io.npz(source/'neural.npz',arrays);conversion.io.jsonl(source/'metadata.jsonl',metadata)
    completion={'source_sha256':conversion.io.sha(conversion.original.__file__),'test_used':False,'nodes':768,
        'dev_questions_sha256':'panel','encoder_recipe_sha256':conversion.io.sha(model/'recipe.json'),
        'neural_sha256':conversion.io.sha(source/'neural.npz'),'metadata_sha256':conversion.io.sha(source/'metadata.jsonl')}
    conversion.io.write(source/'complete.json',completion)
    return source,observer,recipe,rows,trees,inherited,corpus


class FeatureConversionTests(unittest.TestCase):
    def test_whole768_conversion_keeps_values_slots_and_blanks_reference(self):
        with TemporaryDirectory() as temporary:
            source,observer,recipe,rows,trees,old,corpus=fixture(Path(temporary))
            with patch.object(conversion.original.perception,'CORPUS',corpus):
                result,receipt=conversion.convert(source,observer,recipe,rows,trees,old,'panel','plain')
                changed=[{**r,'answer':'ChangedReference'} for r in rows]
                other,other_receipt=conversion.convert(source,observer,recipe,changed,trees,old,'panel','plain')
            self.assertEqual(len(result),768);self.assertEqual(receipt['bitwise_identical_overlap'],12)
            self.assertEqual(receipt['overlap_with_shared_cache'],12);self.assertEqual(receipt['max_absolute_logit_difference'],0.)
            self.assertEqual(receipt,other_receipt);self.assertEqual(observer._recipe,{})
            self.assertEqual(receipt['new_encoder_calls'],0);self.assertIsNone(observer._model)
            for key,values in result.items():
                for a,b in zip(values,other[key]):np.testing.assert_array_equal(a,b)

    def test_changed_ancestor_rejected_even_with_unchanged_child_prompt(self):
        with TemporaryDirectory() as temporary:
            source,observer,recipe,rows,trees,old,corpus=fixture(Path(temporary))
            trees['q0']['']['text']='Changed ancestor'
            with patch.object(conversion.original.perception,'CORPUS',corpus):
                with self.assertRaisesRegex(ValueError,'exact paid question/response'):
                    conversion.convert(source,observer,recipe,rows,trees,old,'panel','plain')

    def test_changed_array_hash_and_slot_disagreement_fail_closed(self):
        with TemporaryDirectory() as temporary:
            source,observer,recipe,rows,trees,old,corpus=fixture(Path(temporary))
            key=next(iter(old));old[key]=(*old[key][:3],33)
            with patch.object(conversion.original.perception,'CORPUS',corpus):
                with self.assertRaisesRegex(ValueError,'physical token slots'):
                    conversion.convert(source,observer,recipe,rows,trees,old,'panel','plain')
                (source/'neural.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Registered evidence'):
                    conversion.convert(source,observer,recipe,rows,trees,{},'panel','plain')

    def test_overlap_numeric_difference_is_disclosed_without_per_row_selection(self):
        with TemporaryDirectory() as temporary:
            source,observer,recipe,rows,trees,old,corpus=fixture(Path(temporary))
            key=next(iter(old));values=old[key];original=values[0].copy();old[key]=(values[0]+1.,*values[1:])
            with patch.object(conversion.original.perception,'CORPUS',corpus):
                converted,receipt=conversion.convert(source,observer,recipe,rows,trees,old,'panel','plain')
            np.testing.assert_array_equal(converted[key][0],original)
            self.assertEqual(receipt['bitwise_identical_overlap'],11)
            self.assertAlmostEqual(receipt['max_absolute_logit_difference'],1.,places=6)


if __name__=='__main__':unittest.main()
