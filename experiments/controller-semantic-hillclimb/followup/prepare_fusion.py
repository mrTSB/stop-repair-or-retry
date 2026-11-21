"""Compose existing observer weights and aligned excluded-question predictions locally."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    sources = [('base', args.base), ('large', args.large)]
    contracts = [json.loads((p / 'contract.json').read_bytes()) for _, p in sources]
    if len({r['data_sha256'] for r in contracts}) != 1:
        raise ValueError('Fusion observations must have the exact same training examples and ordering')
    args.output.mkdir(parents=True, exist_ok=False)
    full = args.output / 'full'
    full.mkdir()
    components, weights, values, temperatures = [], {}, [], []
    parameters = 0
    tokenizers = set()
    corpus = set()
    for name, source in sources:
        original = source / 'full'
        recipe = json.loads((original / 'recipe.json').read_bytes())
        if recipe['max_length'] != 512 or recipe.get('fold') != 'full':
            raise ValueError('Unexpected component checkpoint')
        tokenizers.add(sha(original / 'spm.model'))
        corpus.add(recipe['corpus_sha256'])
        target = full / name
        target.mkdir()
        for file in original.iterdir():
            if file.is_file():
                os.link(file, target / file.name)
        parameters += recipe['parameters']
        summary = json.loads((source / 'training-summary.json').read_bytes())
        temperature = summary['temperature_selected_on_training_oof']
        components.append({'directory': name, 'parameters': recipe['parameters'], 'temperature': temperature,
                           'source_recipe_sha256': sha(original / 'recipe.json'),
                           'private_volume': 'srr-semantic-training-20261120',
                           'remote': '/' + source.name + '/full'})
        temperatures.append(temperature)
        values.append(np.load(source / 'oof.npz'))
        weights.update({name + '/' + key: value for key, value in recipe['weights'].items()})
    if len(tokenizers) != 1 or len(corpus) != 1:
        raise ValueError('Different tokenization or reference corpus')
    if values[0]['state'].shape != values[1]['state'].shape:
        raise ValueError('Misaligned out-of-question predictions')
    recipe = {'kind': 'parallel-observer-fusion-v1', 'component_count': 2,
              'components': components, 'parameters': parameters, 'max_length': 512,
              'input_profile': 'full-options-openstax-v1', 'corpus_sha256': next(iter(corpus)),
              'weights': weights, 'tokenizer_sha256': next(iter(tokenizers))}
    (full / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    np.savez_compressed(args.output / 'oof.npz',
        state=np.concatenate([v['state'] / t for v, t in zip(values, temperatures)], axis=1),
        plain=np.mean([v['plain'] for v in values], axis=0),
        guided=np.mean([v['guided'] for v in values], axis=0))
    contract = {'data_sha256': contracts[0]['data_sha256'], 'folds': 3,
                'kind': recipe['kind'], 'neural_training_performed': False,
                'source_sha256': sha(__file__),
                'components': {name: {'contract_sha256': sha(path / 'contract.json'),
                                      'oof_sha256': sha(path / 'oof.npz')}
                               for name, path in sources}}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    summary = {'temperature_selected_on_training_oof': 1., 'fresh_data_used': False,
               'development_data_used': False, 'neural_training_performed': False,
               'observation': 'Joint calibration sees concatenated independently question-excluded logits',
               'parameters': parameters, 'physical_components': 2}
    (args.output / 'training-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'large', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
