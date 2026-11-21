"""Replay a new-source hybrid using whole completed response and neural banks.

The matched LLM control supplies responses, while a compatible completed C
supplies raw neural observations. No source record is selected by correctness.
Only the unchanged development evaluator may acquire a genuinely missing node.
"""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

import development_mixed_mentor as study
import matched_review_b as control
from prepare_mixed_mentor import sha

SCHEMA = 'new-source-development-replay-v1'


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def wait(path):
    while not path.exists():
        time.sleep(30)
    return read(path)


def wait_completed(directory):
    wait(directory / 'summary.json')
    imported = directory / 'cache-import.json'
    if imported.exists() and read(imported).get('schema') == SCHEMA:
        wait(directory / 'replay-complete.json')


def completed_bank(directory, panel):
    summary, identity = read(directory / 'summary.json'), read(directory / 'identity.json')
    if (summary.get('test_used') is not False or identity.get('test_used') is not False
            or summary.get('development_questions') != 256
            or identity.get('dev_questions_sha256') != panel):
        raise ValueError('Require a complete same-panel development source')
    source = directory / 'dev-observations.jsonl'
    with source.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        payload = stream.read()
    if (payload and not payload.endswith(b'\n')) or hashlib.sha256(payload).hexdigest() != summary['bank_sha256']:
        raise ValueError('Incomplete or changed development response bank')
    imported = directory / 'cache-import.json'
    if imported.exists() and read(imported).get('schema') == SCHEMA:
        marker = directory / 'replay-complete.json'
        expected = {'schema': SCHEMA, 'source_sha256': sha(__file__),
                    'summary_sha256': sha(directory / 'summary.json'),
                    'identity_sha256': sha(directory / 'identity.json'),
                    'import_sha256': sha(imported)}
        if not marker.exists() or read(marker) != expected:
            raise ValueError('New-source replay has not sealed its final summary')
    return payload, {'directory': str(directory.resolve()),
                     'summary_sha256': sha(directory / 'summary.json'),
                     'identity_sha256': sha(directory / 'identity.json'),
                     'bank_sha256': summary['bank_sha256']}


def model_assets(directory):
    complete = read(directory / 'complete.json')
    if complete['identity_sha256'] != sha(directory / 'identity.json'):
        raise ValueError('Changed fitting identity')
    recipes = set()
    for spec in complete['models']:
        path = directory / (spec['key'] + '.pkl')
        if sha(path) != spec['sha256']:
            raise ValueError('Changed fitted model')
        model = pickle.loads(path.read_bytes())
        if model.base.classifier._model is not None or model.mentor.classifier._model is not None:
            raise ValueError('Require archived models without resident inference state')
        a = sha(Path(model.base.classifier.directory) / 'recipe.json')
        b = sha(Path(model.mentor.classifier.directory) / 'recipe.json')
        if a != b or a != model.fit_receipt['observer_recipe_sha256']:
            raise ValueError('Model observation profiles differ')
        recipes.add(a)
    if len(complete['models']) != 4 or len(recipes) != 1:
        raise ValueError('Require all four variants with the same raw neural observer')
    return complete, next(iter(recipes))


def cache_values(payload):
    try:
        values = pickle.loads(payload)
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, TypeError) as error:
        raise ValueError('Unreadable raw neural cache') from error
    if not isinstance(values, dict):
        raise ValueError('Malformed raw neural cache')
    for key, value in values.items():
        if (not isinstance(key, tuple) or len(key) != 5 or key[0] not in ('solver', 'review')
                or not all(isinstance(part, str) for part in key)
                or len(key[3]) != 64 or any(c not in '0123456789abcdef' for c in key[3])):
            raise ValueError('Malformed old-observer cache key')
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            raise ValueError('Malformed raw forward cache value')
        for actual, width in zip(value[:3], (4, 8, 8)):
            array = np.asarray(actual)
            if array.shape != (width,) or not np.isfinite(array).all():
                raise ValueError('Nonfinite or incorrectly shaped raw neural output')
        if not isinstance(value[3], (int, np.integer)) or not 1 <= value[3] <= 512:
            raise ValueError('Invalid old-observer input token count')
    return values


def copy_prefix(output, responses, cache, receipt):
    """Preserve any collected suffix on resume; never overwrite a response."""
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / 'cache-import.json'
    bank = output / 'dev-observations.jsonl'
    immutable_cache = output / 'imported-neural-cache.pkl'
    mutable_cache = output / 'private-neural-cache.pkl'
    source_values = cache_values(cache)
    if provenance.exists():
        if read(provenance) != receipt:
            raise ValueError('Changed replay input identity')
        current = bank.read_bytes() if bank.exists() else None
        if (current is None or (current and not current.endswith(b'\n')) or not current.startswith(responses)
                or not immutable_cache.exists()
                or sha(immutable_cache) != receipt['neural_cache_sha256']):
            raise ValueError('Imported development prefix or immutable cache changed')
        if not mutable_cache.exists():
            shutil.copyfile(immutable_cache, mutable_cache)
        resumed = cache_values(mutable_cache.read_bytes())
        if any(key not in resumed or any(not np.array_equal(x, y) for x, y in zip(value, resumed[key]))
               for key, value in source_values.items()):
            raise ValueError('Resumed cache changed or removed an imported forward observation')
        return
    if any(path.name != '.replay.lock' for path in output.iterdir()):
        raise ValueError('Unregistered nonempty replay output requires inspection')
    bank.write_bytes(responses)
    immutable_cache.write_bytes(cache)
    mutable_cache.write_bytes(cache)
    write(provenance, receipt)


def validate_b(result, identity, panel):
    if (result['development_questions'] != 256 or result.get('test_used') is not False
            or identity.get('test_used') is not False
            or identity['dev_questions_sha256'] != panel
            or identity.get('source_sha256') != sha(control.__file__)
            or 'ordinary_B_source' not in identity or 'root_stop_rules' not in identity
            or result['B'] != control.choose(result['ordinary_B'], result['matched_B'])):
        raise ValueError('Require the exact strongest completed ordinary/matched LLM control')


def finalize(output):
    summary_path = output / 'summary.json'
    marker_path = output / 'replay-complete.json'
    if marker_path.exists():
        expected = {'schema': SCHEMA, 'source_sha256': sha(__file__),
                    'summary_sha256': sha(summary_path), 'identity_sha256': sha(output / 'identity.json'),
                    'import_sha256': sha(output / 'cache-import.json')}
        if read(marker_path) != expected:
            raise ValueError('Sealed final replay artifacts changed')
        return
    result = read(summary_path)
    result['B_comparison'] = 'Highest independent ordinary or matched-review B on the identical256question panel.'
    result['A_reference_scope'] = 'Inherited A8 only; final comparison must retain the strongest completed A.'
    result['final_gap_claim'] = False
    result['replay_import_sha256'] = sha(output / 'cache-import.json')
    write(summary_path, result)
    marker = {'schema': SCHEMA, 'source_sha256': sha(__file__),
              'summary_sha256': sha(summary_path), 'identity_sha256': sha(output / 'identity.json'),
              'import_sha256': sha(output / 'cache-import.json')}
    write(marker_path, marker)


def run_locked(args):
    # These waits cannot read development question values or start inference.
    for folder in (args.responses, args.cache):
        wait_completed(folder)
    wait(args.b_reference)
    wait(args.models / 'complete.json')
    panel = sha(args.data / 'splits/dev-questions.jsonl')
    responses, response_receipt = completed_bank(args.responses, panel)
    _, cache_receipt = completed_bank(args.cache, panel)
    chosen, chosen_recipe = model_assets(args.models)
    cached, cached_recipe = model_assets(args.cache_models)
    identity = read(args.models / 'identity.json')
    if (not chosen.get('new_source_only') or chosen['fit_questions'] != 512
            or chosen['paired_review_questions'] != 128
            or chosen.get('development_or_test_used') is not False):
        raise ValueError('Only completed 512-question/128-pair new-source fits are supported')
    if any(sha(path) != digest for path, digest in identity['source_files'].items()):
        raise ValueError('Registered new-source fitter dependency changed')
    cache_identity = read(args.cache / 'identity.json')
    if (cache_identity['models_sha256'] != sha(args.cache_models / 'complete.json')
            or cache_identity['source_sha256'] != sha(study.__file__)
            or chosen_recipe != cached_recipe):
        raise ValueError('Neural cache does not belong to compatible completed models')
    b = read(args.b_reference)
    b_identity = read(args.b_reference.parent / 'identity.json')
    validate_b(b, b_identity, panel)
    source_cache = args.cache / 'private-neural-cache.pkl'
    cache_bytes = source_cache.read_bytes()
    # The payload is the observer's raw forward cache, not calibrated estimates.
    cache_values(cache_bytes)
    receipt = {'schema': SCHEMA, 'source_sha256': sha(__file__), 'evaluator_sha256': sha(study.__file__),
               'response_source': response_receipt, 'neural_source': cache_receipt,
               'model_complete_sha256': sha(args.models / 'complete.json'),
               'cache_model_complete_sha256': sha(args.cache_models / 'complete.json'),
               'neural_recipe_sha256': chosen_recipe, 'neural_cache_sha256': hashlib.sha256(cache_bytes).hexdigest(),
               'B_reference_sha256': sha(args.b_reference), 'dev_questions_sha256': panel,
               'selection_by_correctness': False, 'test_used': False,
               'reuse': 'Every completed response and raw neural entry; each virtual paid observation is charged.',
               'final_gap_claim': False}
    copy_prefix(args.output, responses, cache_bytes, receipt)
    summary_path = args.output / 'summary.json'
    if summary_path.exists():
        if read(args.output / 'identity.json')['models_sha256'] != receipt['model_complete_sha256']:
            raise ValueError('Completed replay belongs to different models')
        finalize(args.output)
        completed_bank(args.output, panel)
        print('Verified completed new-source development replay; no work repeated.', flush=True)
        return
    print('Imported complete development banks; starting new-source hybrid comparison.', flush=True)
    args.collect, args.wait = True, True
    study.run(args)
    finalize(args.output)
    completed_bank(args.output, panel)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.replay.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'models', 'responses', 'cache', 'cache-models', 'b-reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    run(parser.parse_args())
