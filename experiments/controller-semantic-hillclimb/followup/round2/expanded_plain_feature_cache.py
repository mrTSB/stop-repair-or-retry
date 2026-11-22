"""Explicit whole768-cache conversion for pure-A4; no forward or fitting call.

The source is the completed old-V1 nonlinear development feature bank. Every
question, ancestor-derived input hash, prompt, recipe and actual slot count is
checked. All768 rows come from this one source; overlapping shared-bank rows
are audited, never chosen according to predictions or correctness.
"""
import json
from pathlib import Path

import numpy as np

import expanded_mentor_io as io
import development_nonlinear as original

SCHEMA = 'round2-expanded-plain-whole768-feature-cache-v1'


def convert(directory, observer, recipe, rows, trees, inherited_cache, panel, plain_completion):
    completed = io.read(directory/'complete.json')
    if (completed.get('source_sha256') != io.sha(original.__file__) or completed.get('test_used') is not False
            or completed.get('nodes') != 768 or completed.get('dev_questions_sha256') != panel
            or completed.get('encoder_recipe_sha256') != io.sha(Path(observer.directory)/'recipe.json')):
        raise ValueError('Require the complete same-panel old-V1 development feature source')
    files = {str((directory/'complete.json').resolve()): io.sha(directory/'complete.json'),
        str((directory/'neural.npz').resolve()): completed['neural_sha256'],
        str((directory/'metadata.jsonl').resolve()): completed['metadata_sha256']}
    io.verify(files)
    if observer._model is not None: raise ValueError('Conversion requires an unloaded observer')
    metadata = [json.loads(line) for line in (directory/'metadata.jsonl').read_bytes().splitlines()]
    previous_recipe = observer._recipe
    observer._recipe = recipe
    try:
        with np.load(directory/'neural.npz') as archive:
            outputs = {name: archive[name].copy() for name in ('state','plain','guided')}
        lookup, checked = original.verify_features(rows, trees, observer, metadata, outputs, True)
    finally: observer._recipe = previous_recipe
    if len(rows) != 256 or len(lookup) != 768 or observer._model is not None:
        raise ValueError('Incomplete or unexpectedly inferred conversion')
    corpus = Path(original.perception.CORPUS)/'passages.jsonl'
    if io.sha(corpus) != recipe['corpus_sha256'] or observer._retriever.sha256 != recipe['corpus_sha256']:
        raise ValueError('Input evidence corpus changed')
    files[str(corpus.resolve())] = recipe['corpus_sha256']
    converted, overlaps, identical, max_difference = {}, 0, 0, 0.
    for row in rows:
        for path in ('','repair','retry'):
            values = lookup[row['id'],path]
            key = observer.key({**row,'answer':''}, path, trees[row['id']][path], 'solver')
            converted[key] = values
            if key in inherited_cache:
                old = inherited_cache[key]; overlaps += 1
                if int(old[3]) != int(values[3]): raise ValueError('Same input has inconsistent physical token slots')
                equal = all(np.array_equal(a,b) for a,b in zip(old,values)); identical += int(equal)
                max_difference = max(max_difference, *[float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)))) for a,b in zip(old[:3],values[:3])])
    if len(converted) != 768: raise ValueError('Duplicate converted feature identity')
    return converted, {'schema': SCHEMA, 'directory':str(directory.resolve()),'files':files,
        'panel_sha256':panel,'plain_completion_sha256':plain_completion,'recipe_sha256':completed['encoder_recipe_sha256'],
        'source_nodes':768,'converted_nodes':len(converted),'checked_inputs_sha256':original.fingerprint(checked),
        'overlap_with_shared_cache':overlaps,'bitwise_identical_overlap':identical,'max_absolute_logit_difference':max_difference,
        'source_choice':'Entire fixed768-row feature source; no per-row source or correctness selection',
        'new_encoder_calls':0,'new_cloud_calls':0,'labels_used_for_conversion':False,'test_used':False}
