"""Join two fully sealed disjoint TRAIN blocks; decode only their selected labels."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import auxiliary_second_completion_features as second
import fit_auxiliary_completion as first

io, fixed, nested = first.io, first.fixed, first.nested


def merge(old_rows, old_trees, old_data, new_rows, new_trees, new_data):
    rows = [*old_rows, *new_rows]
    if (len(old_rows) != 768 or len(new_rows) != 512 or len({r['id'] for r in rows}) != 1280
            or Counter(r['subject'] for r in rows) != {'chemistry': 640, 'physics': 640}):
        raise ValueError('Require 768 plus512 disjoint balanced science training questions')
    data = {'records': [*old_data['records'], *new_data['records']],
            'features': {k: np.concatenate([old_data['features'][k], new_data['features'][k]]) for k in nested.FAMILIES},
            'labels': np.concatenate([old_data['labels'], new_data['labels']]),
            'valid': np.concatenate([old_data['valid'], new_data['valid']])}
    if [(r['id'], r['path']) for r in data['records']] != [(r['id'], p) for r in rows for p in ('', 'repair', 'retry')]:
        raise ValueError('Changed complete question-prefix ordering')
    if any(data['features'][f].shape != (3840, width) for f, width in zip(nested.FAMILIES, (130, 386))):
        raise ValueError('Changed feature dimensions')
    return rows, {**old_trees, **new_trees}, data


def inputs(args):
    old_rows, old_trees, observer, old_data, old_record = first.inputs(SimpleNamespace(features=args.first_features, output=args.output))
    spec = io.read(args.second_features/'registration.json')
    source = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()},
                             registration_sha256=io.sha(args.second_features/'registration.json'))
    if source.output.resolve() != args.second_features.resolve(): raise ValueError('Changed second feature archive')
    bundle, expected = second.inputs(source)
    if expected != spec: raise ValueError('Changed registered second-block feature inputs')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    values, done = second.verify(source, bundle, spec)
    if observer.directory != bundle['observer'].directory or observer._recipe != bundle['observer']._recipe:
        raise ValueError('Require identical old excluded encoder assets')
    new_rows = first.reference_join(bundle['rows'], bundle['collection_args'].cache/'SuperGPQA-all.jsonl')
    new_data = first.table(new_rows, bundle['trees'], bundle['metadata'], values)
    rows, trees, data = merge(old_rows, old_trees, old_data, new_rows, bundle['trees'], new_data)
    evidence = {**old_record['input_evidence'], **spec['input_evidence'],
        **{str((args.second_features/n).resolve()): h for n, h in done['digests'].items()},
        str((args.second_features/'complete.json').resolve()): io.sha(args.second_features/'complete.json')}
    return rows, trees, observer, data, old_data, evidence
