"""Frozen old-observer extraction on the complete512-question math TRAIN grid."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import auxiliary_completion_features as original
import collect_auxiliary_math_pairs as pairs
import math_specialization_diagnostic as math

io, fixed = original.io, original.fixed
SCHEMA = 'auxiliary-math512-old-V1-features-v1'


def inputs(args):
    spec = io.read(args.collection/'registration.json')
    source = SimpleNamespace(**{k:Path(v) for k,v in spec['paths'].items()}, registration_sha256=io.sha(args.collection/'registration.json'))
    if source.output.resolve() != args.collection.resolve(): raise ValueError('Different paired math archive')
    rows, known, done = pairs.validated_complete(source)
    old_spec = io.read(args.math_diagnostic/'registration.json'); old_done = io.read(args.math_diagnostic/'complete.json')
    old_args = SimpleNamespace(**{k:Path(v) for k,v in old_spec['paths'].items()})
    old_data, expected = math.inputs(old_args)
    if expected != old_spec or not old_done['screen_passed']: raise ValueError('Changed retained math diagnostic')
    io.verify(old_spec['source_files']); io.verify(old_spec['input_evidence'])
    io.verify({str(args.math_diagnostic/k):v for k,v in old_done['digests'].items()})
    base = io.read(old_args.base_fit/'registration.json'); observer = math.source.cached.observer_from(Path(base['paths']['base']))
    reservation = io.read(source.reservation/'registration.json')
    if reservation['paths']['base_fit'] != str(old_args.base_fit.resolve()): raise ValueError('Different frozen observer ancestry')
    if {r['id'] for r in rows}&{r['id'] for r in old_data['records']}: raise ValueError('Expanded math overlaps original fitting questions')
    trees = {r['id']:{p:known[r['id'],p]['response'] for p in pairs.contract.PATHS} for r in rows}
    entries, metadata = [], []
    for row in rows:
        for path in ('','repair','retry'):
            paid = {p:trees[row['id']][p] for p in ('',path)}; entry = observer.entry(row,path,paid)
            entries.append(entry); metadata.append({'id':row['id'],'subject':'math','path':path,
                'prompt_sha256':paid[path]['prompt_sha256'],'input_sha256':fixed.fingerprint(entry)})
    if len(entries) != 1536: raise ValueError('Require all512 math questions and three paid prefixes')
    evidence = {**spec['input_evidence']['files'], **old_spec['input_evidence'],
        **{str((args.collection/k).resolve()):v for k,v in done['digests'].items()},
        **{str((args.math_diagnostic/k).resolve()):v for k,v in old_done['digests'].items()},
        str((args.collection/'complete.json').resolve()):io.sha(args.collection/'complete.json'),
        str((args.math_diagnostic/'complete.json').resolve()):io.sha(args.math_diagnostic/'complete.json')}
    record = {'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{k:str(getattr(args,k).resolve()) for k in ('collection','math_diagnostic','output')},
        'entry_sha256':fixed.fingerprint(entries),'metadata_sha256':fixed.fingerprint(metadata),
        'new_questions':512,'new_observations':1536,'physical_batched_forwards':384,
        'combined_math_training_questions':768,'device':'cpu','threads':2,'batch':4,'chunk':32,'max_length':512,
        'parameters':observer._recipe['parameters'],'observer_directory':str(Path(observer.directory).resolve()),
        'extra_project256_passes':0,'cloud_calls':0,'encoder_training':False,'old_encoder_excluded_auxiliary':True,
        'new_auxiliary_references_decoded':False,'development_or_test_read':False,
        'future_fit':'Same eight-state math POMDP; nested whole-question folds on768 math TRAIN questions',
        'resume_rule':'Durable attempted chunk before inference; unresolved attempts block automatic repetition'}
    return {'rows':rows,'trees':trees,'entries':entries,'metadata':metadata,'observer':observer,'collection_args':source},record


@contextmanager
def configured():
    before = original.SCHEMA, original.pairs
    original.SCHEMA, original.pairs = SCHEMA,pairs
    try: yield
    finally: original.SCHEMA, original.pairs = before


def verify(args,bundle,record):
    with configured(): return original.verify(args,bundle,record)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('register','extract'))
    for key in ('collection','math-diagnostic','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with fixed.training_only_reads():
        with fixed.guarded.no_model_calls(): bundle,record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'inference_started':False},flush=True);return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
            or io.read(args.output/'registration.json') != record): raise ValueError('Changed math observation registration')
        with configured(): original.extract(args,bundle,record)


if __name__ == '__main__': main()
