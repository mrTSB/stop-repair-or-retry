"""Narrow provenance-only correction to the preserved pure-A4 v1 driver.

Retained candidates have an outer comparison summary and a resolved model
summary. Their distinct identities must be checked at their own directories.
The inherited cache conversion,14-setting policy replay and cost arithmetic
are unchanged. The failed v1 directory and source remain immutable.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

import development_expanded_plain_a4 as original

io = original.io
SCHEMA = 'round2-expanded-pure-plain-A4-development-v2'
original_seal = original.seal_payload


def verify_a_references(references):
    for ref in references:
        resolved = ref['resolved']
        io.verify({ref['summary']:ref['summary_sha256'],
            resolved['summary']:resolved['summary_sha256'],
            resolved['model_path']:resolved['model_sha256'],
            str(Path(resolved['summary']).parent/'identity.json'):resolved['identity_sha256']})


def prior_attempt(directory):
    registered = io.read(directory/'registration.json')
    if registered['schema'] != original.SCHEMA or set(p.name for p in directory.iterdir()) != {'registration.json'}:
        raise ValueError('Require preserved pre-replay v1 registration with no policy outputs')
    io.verify(registered['source_files'])
    return {'directory':str(directory.resolve()),'registration_sha256':io.sha(directory/'registration.json'),
        'failure':'Retained-reference identity path used outer instead of resolved summary directory',
        'policy_evaluations_before_failure':0,'sources_and_outputs_unchanged':True}


def contract(args):
    result = original.contract(args)
    result.update(schema=SCHEMA,source_files=io.sources(sys.modules[__name__]),
        provenance_validation='Outer summary and recursively resolved summary/identity independently hash-verified',
        inherited_runtime_sha256=io.sha(original.__file__),prior_failed_attempt=prior_attempt(args.previous_attempt),
        policy_cache_compute_and_grid_unchanged=True)
    verify_a_references(result['A_references'])
    return result


def register(args):
    if args.output.exists():raise ValueError('Use a new v2 pure-A4 output')
    record=contract(args);args.output.mkdir(parents=True);io.write(args.output/'registration.json',record)
    print({'registered':True,'registration_sha256':io.sha(args.output/'registration.json')},flush=True)


def validate_registration(args):
    path=args.output/'registration.json';record=io.read(path)
    if io.sha(path)!=args.registration_sha256 or record['schema']!=SCHEMA:
        raise ValueError('Explicit v2 pure-A4 registration changed')
    io.verify(record['source_files']);io.verify(record['observer_assets'])
    io.verify({record['model']['model_path']:record['model']['sha256']})
    if record['plain_feature_source']!=(str(args.plain_features.resolve()) if args.plain_features is not None else None):
        raise ValueError('Changed explicitly registered whole768 feature source')
    if record['plain_feature_conversion'] is not None:io.verify(record['plain_feature_conversion']['files'])
    if (record['paths']!={n:str(getattr(args,n).resolve()) for n in record['paths']}
        or record['dev_questions_sha256']!=io.sha(args.data/'splits/dev-questions.jsonl')
        or record['plain_complete_sha256']!=io.sha(args.data/'dev-plain-complete.json')
        or record['fit_complete_sha256']!=io.sha(args.models/'complete.json')
        or record['settings']!=original.SETTINGS):
        raise ValueError('Changed v2 pure-A4 registered inputs or grid')
    verify_a_references(record['A_references'])
    # The old schema is explicit here because execution temporarily selects v2
    # output metadata on the unchanged inherited runtime module.
    old=io.read(args.previous_attempt/'registration.json')
    if (old['schema']!='round2-expanded-pure-plain-A4-development-v1'
        or str(args.previous_attempt.resolve())!=record['prior_failed_attempt']['directory']
        or io.sha(args.previous_attempt/'registration.json')!=record['prior_failed_attempt']['registration_sha256']
        or set(p.name for p in args.previous_attempt.iterdir())!={'registration.json'}):
        raise ValueError('Preserved failed v1 attempt changed')
    if original.cache_source(args.source,record['dev_questions_sha256'],io.sha(args.data/'dev-plain-complete.json'))[1]!=record['cache_source']:
        raise ValueError('Changed v2 registered cache')
    return record


def seal_payload(directory):
    payload=original_seal(directory)
    return {**payload,'schema':SCHEMA,'source_sha256':io.sha(__file__),
            'inherited_runtime_sha256':io.sha(original.__file__)}


@contextmanager
def validation_override():
    before=(original.validate_registration,original.seal_payload,original.SCHEMA)
    original.validate_registration,original.seal_payload,original.SCHEMA=validate_registration,seal_payload,SCHEMA
    try:yield
    finally:original.validate_registration,original.seal_payload,original.SCHEMA=before


def run(args):
    with validation_override():original.run(args)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','run'))
    for name in ('data','models','source','registry','output','previous-attempt'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plain-features',type=Path);parser.add_argument('--registration-sha256')
    args=parser.parse_args()
    if args.mode=='register':register(args)
    elif not args.registration_sha256:parser.error('Execution requires reviewed registration SHA')
    else:run(args)


if __name__=='__main__':main()
