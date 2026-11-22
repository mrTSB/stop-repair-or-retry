"""Versioned runner correcting nested prerequisite evidence validation.

The first registered runner failed before inference because it interpreted
metadata keys as file paths. Its source and empty registration are preserved.
This runner uses exactly the same protocol, prompts, budget and grading code;
its new registration explicitly binds this adapter and the original sources.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

import grounded_root_review_pilot as original
import completion_v4_diagnostic as guarded

SCHEMA = 'grounded-review-runner-nested-prerequisite-evidence-v2'
original_registration = original.registration


def require_priority(args):
    io, paired, protocol = original.io, original.paired, original.protocol
    done=io.read(args.priority/'complete.json'); record=io.read(args.priority/'registration.json')
    if (done['schema']!=paired.SCHEMA or record['schema']!=paired.SCHEMA
            or done['registration_sha256']!=io.sha(args.priority/'registration.json')
            or done['coverage']['physical_calls']!=5120 or done['coverage']['logical_actions']!=5120
            or done['coverage']['physical_attempts']!=5120 or done['test_used'] or done['scored']):
        raise ValueError('Complete ungraded paired collection required')
    evidence=record['input_evidence']
    if not isinstance(evidence.get('files'),dict) or not evidence['files']:
        raise ValueError('Require explicit nonempty prerequisite file manifest')
    io.verify(record['source_files']);io.verify(evidence['files'])
    io.verify({str(args.priority/n):h for n,h in done['digests'].items()})
    capacity=protocol.strict_records(args.priority/'capacity.jsonl')
    if not capacity or capacity[-1]['min_containers']!=0 or capacity[-1]['max_containers']!=2:
        raise ValueError('Prior collector has not restored zero minimum capacity')


def registration(args):
    record=original_registration(args)
    failed=args.data.parent/'grounded1024-root-review-pilot-v1'
    if {p.name for p in failed.iterdir()}!={'registration.json'}:
        raise ValueError('The prior runner must have failed before any inference or attempts')
    prior_path=failed/'registration.json'
    prior=original.io.read(prior_path)
    original.io.verify(prior['source_files']);original.io.verify(prior['input_evidence'])
    record['input_evidence'][str(prior_path.resolve())]=original.io.sha(prior_path)
    record['source_files']=original.io.sources(sys.modules[__name__])
    record['runner_schema']=SCHEMA
    record['runner_change']='Verify nested input_evidence.files, preserving all protocol and completion checks'
    record['prior_failed_before_inference']='grounded1024-root-review-pilot-v1'
    return record


@contextmanager
def adapted_runner():
    # Process-local dependency injection. No registered source file is edited.
    previous_registration,previous_priority=original.registration,original.require_priority
    original.registration,original.require_priority=registration,require_priority
    try: yield
    finally: original.registration,original.require_priority=previous_registration,previous_priority


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','collect','grade'))
    for name in ('data','mentor','complete','late','output','priority'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');args=parser.parse_args()
    with guarded.training_only_reads(), adapted_runner():
        {'register':original.register,'collect':original.collect,'grade':original.grade}[args.phase](args)


if __name__=='__main__': main()
