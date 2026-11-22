"""Separate expanded-corpus pilot, reusing exactly identical paid trajectories.

All registered predecessor files stay unchanged. This adapter explicitly binds
the new corpus and its configuration in a new registration. Unchanged prompts
reuse sealed prior physical responses; every trajectory still pays full inference
cost. Novelty is measured beyond the first grounded pilot as well as all older
paths. No benchmark questions or outcomes selected the added lecture corpus.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import grounded_review_runner_v2 as previous

original=previous.original
spec,io,protocol=original.spec,original.io,original.protocol
SCHEMA='expanded-lecture-corpus-grounded-review-runner-v1'
PRIOR='grounded1024-root-review-pilot-v2'
CORPUS=Path(__file__).resolve().parents[4]/'runs/controller-semantic-followup/round2/advanced-textbooks-v1'


def prior_args(args):
    output=args.data.parent/PRIOR
    return SimpleNamespace(**{**vars(args),'output':output,'registration_sha256':io.sha(output/'registration.json')})


def require_priority(args):
    previous.require_priority(args)
    capacity=protocol.strict_records(args.data.parent/PRIOR/'capacity.jsonl')
    if not capacity or capacity[-1]['min_containers']!=0 or capacity[-1]['max_containers']!=2:
        raise ValueError('The first grounded pilot must finish and release capacity')


def verify_predecessor(args):
    # Called outside the new corpus scope so the previous contract reconstructs
    # exactly with its original corpus and source hashes.
    with previous.adapted_runner(): original.validated_complete(prior_args(args))
    require_priority(args)
    done=io.read(CORPUS/'complete.json')
    if done['benchmark_records_read'] or done['model_calls']: raise ValueError('Require question-independent corpus')
    io.verify({str(CORPUS/n):h for n,h in done['digests'].items()})


@contextmanager
def configured_runner():
    old=(spec.CORPUS,spec.EXTRA_BANKS,original.registration,original.require_priority)
    spec.retriever.cache_clear();spec.context.cache_clear()
    spec.CORPUS=CORPUS;spec.EXTRA_BANKS=(*spec.EXTRA_BANKS,PRIOR)
    original.registration,original.require_priority=registration,require_priority
    try: yield
    finally:
        spec.CORPUS,spec.EXTRA_BANKS,original.registration,original.require_priority=old
        spec.retriever.cache_clear();spec.context.cache_clear()


def compatible(rows,trees,raw,attempts):
    lookup={r['request_sha256']:r for r in raw};attempt_lookup=original.attempts_checked(attempts,raw)
    logical=[];chosen=[];questions=[]
    for row in rows:
        tree=trees[row['id']];current=tree['']
        try:teacher,calls=spec.review(row,tree,lookup)
        except spec.MissingPhysical:continue
        group=[spec.logical(row,tree[''],'judge','',teacher,calls)];physical=list(calls)
        for path in spec.PATHS:
            node,calls=spec.solver(row,current,teacher,path,lookup)
            group.append(spec.logical(row,tree[''],'solver',path,node,calls,current,teacher))
            physical.extend(calls);current=node
        logical.extend(group);chosen.extend(physical);questions.append(row['id'])
    keys=[r['request_sha256'] for r in chosen]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate reused physical request')
    return logical,chosen,[attempt_lookup[k] for k in keys],questions


def reused_inputs(args):
    rows,trees,*_=spec.public_inputs(args);folder=args.data.parent/PRIOR
    return compatible(rows,trees,protocol.strict_records(folder/'raw-generations.jsonl'),
                      protocol.strict_records(folder/'attempts.jsonl'))


def registration(args):
    record=previous.original_registration(args)
    logical,raw,attempts,questions=reused_inputs(args)
    if 64-len(questions)<6:raise ValueError('Fewer than six changed trajectories cannot pass the novelty screen')
    record['source_files']=io.sources(sys.modules[__name__])
    folder=args.data.parent/PRIOR;done=io.read(folder/'complete.json')
    record['input_evidence'].update({str((folder/n).resolve()):h for n,h in done['digests'].items()})
    record['input_evidence'][str((CORPUS/'complete.json').resolve())]=io.sha(CORPUS/'complete.json')
    record.update(runner_schema=SCHEMA,corpus_directory=str(CORPUS),
        corpus_change_only=True,prior_grounded_pilot_in_known_union=True,
        unchanged_prompts_reused_questions=len(questions),reused_physical_calls=len(raw),
        reused_logical_nodes=len(logical),reused_question_ids=questions,
        maximum_new_physical_calls=256-len(raw),all_reused_tokens_charged_at_inference=True,
        reuse_identity=spec.old.digest({'logical':logical,'raw':raw,'attempts':attempts}))
    return record


def seed_reused(args):
    logical,raw,attempts,questions=reused_inputs(args)
    record=registration(args)
    if (io.sha(args.output/'registration.json')!=args.registration_sha256
            or io.read(args.output/'registration.json')!=record):raise ValueError('Changed registered corpus comparison')
    payloads={n:''.join(json.dumps(v,sort_keys=True)+'\n' for v in values).encode()
              for n,values in (('train.jsonl',logical),('raw-generations.jsonl',raw),('attempts.jsonl',attempts))}
    receipt={'schema':SCHEMA,'registration_sha256':args.registration_sha256,'questions':len(questions),
        'physical_calls':len(raw),'reuse_identity':record['reuse_identity'],
        'prefixes':{n:{'bytes':len(v),'sha256':spec.hashlib.sha256(v).hexdigest()} for n,v in payloads.items()}}
    target=args.output/'reuse-receipt.json'
    with io.lock(args.output/'.reuse.lock'):
        if target.exists():
            if io.read(target)!=receipt:raise ValueError('Changed initial reuse receipt')
            for name,content in payloads.items():
                if not (args.output/name).read_bytes().startswith(content):raise ValueError('Changed reused physical prefix')
            return
        if any((args.output/n).exists() for n in payloads):raise ValueError('Partial reuse seed requires inspection')
        for name,content in payloads.items():io.atomic_bytes(args.output/name,content)
        io.write(target,receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','collect','grade'))
    for name in ('data','mentor','complete','late','output','priority'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');args=parser.parse_args()
    with previous.guarded.training_only_reads():
        verify_predecessor(args)
        with configured_runner():
            if args.phase=='register':original.register(args)
            elif args.phase=='collect':seed_reused(args);original.collect(args)
            else:
                seed_reused(args);original.grade(args)
                done=io.read(args.output/'complete.json');record=io.read(args.output/'registration.json')
                io.write(args.output/'acquisition-reuse-summary.json',{'runner_schema':SCHEMA,
                    'physical_calls_in_full_response_ledger':done['coverage']['physical_calls'],
                    'reused_physical_calls':record['reused_physical_calls'],
                    'new_physical_calls':done['coverage']['physical_calls']-record['reused_physical_calls'],
                    'full_inference_costs_charged':True,'test_used':False})


if __name__=='__main__':main()
