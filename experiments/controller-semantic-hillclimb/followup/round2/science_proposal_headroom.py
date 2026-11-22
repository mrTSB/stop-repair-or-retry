"""Bounded TRAIN diagnostic of finalV3 option proposals before cloud expansion.

All options on the same fixed64 science questions are scored and charged.
This is answer-availability evidence, never solver/controller accuracy.
"""
import argparse
import gc
import math
from pathlib import Path
import sys
import time

import torch

import option_proposal_pairs as proposal
import science_option_observer as option_model
import strong_late_review_protocol as controls
import completion_v4_diagnostic as guarded
import extract_expanded_mentor as resource

io=controls.io
SCHEMA='final-V3-option-proposal-TRAIN-headroom-v1'


def inputs(args):
    rows,trees,reviews,ordinary,ordinary_reviews,chosen,_,evidence=controls.public_inputs(args)
    summary=io.read(args.verifier/'training-summary.json');recipe=io.read(args.verifier/'full/recipe.json')
    registered=io.read(args.verifier/'training-registration.json')
    if (summary['training_steps']!=2160 or summary['training_questions']!=1440 or summary['project_dev_or_test_used']
            or not summary['calibration_used_reloaded_archive'] or summary['recipe_sha256']!=io.sha(args.verifier/'full/recipe.json')
            or summary['registration_sha256']!=io.sha(args.verifier/'training-registration.json')
            or recipe['input_profile']!=option_model.PROFILE):raise ValueError('Require the completed finalV3 option model')
    # Its training dataset has a sealed exclusion of all current project splits.
    auxiliary=args.data.parent/'science-auxiliary';meta=io.read(auxiliary/'metadata.json');seal=io.read(auxiliary/'provenance/seal-receipt.json')
    if (meta['protected_split_sha256']['train']!=io.sha(args.data/'splits/train-questions.jsonl')
            or seal['metadata_sha256']!=io.sha(auxiliary/'metadata.json') or seal['protected_overlap']!=0
            or recipe['training_questions']!=1440 or registered['auxiliary_metadata_sha256']!=io.sha(auxiliary/'metadata.json')
            or registered['seal_receipt_sha256']!=io.sha(auxiliary/'provenance/seal-receipt.json')
            or recipe['auxiliary_metadata_sha256']!=registered['auxiliary_metadata_sha256']
            or recipe['protected_split_sha256']!=meta['protected_split_sha256']):
        raise ValueError('FinalV3 original TRAIN exclusion changed')
    io.verify(registered['source_files'])
    evidence.update({str(p.resolve()):io.sha(p) for p in (args.verifier/'training-summary.json',
        args.verifier/'training-registration.json',auxiliary/'metadata.json',auxiliary/'provenance/seal-receipt.json')})
    evidence.update({str(p.resolve()):io.sha(p) for p in (args.verifier/'full').iterdir() if p.is_file()})
    if recipe['weights']!=summary['calibration_weight_sha256']:raise ValueError('Calibration and deployed weights differ')
    io.verify({str(args.verifier/'full'/n):h for n,h in recipe['weights'].items()})
    extra={r['id']:[] for r in rows}
    for name in ('strong-late-review-pilot-v1','blind-first-root-review-pilot-v1'):
        directory=args.data.parent/name;done=io.read(directory/'complete.json');spec=io.read(directory/'registration.json')
        if (done['registration_sha256']!=io.sha(directory/'registration.json') or done['test_used'] or not done['training_only']):
            raise ValueError('Prior action availability is unsealed')
        io.verify(spec['source_files']);io.verify({str(directory/n):h for n,h in done['digests'].items()})
        evidence.update({str((directory/n).resolve()):h for n,h in done['digests'].items()})
        evidence[str((directory/'complete.json').resolve())]=io.sha(directory/'complete.json')
        for item in controls.protocol.strict_records(directory/'train.jsonl'):
            if item['kind']=='solver' and item['id'] in extra:extra[item['id']].append(item['response'])
    record={'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{n:str(getattr(args,n).resolve()) for n in ('data','mentor','complete','verifier','output')},
        'selection':chosen,'questions':64,'option_passes':sum(len(proposal.options(r)) for r in rows),
        'model':'final2160-step listwiseV3, no checkpoint selection','temperature':recipe['temperature'],
        'parameters':recipe['parameters'],'device':'cpu','threads':2,'every_option_scored':True,
        'public_rows_sha256':proposal.canonical_sha(rows),'controls_sha256':proposal.canonical_sha({'trees':trees,'ordinary':ordinary,'extra':extra}),
        'screen':{'novel_proposals_beyond_all_existing_solver_paths_at_least':6,
            'interpretation':'Training headroom gate before any paired compact feedback pilot; not guaranteed learnability or accuracy'},
        'new_cloud_calls':0,'new_encoder_training':False,'reference_answers_supplied':False,'development_or_test_read':False,
        'cost':'One actual trimmed encoder forward for every option; question retrieval/CPU latency separately recorded',
        'resume':'Question attempt journal before any forward; unresolved attempt blocks automatic repeat'}
    return rows,trees,ordinary,extra,record


def restore(args,rows,require_complete=False):
    attempts=controls.protocol.strict_records(args.output/'attempts.jsonl')
    saved=controls.protocol.strict_records(args.output/'proposals.jsonl');by_id={r['id']:r for r in rows};attempted=set();known={}
    for item in attempts:
        qid=item['id']
        if qid in attempted or qid not in by_id or item!={'id':qid,'public_sha256':proposal.canonical_sha(by_id[qid]),
                'registration_sha256':args.registration_sha256}:raise ValueError('Changed or duplicate proposal attempt')
        attempted.add(qid)
    for item in saved:
        qid=item['id'];p=item['proposal']
        if qid in known or qid not in attempted or item['registration_sha256']!=args.registration_sha256:raise ValueError('Unrecognized proposal')
        letters='ABCDEFGHIJ'[:len(proposal.options(by_id[qid]))]
        if ([v['letter'] for v in p['all_options']]!=list(letters) or p['encoder_calls']!=len(letters)
                or p['encoder_token_slots']!=sum(v['token_slots'] for v in p['all_options'])
                or any(not math.isfinite(v['calibrated_logit']) or type(v['token_slots'])is not int
                    or not 0<v['token_slots']<=512 for v in p['all_options'])
                or p['proposed_letter']!=max(p['all_options'],key=lambda v:(v['calibrated_logit'],-ord(v['letter'])))['letter']):
            raise ValueError('Incomplete option coverage or changed proposal choice')
        known[qid]=item
    if attempted!=set(known):raise ValueError('Unresolved encoder attempt; inspect before any repeat')
    if require_complete and set(known)!=set(by_id):raise ValueError('All64 questions must complete before grading')
    return known


def extract(args,rows,record):
    with io.lock(args.output/'.run.lock'):
        if (args.output/'complete.json').exists():raise ValueError('Preserve completed proposals')
        known=restore(args,rows);observer=None
        try:
            for row in rows:
                if row['id'] in known:continue
                resource.memory_guard(args.output,observer is not None)
                if observer is None:
                    observer=option_model.from_checkpoint(args.verifier/'full','cpu');observer.load();torch.set_num_threads(2)
                    if observer._recipe['parameters']!=record['parameters']:raise ValueError('Observer parameter count changed')
                io.append(args.output/'attempts.jsonl',{'id':row['id'],'public_sha256':proposal.canonical_sha(row),
                    'registration_sha256':args.registration_sha256})
                start=time.perf_counter();value=proposal.rank_options(observer,row)
                item={'id':row['id'],'subject':row['subject'],'proposal':value,
                    'elapsed_seconds':time.perf_counter()-start,'registration_sha256':args.registration_sha256}
                io.append(args.output/'proposals.jsonl',item);known[row['id']]=item;observer._cache.clear()
                if len(known)%8==0:print({'complete_proposals':len(known),'total':64},flush=True)
        finally:observer=None;gc.collect()
        known=restore(args,rows,True);io.verify(record['source_files']);io.verify(record['input_evidence'])
        calls=sum(v['proposal']['encoder_calls'] for v in known.values());slots=sum(v['proposal']['encoder_token_slots'] for v in known.values())
        if calls!=record['option_passes']:raise ValueError('Option pass ledger mismatch')
        io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
            'digests':{n:io.sha(args.output/n) for n in ('registration.json','attempts.jsonl','proposals.jsonl','memory.jsonl')},
            'questions':64,'encoder_calls':calls,'actual_trimmed_token_slots':slots,'encoder_compute':slots*record['parameters']/1e9/3.09,
            'scored':False,'new_cloud_calls':0,'development_or_test_read':False})


def grade(args,rows,trees,ordinary,extra,record):
    done=io.read(args.output/'complete.json')
    if (done['schema']!=SCHEMA or done['registration_sha256']!=args.registration_sha256 or done['scored']
            or done['new_cloud_calls'] or done['development_or_test_read']):raise ValueError('Require complete ungraded proposals')
    io.verify({str(args.output/n):h for n,h in done['digests'].items()});known=restore(args,rows,True)
    if (args.output/'graded').exists():raise ValueError('Preserve first complete diagnostic')
    labels={r['id']:r for r in controls.previous.fixed.lines(args.data/'splits/train-questions.jsonl')}
    outcomes=[]
    for row in rows:
        qid=row['id']
        if controls.protocol.public_row(labels[qid])!=row:raise ValueError('TRAIN reference identity changed')
        target=labels[qid]['answer'];p=known[qid]['proposal'];right=p['proposed_letter']==target
        available=any(n['answer'] and n['answer']==target for n in [*trees[qid].values(),*ordinary[qid].values(),*extra[qid]])
        outcomes.append({'id':qid,'subject':row['subject'],'proposal_correct':right,'existing_available':bool(available),'novel':right and not available})
    reports={s:{'n':len(panel),'proposal_correct':sum(v['proposal_correct'] for v in panel),
        'existing_solver_union':sum(v['existing_available'] for v in panel),'novel_proposals':sum(v['novel'] for v in panel)}
        for s in ('all','chemistry','physics') for panel in [[v for v in outcomes if s=='all' or v['subject']==s]]}
    summary={'schema':SCHEMA,'reports':reports,'screen_passed':reports['all']['novel_proposals']>=6,
        'source_complete_sha256':io.sha(args.output/'complete.json'),'new_cloud_calls':0,'controller_accuracy_claim':False,
        'fresh_validation':False,'development_or_test_read':False,'mean_proposal_encoder_compute':done['encoder_compute']/64,
        'interpretation':'An option proposal is not a compact solver answer. Even a passed screen needs paid paired feedback and learned validation.'}
    (args.output/'graded').mkdir();io.write(args.output/'graded/private-outcomes.json',outcomes);io.write(args.output/'graded/summary.json',summary)
    print(summary,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','register','extract','grade'))
    for n in ('data','mentor','complete','verifier','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--registration-sha256');args=p.parse_args()
    with guarded.training_only_reads():
        with guarded.guarded.no_model_calls():rows,trees,ordinary,extra,record=inputs(args)
        if args.phase=='plan':print({'questions':64,'encoder_passes':record['option_passes'],'new_cloud_calls':0});return
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'inference_started':False});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Registered proposal diagnostic changed')
        if args.phase=='extract':extract(args,rows,record)
        else:
            with guarded.guarded.no_model_calls():grade(args,rows,trees,ordinary,extra,record)


if __name__=='__main__':main()
