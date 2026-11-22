"""Read-only join of two fully sealed DEV action grids; never a test reader.

This verifies the retained-review audit and all384 Complete calls before
opening DEV references. It provides data for a separately registered hybrid/
matched-B experiment and a clearly labeled hindsight feasibility bound.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
from types import SimpleNamespace

import development_retained_path_audit as retained
import development_complete_actions as complete
import expanded_mentor_io as io
from prepare_mixed_mentor import plain_data
from scoring import correct as answer_correct


def strong_data(data, directory):
    contract=io.read(directory/'registration.json')
    marker=io.read(directory/'collection-complete.json')
    args=SimpleNamespace(**{k:Path(v) for k,v in contract['paths'].items()})
    if (args.data.resolve()!=data.resolve() or args.output.resolve()!=directory.resolve()
            or contract['schema']!=retained.SCHEMA or contract['scope']!=retained.SCOPE
            or marker['source_sha256']!=io.sha(retained.__file__)
            or marker['registration_sha256']!=io.sha(directory/'registration.json')):
        raise ValueError('Retained science grid belongs to another source/panel')
    # Verify the historical captured closure as stored. A new caller is not
    # required to have the same ambient sys.modules inventory as that runner.
    io.verify(contract['source_files'])
    io.verify({str(directory/name):digest for name,digest in marker['digests'].items()})
    public,original,_,payload,_,_,evidence=retained.source_inputs(args)
    if evidence!=contract['evidence'] or retained.missing_plan(public,original)!=contract['plan']:
        raise ValueError('Retained grid original whole-bank ancestry changed')
    if not (directory/'dev-observations.jsonl').read_bytes().startswith(payload):
        raise ValueError('The imported whole response bank is not retained exactly')
    records=retained.serial.strict_records(directory/'dev-observations.jsonl')
    raw=retained.serial.strict_records(directory/'raw-generations.jsonl')
    public,trees,judges=retained.public_view(data,records)
    allowed={(r['id'],r['path']) for r in contract['plan']['missing']}
    count=len(payload.splitlines())
    if any(r['kind']!='solver' or (r['id'],r['path']) not in allowed or not r.get('metadata_acquired')
           for r in records[count:]):
        raise ValueError('Retained suffix contains an unregistered or teacher action')
    if marker!=retained.completion_payload(args,contract,public,trees,records,raw):
        raise ValueError('Require the exact complete1280-cell retained science seal')
    files={str((directory/'collection-complete.json').resolve()):io.sha(directory/'collection-complete.json'),
           **{str((directory/name).resolve()):value for name,value in marker['digests'].items()},
           **contract['source_files']}
    return public,trees,judges,{'directory':str(directory.resolve()),'files':files,
        'complete_sha256':io.sha(directory/'collection-complete.json'),'historical_source_closure_verified':True,
        'new_caller_ambient_modules_not_part_of_historical_identity':True,'current_B_correct':contract['evidence']['B_correct'],
        'B_reference':str(args.b_reference.resolve()),'B_summary_sha256':contract['evidence']['B_summary_sha256']}


def load(data, retained_directory, complete_directory):
    public,strong,judges,strong_receipt=strong_data(data,retained_directory)
    science,completed,complete_receipt=complete.development_data(data,complete_directory)
    rows,plain=plain_data(data,'dev')
    if (len(rows)!=256 or Counter(r['subject'] for r in rows)!= {'math':128,'chemistry':64,'physics':64}
            or [retained.serial.public_row(r) for r in rows]!=public):
        raise ValueError('Full labeled DEV panel differs from the paid public panel')
    by_id={r['id']:r for r in rows}
    if (len(science)!=128 or Counter(r['subject'] for r in science)!= {'chemistry':64,'physics':64}
            or {r['id'] for r in science}!={r['id'] for r in rows if r['subject']!='math'}):
        raise ValueError('Complete grid omitted a science question')
    for row in rows:
        qid=row['id']
        if any(strong[qid][path]!=node for path,node in plain[qid].items()):
            raise ValueError('Original five controls changed in the joined source')
    for row in science:
        qid=row['id']
        if row!=by_id[qid] or any(completed[qid][path]!=strong[qid][path] for path in retained.PLAIN):
            raise ValueError('Complete and retained grids changed a question/reference/plain response')
        for path in ('complete','repair/complete','retry/complete'):
            if path in strong[qid]: raise ValueError('Unexpected preexisting Complete protocol')
            strong[qid][path]=completed[qid][path]
    receipt={'schema':'complete-retained-DEV-join-v1','retained':strong_receipt,'complete':complete_receipt,
        'questions':256,'science_questions':128,'dev_questions_sha256':io.sha(data/'splits/dev-questions.jsonl'),
        'all_original_controls_unchanged':True,'outcome_filtering':False,'all_action_grids_sealed_before_labels':True,
        'new_cloud_calls':0,'new_neural_calls':0,'reserved_test_used':False}
    return rows,strong,judges,receipt


def availability(rows,trees,reference_correct):
    groups={'plain7':retained.PLAIN,'strong10':retained.PATHS,
        'plain_plus_Complete':(*retained.PLAIN,'complete','repair/complete','retry/complete'),
        'combined13':(*retained.PATHS,'complete','repair/complete','retry/complete')}
    panels={}
    for subject in ('all_science','chemistry','physics'):
        panel=[r for r in rows if r['subject']!='math' and (subject=='all_science' or r['subject']==subject)]
        if len(panel)!=(128 if subject=='all_science' else 64): raise ValueError('Require all128DEV science questions')
        correct={key:[any(answer_correct(trees[r['id']][p],r['answer']) for p in paths)
                      for r in panel] for key,paths in groups.items()}
        panels[subject]={'questions':len(panel),'hindsight_available_correct':{k:sum(v) for k,v in correct.items()},
            'Complete_novel_beyond_strong':sum(a and not b for a,b in zip(correct['combined13'],correct['strong10']))}
    math_rows=[r for r in rows if r['subject']=='math']
    if len(math_rows)!=128: raise ValueError('Require the same128math questions')
    math_known=sum(any(answer_correct(n,r['answer']) for n in trees[r['id']].values()) for r in math_rows)
    science=panels['all_science']['hindsight_available_correct']['combined13']
    required=math.ceil(reference_correct+.1*256-1e-12)
    return {'schema':'complete-retained-DEV-hindsight-availability-v1','science_panels':panels,
        'known_math_union':math_known,'known_full_union':math_known+science,
        'optimistic_full_bound_with_perfect_math':128+science,'comparison_reference_correct':reference_correct,
        'correct_required_for_10_point_gap':required,'target_possible_even_with_perfect_math':128+science>=required,
        'interpretation':'Hindsight DEV availability only; not controller accuracy, learnability or fresh validation',
        'reserved_test_used':False,'new_cloud_or_neural_calls':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','retained','complete','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve the existing availability audit')
    rows,trees,_,receipt=load(args.data,args.retained,args.complete)
    result=availability(rows,trees,receipt['retained']['current_B_correct'])
    io.write(args.output,{'join_receipt':receipt,**result})
    print(result,flush=True)
