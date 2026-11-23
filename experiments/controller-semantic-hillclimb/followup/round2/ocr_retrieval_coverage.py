"""Answer-free TRAIN coverage check before acquiring a new OCR review pilot.

Uses the same fixed64 public questions as the preceding science pilots. It
does not decode references or read any solver/reviewer outcomes. Retrieval
changes are potential evidence coverage, never measured answer improvements.
"""
import argparse
import json
from pathlib import Path

import dense_textbook_retrieval as old
import ocr_dense_index as new
import science_mentor_expansion as public

read,write,sha = new.read,new.write,new.sha
SCHEMA = 'answer-free-OCR-retrieval-TRAIN-coverage-v1'


def contract(args):
    parent = read(args.selection/'registration.json')
    chosen = parent['selection']
    if len(chosen) != 64 or len({r['id'] for r in chosen}) != 64:
        raise ValueError('Require the unchanged64-question science selection')
    return {'schema':SCHEMA,'source_sha256':sha(__file__),
        'dependencies':{str(Path(m.__file__).resolve()):sha(m.__file__) for m in (old,new,public)},
        'selection_registration_sha256':sha(args.selection/'registration.json'),
        'selection':chosen,'public_train_file_sha256':sha(args.data/'splits/train-questions.jsonl'),
        'old_complete_sha256':sha(old.INDEX/'complete.json'),
        'new_complete_sha256':sha(args.index/'complete.json'),
        'query':'Same public question stem, no choices, capped800characters',
        'forward_plan':'One BGE query per unique question; same vector compares old and new indexes',
        'references_decoded':False,'solver_or_reviewer_outputs_read':False,
        'development_used':False,'test_used':False,'cloud_calls':0,'controller_accuracy_claim':False,
        'preacquisition_screen':'At least6 changed context questions AND at least6 questions containing new OCR excerpts; necessary, not sufficient, for6 novel correct trajectories'}


def run(args,record):
    if (sha(args.output/'registration.json') != args.registration_sha256
            or read(args.output/'registration.json') != record): raise ValueError('Changed coverage registration')
    if {p.name for p in args.output.iterdir()} != {'registration.json'}: raise ValueError('Preserve previous coverage outputs')
    original,updated = old.Retriever(),new.Retriever(args.index)
    by_id = {r['id']:r for r in public.public_train(args.data)}
    outcomes,receipts = [],[]
    for chosen in record['selection']:
        row = by_id[chosen['id']]
        if row['subject'] != chosen['subject']: raise ValueError('Changed selected subject')
        query = row['question'].removesuffix(public.protocol.old.SCIENCE_SUFFIX).partition('\n\nOptions:\n')[0][:800]
        vector,receipt = updated.encoded(query)
        contexts = []
        for retriever in (original,updated):
            dense = retriever.vectors@vector
            lexical = (retriever.matrix@retriever.vectorizer.transform([query]).T).toarray().ravel()
            found = old.rank(retriever.passages,dense,lexical,row['subject'])
            contexts.append(found)
        context = lambda found:'\n\n'.join(r['title']+': '+r['text'] for r in found)
        outcomes.append({'id':row['id'],'subject':row['subject'],
            'context_changed':context(contexts[0]) != context(contexts[1]),
            'OCR_excerpts':sum(r['title'].startswith('OCR lecture page ') for r in contexts[1]),
            'old':contexts[0],'updated':contexts[1]})
        receipts.append(receipt)
    changed = sum(r['context_changed'] for r in outcomes)
    with_OCR = sum(r['OCR_excerpts'] > 0 for r in outcomes)
    summary = {'schema':SCHEMA,'questions':len(outcomes),'context_changed_questions':changed,
        'new_OCR_evidence_questions':with_OCR,'OCR_excerpts':sum(r['OCR_excerpts'] for r in outcomes),
        'preacquisition_screen_passed':changed >= 6 and with_OCR >= 6,
        'actual_query_forwards':updated.encoded.cache_info().misses,
        'query_compute_if_used_by_controller':sum(r['compute'] for r in receipts),
        'references_decoded':False,'solver_or_reviewer_outputs_read':False,
        'development_used':False,'test_used':False,'cloud_calls':0,'controller_accuracy_claim':False}
    if contract(args) != record: raise ValueError('Changed input evidence during coverage check')
    write(args.output/'private-contexts.json',outcomes)
    write(args.output/'query-receipts.json',receipts)
    write(args.output/'summary.json',summary)
    names = ('registration.json','private-contexts.json','query-receipts.json','summary.json')
    write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'digests':{name:sha(args.output/name) for name in names},'controller_accuracy_claim':False})
    print(summary,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','run'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--index',type=Path,default=new.INDEX)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    record = contract(args)
    if args.phase == 'register':
        args.output.mkdir(parents=True,exist_ok=False)
        write(args.output/'registration.json',record)
        print({'registration_sha256':sha(args.output/'registration.json'),'questions':64},flush=True)
    else: run(args,record)


if __name__ == '__main__': main()
