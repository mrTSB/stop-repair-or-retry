"""Add answer-independent, assessment-free textbook evidence to retired examples."""

import argparse
import hashlib
import json
from pathlib import Path

from semantic_data import read,write
from evidence import Retriever,left_text


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=False)
    retriever=Retriever(args.corpus);manifest={'corpus_sha256':retriever.sha256,'profile':'full-options-openstax-v1'}
    for split in ('train','dev'):
        rows={r['id']:r for r in read(args.data/'splits'/f'{split}-questions.jsonl')}
        contexts={qid:retriever.context({**r,'answer':''}) for qid,r in rows.items()}
        examples=read(args.previous/(split+'.jsonl'))
        for node in examples:node['left']=left_text(rows[node['id']],contexts[node['id']][0])
        write(args.output/(split+'.jsonl'),examples)
        write(args.output/(split+'-evidence.jsonl'),[{'id':qid,'passages':entries} for qid,(_,entries) in contexts.items()])
        manifest[split]={'questions':len(rows),'nodes':len(examples),'training_nodes':sum(r['train'] for r in examples),
            'questions_with_evidence':sum(bool(v[0]) for v in contexts.values()),
            'sha256':hashlib.sha256((args.output/(split+'.jsonl')).read_bytes()).hexdigest()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args())
