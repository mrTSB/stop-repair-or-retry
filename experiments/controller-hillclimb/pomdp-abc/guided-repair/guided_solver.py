"""Paired training probes for using a paid judge's diagnosis during solver repair."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from acquire import MODEL, MODEL_DIGEST, SEED, balanced, chat, confidence, read, request
from compact import parse_answer
from collect import SCIENCE_SUFFIX

CAP = 256


def check_text(verdict):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{',verdict['text']):
        try:
            obj,_ = decoder.raw_decode(verdict['text'][match.start():])
            if isinstance(obj,dict) and isinstance(obj.get('check'),str):
                return obj['check']
        except json.JSONDecodeError:
            continue
    return ''


def repair_prompt(row,current,feedback=''):
    question = row['question'].removesuffix(SCIENCE_SUFFIX)
    unit = 'one option letter' if row['domain']=='science' else 'a number'
    review = f'\n\nIndependent review, which may contain errors:\n{feedback}' if feedback else ''
    return chat(
        f'{question}\n\nPrevious solution:\n{current["text"]}{review}\n\n'
        'Verify the reasoning and calculations against the question. Use any correct review information '
        'and reject any mistaken suggestion. Preserve a correct solution, and fix an incorrect solution. '
        f'Give at most 90 words of scratch work, then a line "Final: X" where X is {unit}.')


def repair(row,current,feedback='',variant='plain'):
    formatted = repair_prompt(row,current,feedback)
    seed = int(hashlib.sha256(f'{SEED}:{row["id"]}:repair'.encode()).hexdigest()[:8],16)%(2**31)
    start = time.monotonic()
    response = request('/api/generate',{
        'model':MODEL,'prompt':formatted,'raw':True,'stream':False,'logprobs':True,'keep_alive':'30m',
        'options':{'temperature':.65,'seed':seed,'num_predict':CAP,'num_ctx':4096}})
    if not response.get('done') or response.get('model')!=MODEL:
        raise RuntimeError('Incomplete repair response')
    text = response['response']
    return {'model':MODEL,'model_digest':MODEL_DIGEST,'text':text,'answer':parse_answer(row['domain'],text),
            'prompt_sha256':hashlib.sha256(formatted.encode()).hexdigest(),'finish':response['done_reason'],
            'input_tokens':int(response['prompt_eval_count']),'output_tokens':int(response.get('eval_count',0)),
            'confidence':confidence(text,response.get('logprobs',[]),row['domain']),
            'seed':seed,'temperature':.65,'latency_seconds':time.monotonic()-start,
            'feedback':feedback,'variant':variant,'output_cap':CAP}


def pilot(source,output):
    questions = {r['id']:r for r in read(source/'splits/train-questions.jsonl')}
    trees = {r['id']:r['tree'] for r in read(source/'train-trees.jsonl')}
    brief = {r['id']:r['judge'] for r in read(source/'train-judges.jsonl') if r['path']==''}
    deliberate = read(source/'judge-deliberate-pilot.jsonl')
    tags = {r['name']:r['digest'] for r in request('/api/tags')['models']}
    if tags.get(MODEL)!=MODEL_DIGEST:
        raise ValueError('Installed compact solver changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('a+',encoding='utf-8') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stream.seek(0)
        prior = [json.loads(line) for line in stream]
        known = {(r['id'],r['variant']) for r in prior}
        if len(known)!=len(prior):
            raise ValueError('Duplicate repair probes')
        for index,item in enumerate(deliberate,1):
            row = questions[item['id']]
            contexts = {'plain':'','brief':check_text(brief[row['id']]),'deliberate':check_text(item['judge'])}
            for variant,feedback in contexts.items():
                if (row['id'],variant) in known:
                    continue
                response = repair(row,trees[row['id']][''],feedback,variant)
                stream.write(json.dumps({'id':row['id'],'variant':variant,'repair':response})+'\n')
                stream.flush()
            if index%8==0 or index==len(deliberate):
                print(f'guided repair training pilot {index}/{len(deliberate)}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    pilot(args.source,args.output)


if __name__=='__main__':
    main()

