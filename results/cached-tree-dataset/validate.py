"""Validate the published solver-only dataset and independent fixed-path accounting."""
import gzip,hashlib,json,re
from pathlib import Path
from scoring import correct,parse
R=Path(__file__).resolve().parent
PATHS=['','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry']
EXPECTED={'stop':[''],'repair_once':['','repair'],'repair':['','repair','repair/repair'],'retry_once':['','retry'],'retry':['','retry','retry/retry']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=json.loads((R/'FILE_MANIFEST.json').read_text())
 for name,info in manifest['files'].items():
  p=R/name;assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],name
 selection=json.loads((R/'provenance/selection-manifest.json').read_text());generation=json.loads((R/'provenance/generation.json').read_text());constants=generation['constants']
 split_rows={};ids={};science_stems=set();quality={}
 for split,count in [('train',100),('dev',60),('test',400)]:
  qp=R/'data'/f'{split}-questions.jsonl';assert sha(qp)==selection['data_sha256'][split]
  qs=[json.loads(x) for x in qp.read_text().splitlines()]
  with gzip.open(R/'data'/f'{split}-trees.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(x) for x in f]
  assert len(qs)==len(rows)==count
  assert [q['id'] for q in qs]==[r['id'] for r in rows]
  ids[split]={q['id'] for q in qs};assert len(ids[split])==count
  assert sum(r['domain']=='math' for r in rows)==count//2
  quality[split]={'questions':count,'solver_nodes':count*7,'invalid_answers':0,'missing_confidence':0,'length_finishes':0}
  for q,row in zip(qs,rows):
   assert all(row[k]==v for k,v in q.items());assert row['split']==split and set(row['tree'])==set(PATHS)
   assert not {'judge_penalty','judges','controller'}&set(row)
   if row['domain']=='science':
    stem=re.sub(r'\W+','',row['question'].split('\n\nOptions:')[0]).casefold()
    assert stem not in science_stems;science_stems.add(stem)
   for path,n in row['tree'].items():
    assert not {'judges','judge','controller'}&set(n)
    obj,status=parse(n['text']);answer='' if obj.get('answer') is None else str(obj['answer']).strip()
    valid=bool(answer) and (row['domain']=='math' or len(answer)==1 and answer.upper() in 'ABCDEFGHIJ')
    conf=str(obj.get('confidence','')).lower();cv=conf in ['low','medium','high']
    assert (n['answer'],n['valid'],n['answer_parse_valid'],n['parse_status'])==(answer,valid,valid,status)
    assert (n['confidence'],n['confidence_valid'])==(conf if cv else 'missing',cv)
    assert n['correct']==correct(n,row['answer'])
    assert n['input_tokens']>0 and 0<n['output_tokens']<=768
    action=path.rsplit('/',1)[-1] if path else 'initial';parent=path.rsplit('/',1)[0] if '/' in path else ''
    assert n['path']==path and n['action']==action and n['parent_path']==(parent if path else None)
    expected=constants['REPAIR'].format(question=row['question'],solution=row['tree'][parent]['text']) if action=='repair' else constants['SOLVER'].format(question=row['question'])
    assert n['prompt']==expected and expected in n['formatted_prompt']
    quality[split]['invalid_answers']+=not valid;quality[split]['missing_confidence']+=not cv;quality[split]['length_finishes']+=n['finish_reason']=='length'
  split_rows[split]=rows
 for a,b in [('train','dev'),('train','test'),('dev','test')]:assert not ids[a]&ids[b]
 for phase,rows in [('calibration',split_rows['train']+split_rows['dev']),('test',split_rows['test'])]:
  for start in range(0,len(rows),16):
   batch=rows[start:start+16];batch_ids=[r['id'] for r in batch]
   for path in PATHS:
    seed=int.from_bytes(hashlib.sha256(f"{constants['SEED']}:{phase}:solver:{path}:{batch_ids}".encode()).digest()[:4],'big')
    for i,row in enumerate(batch):
     n=row['tree'][path];assert (n['batch_seed'],n['batch_position'],n['batch_size'])==(seed,i,len(batch))
 test={r['id']:r for r in split_rows['test']};seen=set();records=[json.loads(x) for x in (R/'per-question.jsonl').read_text().splitlines()]
 assert len(records)==2000
 for v in records:
  key=(v['policy'],v['id']);assert key not in seen;seen.add(key)
  paths=EXPECTED[v['policy']];assert v['visited_paths']==paths
  row=test[v['id']];nodes=[row['tree'][p] for p in paths]
  assert v['answer']==nodes[-1]['answer'] and v['correct']==int(correct(nodes[-1],row['answer']))
  assert v['tokens']==sum(n['input_tokens']+n['output_tokens'] for n in nodes)
  assert v['judge_calls']==0 and abs(v['utility']-(v['correct']-.01*v['tokens']/1000))<1e-12
 results=json.loads((R/'results.json').read_text());assert len(results['summaries'])==15
 for s in results['summaries']:
  rr=[v for v in records if v['policy']==s['policy'] and (s['domain']=='all' or v['domain']==s['domain'])]
  assert s['n']==len(rr) and abs(s['accuracy']-sum(v['correct'] for v in rr)/len(rr))<1e-12
  assert abs(s['mean_tokens']-sum(v['tokens'] for v in rr)/len(rr))<1e-9
 print(json.dumps({'status':'pass','quality':quality,'baselines':5,'test_question_baseline_results':2000,'checks':['file hashes','split separation','source questions','raw response parsing and grading','parent prompts','original batch seeds','solver-only export','fixed baseline paths and token totals','summary aggregates']},indent=2))
if __name__=='__main__':main()
