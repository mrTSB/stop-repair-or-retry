"""Replay baselines directly on frozen trees; no learned controller is loaded."""
import json,gzip,statistics,random,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
from scoring import correct
LAMBDA=.01
NAMES={'stop':'Submit initial','repair_once':'Repair once','repair':'Repair twice','retry_once':'Retry once','retry':'Retry twice'}
def load(split):
 with gzip.open(R/'data'/f'{split}-trees.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
def replay(row,policy):
 assert policy in NAMES
 t=row['tree'];paths=[''];actions=[];p=''
 for step in range(2):
  if policy=='stop' or (policy.endswith('_once') and step==1):action='stop'
  else:action='repair' if policy.startswith('repair') else 'retry'
  actions.append(action)
  if action=='stop':break
  p=p+('/' if p else '')+action;paths.append(p)
 selected=t[p]
 inp=sum(t[p]['input_tokens'] for p in paths);out=sum(t[p]['output_tokens'] for p in paths);ok=int(correct(selected,row['answer']))
 return {'id':row['id'],'domain':row['domain'],'policy':policy,'actions':actions,'visited_paths':paths,'answer':selected['answer'],'correct':ok,'initial_correct':int(t['']['correct']),'input_tokens':inp,'output_tokens':out,'tokens':inp+out,'utility':ok-LAMBDA*(inp+out)/1000,'judge_calls':0}
def ci(v):
 rng=random.Random(20261110);n=len(v);s=sorted(sum(v[rng.randrange(n)] for _ in range(n))/n for _ in range(2000));return [s[50],s[1949]]
def main():
 test=load('test');assert len(test)==400
 results=[replay(row,p) for p in NAMES for row in test];summaries=[]
 for domain in ['all','math','science']:
  for p in NAMES:
   rr=[x for x in results if x['policy']==p and (domain=='all' or x['domain']==domain)]
   wrong=[x for x in rr if not x['initial_correct']];right=[x for x in rr if x['initial_correct']]
   summaries.append({'domain':domain,'policy':p,'n':len(rr),'accuracy':statistics.mean(x['correct'] for x in rr),'accuracy_ci95':ci([x['correct'] for x in rr]),'mean_tokens':statistics.mean(x['tokens'] for x in rr),'mean_utility':statistics.mean(x['utility'] for x in rr),'wrong_to_right_conditional':statistics.mean(x['correct'] for x in wrong),'right_to_wrong_conditional':statistics.mean(not x['correct'] for x in right)})
 # Verify fixed paths and token totals directly from solver nodes, without controller artifacts.
 expected={'stop':[''],'repair_once':['','repair'],'repair':['','repair','repair/repair'],'retry_once':['','retry'],'retry':['','retry','retry/retry']}
 by_id={row['id']:row for row in test}
 for x in results:
  paths=expected[x['policy']];assert x['visited_paths']==paths
  nodes=[by_id[x['id']]['tree'][path] for path in paths]
  assert x['answer']==nodes[-1]['answer']
  assert x['correct']==int(correct(nodes[-1],by_id[x['id']]['answer']))
  assert x['tokens']==sum(n['input_tokens']+n['output_tokens'] for n in nodes)
  assert x['judge_calls']==0
 split_rows={split:load(split) for split in ['train','dev','test']}
 split_ids={split:{row['id'] for row in rr} for split,rr in split_rows.items()}
 for split,n in [('train',100),('dev',60),('test',400)]:
  assert len(split_rows[split])==len(split_ids[split])==n
  assert sum(row['domain']=='math' for row in split_rows[split])==n//2
 for a,b in [('train','dev'),('train','test'),('dev','test')]:assert not split_ids[a]&split_ids[b]
 (R/'per-question.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in results))
 payload={'scope':'Reproducible cached-tree dataset and empirical characterization using five fixed baselines; no controller contribution claimed or controller outputs used.','test_questions':400,'baseline_fitting':'None; all five rules are fixed.','penalty':LAMBDA,'summaries':summaries,'source_sha256':{s:hashlib.sha256((R/'data'/f'{s}-trees.jsonl.gz').read_bytes()).hexdigest() for s in ['train','dev','test']},'validation':'All five fixed paths, submitted answers, correctness and token totals verified directly against solver nodes; split sizes, domain balance and question-ID disjointness verified. No controller result files or fitted objects read.','additional_paid_generation':False}
 (R/'results.json').write_text(json.dumps(payload,indent=2))
 lines=['# Cached tree dataset and empirical characterization','','We present a reproducible dataset of cached reasoning trajectories and characterize its accuracy and token-cost landscape using five fixed baselines on **400 test questions: 200 math and 200 science**. The baselines are descriptive reference points, not proposed methods. This report makes no controller contribution or controller-performance claim.','']
 lines += ['## Dataset and split', '', 'The contribution at this stage is the reproducible cached-tree dataset and its empirical characterization. Underlying questions come from GSM8K and MMLU-Pro; the new artifact is their generated response trees, provenance, scoring and controlled split, not newly authored benchmark questions.', '', '| Split | Math | Science | Total | Purpose |', '|---|---:|---:|---:|---|', '| Training | 50 | 50 | 100 | Future controller fitting |', '| Development | 30 | 30 | 60 | Future tuning and design choices |', '| Test | 200 | 200 | 400 | Reported baseline evaluation |', '', 'Splits are disjoint by question ID, with frozen source hashes and documented duplicate checks. Every question and all seven of its cached solver nodes stay together in one split. These five fixed baselines require no training or development tuning. There are 560 complete trees and 3,920 solver responses in the contribution dataset. This bundle contains solver responses only. No fitted controllers, judge decisions or controller-result files are used in this analysis.', '']
 lines += ['## Dataset construction and scope', '', 'The cache uses Qwen/Qwen2.5-3B-Instruct at pinned revision aa8e72537993ba99e69dfaafa59ed015b17504d1, fixed action prompts, temperature 0.7, top-p 0.9 and a 768-token solver output cap. Each question has an initial response and every repair/retry branch through two additional calls. Repair sees the question and immediate preceding response; retry solves the original question in a fresh context. All five baselines share these same cached solver outputs.', '', 'The measured landscape depends jointly on the sampled questions, frozen solver, prompts, decoding settings and one sampled tree per question. These findings are not properties of the questions alone, nor evidence about adaptive decision making. Math questions come from GSM8K; science questions cover MMLU-Pro biology, chemistry and physics. Exact-ID and normalized-stem duplicate checks and source hashes are documented in the dataset manifest; these checks do not establish absence of semantic duplicates or model pretraining exposure.', '', '## Main characterization', '', 'Repair once raises observed accuracy from 36.25% to 52.75%; a second repair reaches 54.75%, with another 525 tokens per question on average. Fresh retries produce smaller aggregate gains on this cache. These descriptive differences establish an accuracy–cost reference for future work and do not introduce a new algorithm.', '']
 for domain in ['all','math','science']:
  lines += [f'## {domain.title()} questions','','| Baseline | Accuracy | 95% accuracy interval | Mean tokens | Mean utility |','|---|---:|---:|---:|---:|']
  for x in summaries:
   if x['domain']==domain:lines.append(f"| {NAMES[x['policy']]} | {100*x['accuracy']:.2f}% | {100*x['accuracy_ci95'][0]:.1f}–{100*x['accuracy_ci95'][1]:.1f}% | {x['mean_tokens']:.0f} | {x['mean_utility']:.4f} |")
  lines+=['']
 lines += ['## Definitions and interpretation','','Repair once checks the initial solution once and submits that response. Repair twice checks it twice and submits the latest response. Retry once/twice independently solve the original question and submit the latest response; they do not select the best answer. All baselines allow at most two extra solver calls.','','Tokens include every visited solver input and output, including the initial response; there are no judge tokens. Utility is correctness minus 0.01 times tokens/1,000. Intervals use 2,000 question bootstrap resamples on one fixed cache. They do not cover repeated model-generation seeds. The source questions and cache remain unchanged; malformed responses stay in the denominator.','','Controller development and evaluation are deferred. These five fixed strategies describe the existing dataset landscape and require no fitting or tuning. The test outcomes have already been examined; use train/development data for future choices and a fresh held-out set if test results inform redesign.','','Per-question paths and costs: `per-question.jsonl`. Full precision summaries and source hashes: `results.json`. Reproduce locally with `python3 results/cached-tree-dataset/run_baselines.py`.']
 (R/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps([x for x in summaries if x['domain']=='all'],indent=2))
if __name__=='__main__':main()
