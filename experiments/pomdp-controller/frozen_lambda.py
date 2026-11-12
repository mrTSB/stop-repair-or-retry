"""Separate dev-selected lambda .248 experiment. Offline replay; no inference."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from stop_repair_retry import pomdp_experiment as p

LAMBDA=.248
DIAGNOSTIC=ROOT/'results/pomdp-lambda-diagnostic'
ORIGINAL=ROOT/'results/pomdp-two-state'


def untimed(records):return [{k:v for k,v in r.items() if k!='controller_cpu_seconds'} for r in records]

def source_hashes():
    paths=[Path(__file__),Path(p.__file__),Path(p.io.__file__),p.DESIGN,p.io.DEFAULT_DATA/'FILE_MANIFEST.json',
           ORIGINAL/'model.json',ORIGINAL/'frozen.json',DIAGNOSTIC/'protocol.json',DIAGNOSTIC/'dev-sweep.json',
           DIAGNOSTIC/'dev-candidate.json',DIAGNOSTIC/'dev-candidate-paths.json',p.A_RESULTS/'artifact-sha256.json']
    return {str(f.relative_to(ROOT)):p.io.sha(f) for f in paths}

def verify_selection(model):
    candidate=p.io.read_json(DIAGNOSTIC/'dev-candidate.json');sweep=p.io.read_json(DIAGNOSTIC/'dev-sweep.json')
    winner=min(sweep,key=lambda s:(-s['accuracy'],s['mean_tokens'],s['lambda']))
    if candidate!=winner or candidate['lambda']!=LAMBDA:raise ValueError('Candidate selection changed')
    rows=p.io.load_dataset(p.io.DEFAULT_DATA,'dev');saved=p.io.read_json(DIAGNOSTIC/'dev-candidate-paths.json')
    if untimed(saved)!=untimed(p.replay(model,rows,LAMBDA)):raise ValueError('Development candidate does not reproduce')
    return candidate,saved

def comparison(rows,records):
    methods={'pomdp_lambda_0.248':[{'id':r['id'],'domain':r['domain'],'correct':r['correct'],'tokens':r['solver_tokens']} for r in records]}
    methods.update({name:p.fixed_records(rows,paths) for name,paths in p.io.FIXED.items()})
    a=p.io.read_json(p.A_RESULTS/'test/controller-results.json')
    if [r['id'] for r in a]!=[r['id'] for r in rows]:raise ValueError('Controller A test question mismatch')
    methods['llm_controller_A']=a;summaries=[];paired=[]
    for method,rr in methods.items():
        for domain in ('all','math','science'):
            selected=[r for r in rr if domain=='all' or r['domain']==domain];scores=[r['correct'] for r in selected]
            summaries.append({'method':method,'domain':domain,'n':len(selected),'correct':sum(scores),'accuracy':statistics.mean(scores),
              'accuracy_ci95':p.io.interval(scores),'mean_tokens':statistics.mean(r['tokens'] for r in selected),
              'mean_utility_lambda_0.248':statistics.mean(r['correct']-LAMBDA*r['tokens']/1000 for r in selected),
              'mean_utility_reference_0.01':statistics.mean(r['correct']-.01*r['tokens']/1000 for r in selected)})
        if method!='pomdp_lambda_0.248':
            aa=methods['pomdp_lambda_0.248'];accuracy=[x['correct']-y['correct'] for x,y in zip(aa,rr)]
            item={'comparator':method,'accuracy_difference':statistics.mean(accuracy),'accuracy_difference_ci95':p.io.interval(accuracy)}
            for lam in (LAMBDA,.01):
                diffs=[x['correct']-y['correct']-lam*(x['tokens']-y['tokens'])/1000 for x,y in zip(aa,rr)]
                item[f'utility_difference_lambda_{lam}']={'mean':statistics.mean(diffs),'ci95':p.io.interval(diffs)}
            paired.append(item)
    return {'split':'test','n':len(rows),'planning_lambda':LAMBDA,'summaries':summaries,'paired_comparisons':paired,
      'action_sequences':dict(Counter(' -> '.join(s['action'] for s in r['steps']) for r in records)),
      'controller_cpu_seconds':sum(r['controller_cpu_seconds'] for r in records),
      'interpretation':'Lambda selected on 60 dev questions after viewing earlier test results. Descriptive follow-up on an already inspected test set, not a fresh confirmatory evaluation.',
      'llm_note':'Controller A decisions remain frozen from its lambda0.01 prompt. Utility at0.248 reweights that existing policy; it is not a lambda0.248-tuned LLM.',
      'uncertainty_note':'2000 paired question bootstrap resamples conditional on fixed cache/model; no selection/model-fit uncertainty; no multiple-comparison correction.'}

def collect(run):
    if run.exists():raise ValueError('Refuse existing run or automatic re-evaluation')
    model=p.check_run(ORIGINAL);candidate,dev=verify_selection(model);sources=source_hashes()
    run.mkdir(parents=True);shutil.copyfile(ORIGINAL/'model.json',run/'model.json')
    shutil.copyfile(DIAGNOSTIC/'dev-candidate-paths.json',run/'dev-evidence.json')
    shutil.copyfile(__file__,run/'frozen_lambda.frozen.py')
    p.io.save(run/'freeze.json',{'frozen_utc':p.io.now(),'planning_lambda':LAMBDA,'state_model':'unchanged two-state training fit',
      'source_hashes':sources,'model_sha256':p.io.sha(run/'model.json'),'dev_evidence_sha256':p.io.sha(run/'dev-evidence.json'),
      'selected_candidate':candidate,'user_choice':'Explicitly freeze lambda0.248 and evaluate the full test set as a separate experiment.',
      'selection':'602-point train/dev sweep; maximize dev accuracy, then minimize mean tokens, then minimize lambda. No new primary replacement.',
      'prior_test_exposure':True,'new_paid_calls':0,'comparison_utilities':[LAMBDA,.01],
      'llm_policy_lambda':.01,'no_model_refitting':True,'test_question_count':400})
    # These checks and the durable freeze are complete BEFORE loading any test rows.
    if source_hashes()!=sources:raise ValueError('Inputs changed during freeze')
    started=p.io.now();rows=p.io.load_dataset(p.io.DEFAULT_DATA,'test')
    if len(rows)!=400:raise ValueError('Expected complete published test set')
    records=p.replay(model,rows,LAMBDA);p.io.save(run/'test-paths.json',records)
    p.io.save(run/'summary.json',comparison(rows,records))
    p.io.save(run/'status.json',{'status':'complete','test_started_utc':started,'finished_utc':p.io.now(),'questions':400,'new_paid_calls':0})
    return verify(run)

def verify(run):
    freeze=p.io.read_json(run/'freeze.json');status=p.io.read_json(run/'status.json')
    if freeze['source_hashes']!=source_hashes() or freeze['planning_lambda']!=LAMBDA:raise ValueError('Frozen inputs changed')
    if freeze['model_sha256']!=p.io.sha(run/'model.json') or freeze['dev_evidence_sha256']!=p.io.sha(run/'dev-evidence.json'):raise ValueError('Evidence changed')
    model=p.check_run(ORIGINAL);candidate,dev=verify_selection(model)
    if p.io.read_json(run/'model.json')!=model or p.io.read_json(run/'dev-evidence.json')!=dev or freeze['selected_candidate']!=candidate:raise ValueError('Development/model binding failed')
    if not datetime.fromisoformat(freeze['frozen_utc'])<=datetime.fromisoformat(status['test_started_utc'])<=datetime.fromisoformat(status['finished_utc']):raise ValueError('Freeze chronology invalid')
    if status['status']!='complete' or status['questions']!=400:raise ValueError('Incomplete test run')
    rows=p.io.load_dataset(p.io.DEFAULT_DATA,'test');saved=p.io.read_json(run/'test-paths.json')
    if untimed(saved)!=untimed(p.replay(model,rows,LAMBDA)):raise ValueError('Test path/cost/belief reconstruction failed')
    if p.io.read_json(run/'summary.json')!=comparison(rows,saved):raise ValueError('Summary reconstruction failed')
    return {'status':'pass','questions':400,'planning_lambda':LAMBDA,'source_freeze_and_dev_binding':True,'replay_and_comparisons':True,'new_paid_calls':0}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['collect','verify']);ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    result=collect(args.run) if args.command=='collect' else verify(args.run)
    print(json.dumps(result,indent=2))
