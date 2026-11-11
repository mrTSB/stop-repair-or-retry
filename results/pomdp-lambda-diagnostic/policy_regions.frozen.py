"""Enumerate piecewise-linear lambda policy boundaries for this two-call model.

Uses model probabilities only for boundaries; evaluates resulting policies on dev only.
No test evaluation, model fitting, or inference. Floating-point ties use the frozen planner.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
import statistics
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lambda_sweep import p,evaluate,ROOT

ORDER=('stop','repair','retry')

def one_call_lines(model,b):
    return {'stop':(p.dot(b,model['terminal_reward']),0),**{a:(p.dot(p.predict(model,b,a),model['terminal_reward']),p.dot(b,model['cost_tokens'][a])/1000) for a in p.ACTIONS}}

def crossings(lines,left,right):
    points=[]
    for a,b in combinations(ORDER,2):
        u,c=lines[a];v,d=lines[b]
        if abs(c-d)>1e-15:
            x=(u-v)/(c-d)
            if left<x<right:points.append(x)
    return points

def best_line(lines,lam):
    best='stop'
    for a in p.ACTIONS:
        if lines[a][0]-lam*lines[a][1]>lines[best][0]-lam*lines[best][1]+p.CONFIG['tie_tolerance']:best=a
    return lines[best]

def root_lines(model,b,lam):
    lines={'stop':(p.dot(b,model['terminal_reward']),0)}
    for a in p.ACTIONS:
        u=0;c=p.dot(b,model['cost_tokens'][a])/1000
        pred=p.predict(model,b,a)
        for z in range(len(model['observations'])):
            weights=[x*o[z] for x,o in zip(pred,model['observation'][a])];prob=sum(weights)
            if prob:
                bnext=[w/prob for w in weights];v,d=best_line(one_call_lines(model,bnext),lam)
                u+=prob*v;c+=prob*d
        lines[a]=(u,c)
    return lines

def regions(model,maximum=1.2):
    roots=[p.initial_belief(model,z) for z in range(len(model['observations']))]
    child_points=[0.,maximum]
    for b in roots:
        for a in p.ACTIONS:
            for z in range(len(model['observations'])):
                child_points+=crossings(one_call_lines(model,p.update(model,b,a,z)),0,maximum)
    child_points=sorted(set(child_points));all_points=list(child_points)
    for left,right in zip(child_points,child_points[1:]):
        mid=(left+right)/2
        for b in roots:all_points+=crossings(root_lines(model,b,mid),left,right)
    boundaries=sorted(set(all_points))
    samples=set(boundaries)
    for left,right in zip(boundaries,boundaries[1:]):
        samples.add((left+right)/2)
    # Probe either side of floating-point/tie boundaries as an additional numerical check.
    for x in boundaries:
        for dx in (-1e-9,1e-9):
            if 0<=x+dx<=maximum:samples.add(x+dx)
    return boundaries,sorted(samples)

def run(output):
    if output.exists():raise ValueError('Refuse overwrite')
    model=p.io.read_json(ROOT/'results/pomdp-two-state/model.json');p.validate_model(model)
    boundaries,samples=regions(model);output.mkdir(parents=True)
    p.io.save(output/'protocol.json',{'created_utc':p.io.now(),'method':'Enumerate all one-call line intersections at each reachable posterior; on each resulting interval enumerate root action-line intersections. Evaluate boundaries, interval midpoints, and +/-1e-9 boundary probes.',
      'splits':['dev'],'test_evaluation':False,'lambda_range':[0,1.2],'boundaries':boundaries,'sample_lambdas':samples,
      'script_sha256':p.io.sha(__file__),'model_sha256':p.io.sha(ROOT/'results/pomdp-two-state/model.json'),
      'purpose':'Follow-up diagnostic to the finer grid; does not replace frozen primary experiment.',
      'numeric_note':'Piecewise-linear model arithmetic in floating point; frozen tie tolerance1e-12. Not an assertion of globally best model or generalization.'})
    rows=p.io.load_dataset(p.io.DEFAULT_DATA,'dev');summaries=[];policies={}
    for lam in samples:
        s,records=evaluate(model,rows,lam);summaries.append(s)
        if s['policy_signature'] not in policies:
            policies[s['policy_signature']]={'representative_lambda':lam,'paths':[r['path'] for r in records],
                'question_ids':[r['id'] for r in records],'correct':[r['correct'] for r in records],
                'solver_tokens':[r['solver_tokens'] for r in records],'accuracy':s['accuracy'],'mean_tokens':s['mean_tokens']}
    p.io.save(output/'dev-regions.json',summaries);p.io.save(output/'dev-policies.json',policies)
    candidate=min(summaries,key=lambda s:(-s['accuracy'],s['mean_tokens'],s['lambda']))
    p.io.save(output/'status.json',{'status':'complete','boundary_count':len(boundaries),'probes':len(samples),'unique_dev_policies':len(policies),'best_observed_dev':candidate,'finished_utc':p.io.now()})
    print(json.dumps({'boundaries':len(boundaries),'probes':len(samples),'policies':len(policies),'candidate':candidate}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();run(args.output)
