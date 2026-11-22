"""Independent aggregate and dynamic-programming audit of composed DEV results.

Uses only sealed saved outcomes/settings. No model load, forward, fitting,
network call or test file is needed. The previous checkpoint stays immutable.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import c80_targets

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SUBJECTS = ('math','chemistry','physics')
COUNTS = (128,64,64)
BASELINES = ('Stop','Repair once','Repair twice','Retry once','Retry twice')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def stats(items):
    if len(items) != 256 or len({r['id'] for r in items}) != 256:
        raise ValueError('Need all256 unique selected outcomes')
    if any(r['correct'] not in (0,1,False,True) or not math.isfinite(r['compute']) or r['compute'] <= 0 for r in items):
        raise ValueError('Invalid binary outcome or inference cost')
    return {'n':256,'correct':sum(int(r['correct']) for r in items),
        'accuracy':sum(int(r['correct']) for r in items)/256,'mean_compute':math.fsum(r['compute'] for r in items)/256}


def equal(a,b):
    for key in ('n','correct','accuracy','mean_compute'):
        if not math.isclose(a[key],b[key],rel_tol=1e-12,abs_tol=1e-9):
            raise ValueError('Independent aggregate differs: '+key)


def optimum(candidates,ceiling):
    # Independently check the registered Pareto-product selector by dynamic
    # programming over total integer correct counts and additive compute.
    states = {0:(0.,())}
    for subject,count in zip(SUBJECTS,COUNTS):
        next_states = {}
        for correct,(cost,keys) in states.items():
            for candidate in candidates:
                value = candidate['subjects'][subject]
                total = correct+value['correct']
                point = (cost+count*value['mean_compute'],keys+(candidate['key'],))
                if total not in next_states or point < next_states[total]: next_states[total] = point
        states = next_states
    valid = [(correct,cost/256,keys) for correct,(cost,keys) in states.items() if cost/256 < ceiling]
    if not valid: raise ValueError('No hybrid is cheaper than matched B')
    return min(valid,key=lambda v:(-v[0],v[1],v[2]))


def audit(directory):
    done = read(directory/'complete.json')
    spec = read(directory/'registration.json')
    if done['registration_sha256'] != sha(directory/'registration.json') or done['test_used']:
        raise ValueError('Changed or test-contaminated experiment completion')
    evidence = {str(directory/'complete.json'):sha(directory/'complete.json')}
    for name,digest in done['artifacts'].items():
        path = directory/name
        if path.resolve().parent != directory.resolve() or sha(path) != digest:
            raise ValueError('Changed completed composition artifact: '+name)
        evidence[str(path)] = digest
    for path,digest in {**spec['source_files'],**spec['input_evidence']}.items():
        if sha(path) != digest: raise ValueError('Changed registered source/input: '+path)
    summary = read(directory/'summary.json')
    if summary['test_used'] or summary['fresh_validation'] or not summary['all_selected_experts_replayed_exactly']:
        raise ValueError('Require the cache-only reused-development comparison')
    items = read(directory/'private-selected-items.json')
    equal(stats(items['B']),summary['B']['dev'])
    equal(stats(items['C']),summary['C_candidate']['dev'])
    if {r['id'] for r in items['B']} != {r['id'] for r in items['C']}:
        raise ValueError('B/C question identities differ')
    candidates = read(directory/'candidate-subject-metrics.json')
    if len(candidates['B']) != 36 or len(candidates['C']) != 156:
        raise ValueError('Missing candidate rules/experts')
    b_choices = {subject:min(candidates['B'],key=lambda c:(-c['subjects'][subject]['correct'],c['subjects'][subject]['mean_compute'],c['key']))
                 for subject in SUBJECTS}
    b_correct = sum(b_choices[s]['subjects'][s]['correct'] for s in SUBJECTS)
    b_cost = math.fsum(n*b_choices[s]['subjects'][s]['mean_compute'] for s,n in zip(SUBJECTS,COUNTS))/256
    equal({'n':256,'correct':b_correct,'accuracy':b_correct/256,'mean_compute':b_cost},summary['B']['dev'])
    for subject in SUBJECTS:
        if summary['B']['settings']['by_subject'][subject] != b_choices[subject]['settings']:
            raise ValueError('B subject rule is not the genuine accuracy winner')
    best_correct,best_cost,_ = optimum(candidates['C'],b_cost)
    equal({'n':256,'correct':best_correct,'accuracy':best_correct/256,'mean_compute':best_cost},summary['C_candidate']['dev'])
    canonical_dir = Path(spec['paths']['canonical'])
    canonical_spec = read(canonical_dir/'registration.json')
    model_dir = Path(canonical_spec['paths']['source'])/'models'
    for expert in summary['C_candidate']['experts'].values():
        path = model_dir/(expert['model']+'.pkl')
        if sha(path) != expert['model_sha256']: raise ValueError('Changed expert parent model')
        evidence[str(path)] = expert['model_sha256']
    source_points = {}
    with (canonical_dir/'private-paired-outcomes.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            source_points[r['candidate_index'],r['id']] = r
    for item in items['C']:
        source = source_points[item['subject_expert_candidate'],item['id']]
        if item['path'] != source['path'] or item['correct'] != source['correct'] or not math.isclose(item['compute'],source['canonical_compute'],rel_tol=1e-12,abs_tol=1e-9):
            raise ValueError('Selected hybrid outcome differs from its actual parent')
    a = read(Path(spec['paths']['a_reference'])/'summary.json')
    rows = [{'method':name,**a['baselines'][name]} for name in BASELINES]
    rows += [{'method':'A',**a['selected_A']['dev']},{'method':'B',**summary['B']['dev']},{'method':'C',**summary['C_candidate']['dev']}]
    if rows != summary['rows'] or c80_targets.assess(rows) != summary['targets']:
        raise ValueError('Changed headline table or exact-count target checks')
    if summary['B']['dev']['correct'] < summary['prior_B']['dev']['correct']:
        raise ValueError('Weakened B')
    if summary['C_candidate']['dev']['correct'] < summary['prior_C']['dev']['correct']:
        raise ValueError('Keep the previous more accurate C')
    for path in (Path(__file__),Path(c80_targets.__file__)):
        evidence[str(path)] = sha(path)
    starting = read(HERE/'current_checkpoint.json')
    return {'schema':'independently-audited-subject-expert-DEV-checkpoint-v1',
        'split':'Repeatedly reused256-question development; exploratory selection, not fresh confirmation',
        'rows':rows,'targets':summary['targets'],'fresh_validation':False,'fresh_test_reserved':1024,
        'solver':starting['solver'],'reviewer':starting['reviewer'],
        'identities':{'A':summary['A'],'B':summary['B'],'C':summary['C_candidate']},
        'compute_proxy':'(4*solver_tokens +3*reviewer_tokens +observer_billions*actual_token_slots)/3.09; full35B sensitivity retained in rows; not measured FLOPs',
        'audit':{'all_complete_artifact_hashes':True,'source_and_input_hashes':True,'individual_B_C_aggregates':True,
            'B_same_subject_flexibility':True,'C_selection_independent_dynamic_program':True,
            'all_C_parent_outcomes_and_model_hashes':True,'runtime_selected_experts_replayed_without_neural_calls':True},
        'prior_B':summary['prior_B'],'prior_C':summary['prior_C'],
        'C_compute_saved_vs_current_B':1-summary['C_candidate']['dev']['mean_compute']/summary['B']['dev']['mean_compute'],
        'C_compute_saved_vs_prior_C':1-summary['C_candidate']['dev']['mean_compute']/summary['prior_C']['dev']['mean_compute'],
        'evidence':{str(Path(path).resolve().relative_to(ROOT)):digest for path,digest in evidence.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('seal','verify'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    if args.phase == 'seal':
        with args.output.open('x') as stream: stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    elif read(args.output) != result: raise ValueError('Changed public checkpoint or source results')
    print(json.dumps({'verified':True,'rows':[{k:r[k] for k in ('method','n','correct','accuracy','mean_compute')} for r in result['rows']],
                      'targets':result['targets']},indent=2))


if __name__ == '__main__': main()
