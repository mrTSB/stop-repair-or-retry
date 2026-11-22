"""Compose existing C experts by observed subject; give B the same flexibility.

Exploratory selection on the same reused DEV panel. All156 genuine C policies
and all36 independently prompted B rules remain available. No model training,
encoder forward, cloud call or fresh-test access is authorized by this file.
"""
import argparse
from itertools import product
import math
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from threadpoolctl import threadpool_limits

import expanded_c_canonical_cost as canonical
import concrete_checkpoint as checkpoint
import subject_expert_selection as selection
import subject_expert_hybrid as policy
import c80_targets
import development_completion_blend as guard
import training_sparse_text as guarded
from development_mixed_mentor import development_view

io = canonical.io
SCHEMA = 'observed-subject-C-experts-and-matched-B-full-DEV-v1'


def same(actual,expected):
    for key in ('n','correct','accuracy','mean_compute'):
        if not math.isclose(actual[key],expected[key],rel_tol=1e-12,abs_tol=1e-9):
            raise ValueError('Changed completed statistics: '+key)


def inputs(args):
    if checkpoint.read(checkpoint.MANIFEST) != checkpoint.build():
        raise ValueError('The concrete starting checkpoint changed')
    done = io.read(args.canonical/'complete.json')
    if done != canonical.seal_payload(args.canonical): raise ValueError('Changed canonical comparison seal')
    old_spec = io.read(args.canonical/'registration.json')
    old_args = SimpleNamespace(**{k:Path(v) for k,v in old_spec['paths'].items()})
    bundle = canonical.inputs(old_args)
    if canonical.contract(old_args,bundle) != old_spec: raise ValueError('Changed original canonical replay inputs')
    io.verify(old_spec['source_files']);io.verify(old_spec['evidence'])
    rows,_,trees,judges,_,models,caches,*_ = bundle
    subject = {r['id']:r['subject'] for r in rows}
    candidates = io.read(args.canonical/'search.json')
    by_index = {c['original_candidate_index']:c for c in candidates}
    if len(by_index) != 156 or set(by_index) != set(range(156)):
        raise ValueError('Retain all156 completed C candidates')
    paired = {i:{} for i in by_index}
    for item in canonical.previous.common.strict_records(args.canonical/'private-paired-outcomes.jsonl'):
        index,qid = item['candidate_index'],item['id']
        if (index not in paired or qid not in subject or qid in paired[index]
                or not item['same_decisions'] or not item['same_encoder_ledger']):
            raise ValueError('Changed/duplicate canonical per-question outcome')
        paired[index][qid] = {**item,'compute':item['canonical_compute'],'subject':subject[qid]}
    cs = []
    for index,candidate in sorted(by_index.items()):
        if candidate['policy_module'] not in policy.MODULES or set(paired[index]) != set(subject):
            raise ValueError('Each expert must be a complete genuine hybrid')
        same(selection.metrics(list(paired[index].values())),candidate['dev'])
        subjects = {s:selection.metrics([r for r in paired[index].values() if r['subject'] == s]) for s in selection.SUBJECTS}
        cs.append({'key':f'C{index:03d}','candidate':candidate,'subjects':subjects})
    b_summary = io.read(args.b_reference/'summary.json')
    b_rows,b_trees,b_judges = development_view(old_args.data,args.b_reference/'dev-observations.jsonl')
    if {r['id']:r for r in b_rows} != {r['id']:r for r in rows}:
        raise ValueError('B and C must use the exact same256 questions/references')
    b_current = [policy.llm.policy(r,b_trees[r['id']],b_judges,b_summary['B']['settings']) for r in b_rows]
    same(policy.llm.metrics(b_current),b_summary['B']['dev'])
    a = io.read(args.a_reference/'summary.json')
    if a['selected_A']['model_sha256'] != checkpoint.read(checkpoint.MANIFEST)['identities']['A']['model_sha256']:
        raise ValueError('Retain the strongest completed A')
    evidence = {**old_spec['evidence'],
        **{str((args.canonical/k).resolve()):v for k,v in done['files'].items()},
        str((args.canonical/'complete.json').resolve()):io.sha(args.canonical/'complete.json'),
        str(checkpoint.MANIFEST):io.sha(checkpoint.MANIFEST),
        **{str((args.b_reference/k).resolve()):io.sha(args.b_reference/k) for k in ('summary.json','search.json','identity.json','copy.json','dev-observations.jsonl')},
        str((args.a_reference/'complete.json').resolve()):io.sha(args.a_reference/'complete.json'),
        str((args.a_reference/'summary.json').resolve()):io.sha(args.a_reference/'summary.json')}
    record = {'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{k:str(getattr(args,k).resolve()) for k in ('canonical','b_reference','a_reference','output')},
        'development_questions':256,'subjects':selection.COUNTS,'C_candidates':156,'B_candidates':36,
        'B_settings':[{'root_stop':root,'second_stop':stop,'second_action':action}
            for root,stop,action in product(policy.llm.ROOT_STOPS,policy.llm.SECOND_STOPS,policy.llm.SECOND_ACTIONS)],
        'selection':'B: highest accuracy then lower cost per subject. C: exact Cartesian product of per-subject Pareto frontiers, highest accuracy then lower cost strictly below that matched B.',
        'route_visibility':'Only observed subject; each C expert retains genuine POMDP+LLM decisions and exact compute ledger',
        'no_label_inference_routing':True,'fresh_validation':False,'test_used':False,'new_cloud_calls':0,'new_neural_calls':0,
        'baseline_definitions_unchanged':True,'strongest_A_retained':True,'strongest_matched_B_never_nerfed':True,
        'exploratory_selection_on_reused_DEV':True,'not_a_preregistered_fresh_result':True,
        'final_C_target':'>80% and >=10pp above the strongest A/B, with lower inference compute than B'}
    return rows,trees,judges,models,caches,cs,paired,b_rows,b_trees,b_judges,b_summary,a,record


def replay(args,rows,trees,judges,models,caches,cs,paired,b_rows,b_trees,b_judges,b_summary,a,record):
    bs,b_items = [],{}
    for index,setting in enumerate(record['B_settings']):
        items = [policy.llm.policy(r,b_trees[r['id']],b_judges,setting) for r in b_rows]
        key = f'B{index:02d}'
        b_items[key] = {v['id']:v for v in items}
        subjects = {s:selection.metrics([v for r,v in zip(b_rows,items) if r['subject'] == s]) for s in selection.SUBJECTS}
        bs.append({'key':key,'settings':setting,'subjects':subjects})
    b_choices,b_value = selection.best_B(bs)
    if b_value['correct'] < b_summary['B']['dev']['correct']:
        raise ValueError('Never weaken B to improve the gap')
    b_settings = {'by_subject':{s:b_choices[s]['settings'] for s in selection.SUBJECTS}}
    selected_b_items = [policy.evaluate_B(r,b_trees[r['id']],b_judges,b_settings) for r in b_rows]
    same(selection.metrics(selected_b_items),b_value)
    selected,frontiers = selection.best_C(cs,b_value['mean_compute'])
    if selected is None: raise ValueError('No C satisfies the stronger matched-B compute ceiling; preserve all prior controllers')
    c_choices,c_value = selected
    experts = {s:c_choices[s]['candidate'] for s in selection.SUBJECTS}
    source = Path(io.read(args.canonical/'registration.json')['paths']['source'])
    keys = {v['model'] for v in experts.values()}
    raw_models = {key:pickle.loads((source/'models'/(key+'.pkl')).read_bytes()) for key in sorted(keys)}
    identities = {s:{k:v[k] for k in ('model','policy_module','settings','model_sha256','original_candidate_index')} for s,v in experts.items()}
    model = policy.Planner(raw_models,identities,{'schema':SCHEMA,'route_visibility':'Observed subject only',
        'parent_models':{k:io.sha(source/'models'/(k+'.pkl')) for k in keys},
        'genuine_POMDP_experts':True,'B_given_same_subject_flexibility':True,
        'new_training':False,'DEV_used_for_selection':True,'test_used':False,'fresh_validation':False})
    # Save fresh unloaded parents, before attaching any DEV neural cache.
    target = args.output/'hybrid.pkl'
    io.atomic_bytes(target,pickle.dumps(model,protocol=5))
    items = {}
    for key,child in model.models.items():
        with canonical.cached_model(child,caches):
            for row in rows:
                expert = experts[row['subject']]
                if expert['model'] != key: continue
                item = policy.evaluate(row,trees[row['id']],judges,model,{},'C')
                expected = paired[expert['original_candidate_index']][row['id']]
                if item['path'] != expected['path'] or item['correct'] != expected['correct']:
                    raise ValueError('Composed runtime changed a selected expert decision')
                if not math.isclose(item['compute'],expected['compute'],rel_tol=1e-12,abs_tol=1e-9):
                    raise ValueError('Composed runtime changed a paid observer/solver/reviewer cost')
                if item['solver_calls'] > 3 or item['judge_calls'] > 1: raise ValueError('Changed controller horizon')
                items[row['id']] = item
    c_items = [items[r['id']] for r in rows]
    same(selection.metrics(c_items),c_value)
    if not any(v['judge_calls'] for v in c_items): raise ValueError('C must actually consult the LLM')
    c_result = {'policy_module':'subject_expert_hybrid','model_path':str(target.resolve()),'model_sha256':io.sha(target),
        'settings':{},'experts':identities,'dev':canonical.previous.measured(c_items)}
    b_result = {'policy_module':'subject_expert_hybrid.evaluate_B','family':'B','settings':b_settings,
                'dev':policy.llm.metrics(selected_b_items),'subject_rules':{s:b_choices[s]['key'] for s in selection.SUBJECTS}}
    table = [{'method':name,**a['baselines'][name]} for name in checkpoint.BASELINES]
    table += [{'method':'A',**a['selected_A']['dev']},{'method':'B',**b_result['dev']},{'method':'C',**c_result['dev']}]
    result = {'schema':SCHEMA,'A':a['selected_A'],'B':b_result,'C_candidate':c_result,
        'prior_B':b_summary['B'],'prior_C':io.read(args.canonical/'summary.json')['canonical_selection']['selected_C'],
        'rows':table,'targets':c80_targets.assess(table),'pareto_frontier_sizes':frontiers,
        'all_selected_experts_replayed_exactly':True,'all_five_baselines_retained':True,
        'B_subject_specialization_same_capability':True,'development_questions':256,'test_used':False,
        'fresh_validation':False,'new_neural_calls':0,'new_cloud_calls':0,
        'candidate_only_until_explicit_result_audit':True}
    io.write(args.output/'summary.json',result)
    io.write(args.output/'private-selected-items.json',{'B':selected_b_items,'C':c_items})
    io.write(args.output/'candidate-subject-metrics.json',{'B':bs,'C':cs})
    io.verify(record['source_files']);io.verify(record['input_evidence'])
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'artifacts':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},'test_used':False})
    print({'B':b_result['dev'],'C_candidate':c_result['dev'],'targets':result['targets'],'experts':identities},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','replay'))
    for name in ('canonical','b-reference','a-reference','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    with guard.no_test_reads(),guarded.no_model_calls(),threadpool_limits(limits=1):
        bundle = inputs(args)
        record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True,exist_ok=False)
            io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'C_candidates':156,'B_candidates':36},flush=True)
            return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed subject composition contract')
        with io.lock(args.output/'.replay.lock'):
            if any(p.name not in ('registration.json','.replay.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior outputs')
            replay(args,*bundle)


if __name__ == '__main__': main()
