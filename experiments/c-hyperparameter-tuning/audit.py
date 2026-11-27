"""Reload the saved tuned checkpoint and audit allowed changes and paid replay."""
import copy,json,pickle,sys
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/controller-semantic-hillclimb/followup/round2'))
sys.path.insert(0,str(ROOT/'scripts/reproduction'))
import development_bayesian_hybrid_v3 as runner
import audit_bayesian_hybrid as audit
import replay_latest as replay
from threadpoolctl import threadpool_limits
out=Path(sys.argv[1]);io=runner.io;original=runner.original;policy=original.policy
with original.prior.guard.no_test_reads(),original.prior.guarded.no_model_calls(),threadpool_limits(limits=1):
    done=io.read(out/'complete.json');io.verify({str(out/k):v for k,v in done['digests'].items()})
    result=io.read(out/'summary.json');folder=ROOT/'runs/controller-semantic-followup/round2/development-bayesian-model-hybrid-v3'
    spec=io.read(folder/'registration.json');base=io.read(folder/'summary.json')
    with runner.configured():bundle,_,_,record=original.inputs(SimpleNamespace(previous=Path(spec['previous']),output=folder))
    assert record==spec
    rows,trees,judges,_,caches,*_=bundle
    initial=pickle.loads(Path(base['C_candidate']['model_path']).read_bytes())
    model=pickle.loads((out/'hybrid.pkl').read_bytes());reverted=copy.deepcopy(model)
    for s in initial.experts:
        reverted.experts[s]['settings']=initial.experts[s]['settings']
        key=initial.experts[s]['model']
        if isinstance(initial.models[key],policy.Planner):reverted.models[key].weights=initial.models[key].weights
    assert pickle.dumps(reverted,protocol=5)==pickle.dumps(initial,protocol=5),'Changes outside existing hyperparameters'
    expected={v['id']:v for v in io.read(out/'per-question.json')};items=[]
    with ExitStack() as stack:
        for child in model.models.values():
            for parent in child.models if isinstance(child,policy.Planner) else [child]:stack.enter_context(original.canonical.cached_model(parent,caches))
        for row in rows:
            item=policy.routed(row,trees[row['id']],judges,model,{},'C');replay.same(item,expected[row['id']])
            observed,acquired=replay.paid(item,trees[row['id']],judges)
            blind=policy.routed({**row,'answer':''},observed,acquired,model,{},'C');replay.same(blind,item,grade=False)
            assert blind['decisions']==item['decisions'];items.append(item)
    measured=audit.check_items(items,rows,trees,judges);assert measured['correct']==result['tuned_C']['correct']
    audit.close(measured['mean_compute'],result['tuned_C']['mean_compute'])
    report={'passed':True,'saved_checkpoint_reloaded':True,'only_existing_hyperparameters_changed':True,
        'exact_original_model_recovered_after_reverting_settings_and_prior_weights':True,
        'direct_replays':256,'blinded_paid_replays':256,'independent_correctness_and_billing':True,
        'metrics':measured,'test_used':False,'new_model_calls':0,'source_sha256':io.sha(__file__),
        'checkpoint_sha256':io.sha(out/'hybrid.pkl')}
    (out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
