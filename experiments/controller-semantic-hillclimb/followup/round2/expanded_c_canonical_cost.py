"""Registered canonical terminal-cost sensitivity for all156 expanded Cs.

Uniformly use the earlier retained-path-audit retry/reuse receipt wherever a
new expanded-study request exactly duplicated it. Original results stay sealed.
Replay uses existing neural arrays only; missing values, NN loads/forwards and
new solver/reviewer calls fail. Decisions, answers and encoder ledgers must be
identical between original and canonical terminal receipts.
"""
import argparse
from contextlib import contextmanager, ExitStack
import copy
import hashlib
from pathlib import Path
import pickle
import sys
from unittest.mock import patch

import numpy as np

import development_expanded_mentor as previous
import science_option_observer as option
import development_retained_path_audit as canonical_protocol
from prospective_parallel import RequestJournal, raw_coverage
from prospective_freeze import observer_slots

io=previous.io
perception=previous.feature_cache.original.perception
SCHEMA='round2-expanded-C-canonical-terminal-cost-v1'
TIMING_FIELDS={'mean_metadata_cpu_ms'}


def node_key(node):
    return (node['model'],node['prompt_sha256'],node['seed'],node['temperature'],node['output_cap'],node['context_length'],
            node['text'],node['input_tokens'],node['output_tokens'],node['finish'])


def raw_key(record):
    RequestJournal.validate(record);p,r=record['payload'],record['response'];o=p['options']
    return (p['model'],hashlib.sha256(p['prompt'].encode()).hexdigest(),o['seed'],o['temperature'],o['num_predict'],o['num_ctx'],
            r['response'],r['prompt_eval_count'],r.get('eval_count',0),r['done_reason'])


def replacements(new_nodes,new_raw,earlier_nodes,earlier_raw):
    by_request={r['request_sha256']:r for r in earlier_raw}
    current_raw={raw_key(r):r for r in new_raw}
    earlier={(r['id'],r['kind'],r['path']):r for r in earlier_nodes}
    chosen,receipts={},[]
    if len(current_raw)!=len(new_raw) or len(by_request)!=len(earlier_raw):raise ValueError('Duplicate raw identity within a bank')
    for row in new_nodes:
        key=row['id'],row['kind'],row['path']
        if row['kind']!='solver' or row['path']!='retry/reuse':raise ValueError('Only terminal retry/reuse substitution is supported')
        raw=current_raw[node_key(row['response'])];digest=raw['request_sha256']
        if digest not in by_request:raise ValueError('Every new terminal must have an exact earlier request')
        oldraw=by_request[digest];old=earlier[key]
        if any(node.get('model_digest')!=previous.study.old.MODEL_DIGEST
               or node.get('model')!=previous.study.old.MODEL for node in (old['response'],row['response'])):
            raise ValueError('Both terminal receipts must use the frozen compact weights')
        if (raw_key(oldraw)!=node_key(old['response']) or oldraw['returned_unix']>raw['started_unix']
                or old['response']['answer']!=row['response']['answer']):
            raise ValueError('Canonical receipt is not earlier, exact, or answer-identical')
        chosen[row['id'],row['path']]=old['response']
        receipts.append({'id':row['id'],'path':row['path'],'request_sha256':digest,
            'current_response_sha256':raw['response_sha256'],'canonical_response_sha256':oldraw['response_sha256'],
            'current_tokens':row['response']['input_tokens']+row['response']['output_tokens'],
            'canonical_tokens':old['response']['input_tokens']+old['response']['output_tokens']})
    if len(chosen)!=len(new_nodes):raise ValueError('Duplicate logical substitution')
    return chosen,receipts


def validate_canonical(args,panel):
    earlier=io.read(args.canonical/'collection-complete.json')
    registration=io.read(args.canonical/'registration.json')
    if (earlier.get('schema')!=canonical_protocol.SCHEMA or earlier.get('scope')!=canonical_protocol.SCOPE
        or earlier.get('source_sha256')!=io.sha(canonical_protocol.__file__)
        or earlier.get('registration_sha256')!=io.sha(args.canonical/'registration.json')
        or registration.get('schema')!=canonical_protocol.SCHEMA
        or registration['paths']['data']!=str(args.data.resolve())
        or registration['paths']['output']!=str(args.canonical.resolve())
        or registration['evidence']['dev_questions_sha256']!=panel
        or registration['evidence']['plain_completion_sha256']!=io.sha(args.data/'dev-plain-complete.json')
        or registration.get('model_digest')!=previous.study.old.MODEL_DIGEST
        or registration.get('test_used') is not False):
        raise ValueError('Canonical collection lacks its exact source/registration/panel/model binding')
    io.verify(registration['source_files'])
    io.verify({str(args.canonical/n):digest for n,digest in earlier['digests'].items()})
    return earlier


def inputs(args):
    if io.read(args.source/'expanded-replay-complete.json')!=previous.seal_payload(args.source):
        raise ValueError('Original expanded C seal changed')
    registered=io.read(args.source/'registration.json');io.verify(registered['source_files'])
    if registered['dev_questions_sha256']!=io.sha(args.data/'splits/dev-questions.jsonl'):
        raise ValueError('Canonical comparison must use the same256 panel')
    earlier=validate_canonical(args,registered['dev_questions_sha256'])
    current_source=Path(registered['paths']['source'])
    delta={}
    for name in ('dev-observations.jsonl','raw-generations.jsonl'):
        prefix=(current_source/name).read_bytes();raw=(args.source/name).read_bytes()
        if not raw.startswith(prefix) or (raw and not raw.endswith(b'\n')):raise ValueError('Changed whole imported prefix')
        delta[name]=[previous.common.json.loads(line) for line in raw[len(prefix):].splitlines()]
    original_records=previous.common.strict_records(args.source/'dev-observations.jsonl')
    original_raw=previous.common.strict_records(args.source/'raw-generations.jsonl')
    old_records=previous.common.strict_records(args.canonical/'dev-observations.jsonl')
    old_raw=previous.common.strict_records(args.canonical/'raw-generations.jsonl')
    changed,receipt=replacements(delta['dev-observations.jsonl'],delta['raw-generations.jsonl'],old_records,old_raw)
    previous.common.coverage(original_records,original_raw)
    raw_coverage({'strong':[r for r in old_records if (r['id'],r['path']) in changed]},old_raw)
    rows,trees,judges=previous.study.development_view(args.data,args.source/'dev-observations.jsonl')
    previous.common.validate_nodes(rows,trees,judges,original_records)
    canonical={qid:dict(tree) for qid,tree in trees.items()}
    for (qid,path),node in changed.items():canonical[qid][path]=node
    candidate_rows=io.read(args.source/'search.json')
    if len(rows)!=256 or len(candidate_rows)!=156:raise ValueError('Require complete original256×156 replay')
    models=io.read(args.source/'models/complete.json')
    for spec in models['models']:io.verify({str(args.source/'models'/(spec['key']+'.pkl')):spec['sha256']})
    for asset in models['observer_assets']:io.verify({asset['recipe_path']:asset['recipe_sha256'],**asset['weights']})
    caches=previous.previous.cache_values((args.source/'private-neural-caches.pkl').read_bytes())
    evidence={str(args.source/'expanded-replay-complete.json'):io.sha(args.source/'expanded-replay-complete.json'),
        str(args.canonical/'collection-complete.json'):io.sha(args.canonical/'collection-complete.json'),
        str(args.data/'splits/dev-questions.jsonl'):io.sha(args.data/'splits/dev-questions.jsonl'),
        str(args.data/'dev-plain-complete.json'):io.sha(args.data/'dev-plain-complete.json')}
    return rows,trees,canonical,judges,candidate_rows,models,caches,receipt,evidence


def contract(args,bundle):
    return {'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'evidence':bundle[-1],
        'paths':{n:str(getattr(args,n).resolve()) for n in ('data','source','canonical','output')},
        'substitutions':bundle[-2],'candidates':[{'model':v['model'],'settings':v['settings']} for v in bundle[4]],
        'selection':'Every exact duplicate uses the earlier retained-path receipt, uniformly; no favourable cost selection',
        'development_questions':256,'candidate_count':156,'new_encoder_calls':0,'new_cloud_calls':0,'test_used':False}


@contextmanager
def cached_model(model,caches):
    previous.previous.attach(model,caches,{},'cpu')
    with ExitStack() as stack:
        def forbidden(*args,**kwargs):raise ValueError('Missing cached feature: no neural load/forward is authorized')
        for module in (perception,option):
            for name in ('load','encode','predict'):stack.enter_context(patch.object(module,name,side_effect=forbidden))
        seen=set()
        for _,observer in observer_slots(model):
            if id(observer) in seen:continue
            seen.add(id(observer))
            if type(observer) not in (perception.Observer,option.ScienceOptionObserver):raise ValueError('Unsupported cache-only observer class')
            observer._recipe=io.read(Path(observer.directory)/'recipe.json')
            def load(current=observer):
                if current._model is not None or not current._recipe:raise ValueError('Expected unloaded archived observer with physical recipe')
            original_values=observer.values
            def values(row,path,tree,judge=None,original_values=original_values):
                if row.get('answer') or path not in ('','repair','retry','guided'):
                    raise ValueError('Reference or terminal reached an observer')
                return original_values(row,path,tree,judge)
            # Plain instance functions avoid retaining hundreds of thousands
            # of mock call-argument records during this complete replay.
            def restore(current,name,owned,value):
                if owned:setattr(current,name,value)
                else:delattr(current,name)
            for name,replacement in (('load',load),('values',values)):
                owned=name in vars(observer);before=getattr(observer,name)
                setattr(observer,name,replacement)
                stack.callback(restore,observer,name,owned,before)
        yield


def assert_same_policy(original,canonical,old_tree,new_tree):
    for name in ('id','path','correct','decisions','solver_calls','judge_calls','judge_tokens','judge_physical_calls',
                 'verifier_components','verifier_calls','verifier_token_slots'):
        if original[name]!=canonical[name]:raise ValueError('Canonical terminal changed policy/answer/observation: '+name)
    if any(d['path']=='retry/reuse' for d in original['decisions']):raise ValueError('Terminal unexpectedly observed')
    path=original['path'];expected=0
    if path=='retry/reuse':
        expected=sum(new_tree[path][n]-old_tree[path][n] for n in ('input_tokens','output_tokens'))
    if canonical['solver_tokens']-original['solver_tokens']!=expected:raise ValueError('Nonterminal token difference')
    for name in ('compute','total_parameter_compute'):
        if not np.isclose(canonical[name]-original[name],4*expected/3.09,atol=1e-10,rtol=1e-12):
            raise ValueError('Canonical cost differs from the paid terminal token delta')
    return expected


def seal_payload(directory):
    return {'schema':SCHEMA,'files':{n:io.sha(directory/n) for n in ('registration.json','search.json','summary.json','private-paired-outcomes.jsonl')}}


def run(args):
    bundle=inputs(args);expected=contract(args,bundle)
    if args.mode=='register':
        if args.output.exists():raise ValueError('Use a new canonical sensitivity output')
        args.output.mkdir(parents=True);io.write(args.output/'registration.json',expected)
        print({'registered':True,'registration_sha256':io.sha(args.output/'registration.json')},flush=True);return
    if (io.sha(args.output/'registration.json')!=args.registration_sha256
            or io.read(args.output/'registration.json')!=expected):raise ValueError('Canonical sensitivity registration changed')
    with io.lock(args.output/'.replay.lock'):
        marker=args.output/'complete.json'
        if marker.exists():
            if io.read(marker)!=seal_payload(args.output):raise ValueError('Canonical sensitivity seal changed')
            return
        rows,trees,canonical,judges,candidates,models,caches,substitutions,evidence=bundle
        by_model={r['key']:r for r in models['models']};results=[];paired=[]
        for key in by_model:
            model=pickle.loads((args.source/'models'/(key+'.pkl')).read_bytes())
            with cached_model(model,caches):
                for index,candidate in enumerate(candidates):
                    if candidate['model']!=key:continue
                    original_items=[];canonical_items=[]
                    for row in rows:
                        original=previous.dispatch(model,candidate['policy_module'],row,trees[row['id']],judges,candidate['settings'])
                        changed=previous.dispatch(model,candidate['policy_module'],row,canonical[row['id']],judges,candidate['settings'])
                        delta=assert_same_policy(original,changed,trees[row['id']],canonical[row['id']])
                        original_items.append(original);canonical_items.append(changed)
                        paired.append({'candidate_index':index,'id':row['id'],'path':original['path'],'correct':original['correct'],
                            'original_compute':original['compute'],'canonical_compute':changed['compute'],'terminal_token_delta':delta,
                            'same_decisions':True,'same_encoder_ledger':True})
                    original_stats=previous.measured(original_items);canonical_stats=previous.measured(canonical_items)
                    if {k:v for k,v in original_stats.items() if k not in TIMING_FIELDS}!={k:v for k,v in candidate['dev'].items() if k not in TIMING_FIELDS}:
                        raise ValueError('Original cached replay differs from its sealed statistics')
                    results.append({**candidate,'original_candidate_index':index,'original_dev':candidate['dev'],'dev':canonical_stats,
                        'same_decisions_answers_and_encoder_ledgers':True})
            print({'model':key,'completed_candidates':len(results)},flush=True)
        if len(results)!=156 or len(paired)!=156*256:raise ValueError('Incomplete canonical paired grid')
        original_summary=io.read(args.source/'summary.json');b=original_summary['B']['dev']['mean_compute']
        selected=previous.study.select([*results,*original_summary['prior_C']],b)
        original_selected=original_summary['selected_C']
        same_selected=next(r for r in results if (r['model'],r['settings'])==(original_selected['model'],original_selected['settings']))
        result={'schema':SCHEMA,'development_questions':256,'candidates':156,'terminal_substitutions':len(substitutions),
            'original_selected_C':original_selected,'original_selected_under_canonical_receipts':same_selected,
            'canonical_selection':selected,'B':original_summary['B'],
            'original_selected_still_cheaper_than_B':same_selected['dev']['mean_compute']<b,
            'original_selected_canonical_compute_saved_vs_B':1-same_selected['dev']['mean_compute']/b,
            'selection_changed':selected['selected_C'] is None or (selected['selected_C']['model'],selected['selected_C']['settings'])!=(original_selected['model'],original_selected['settings']),
            'maximum_accuracy_unchanged':max(r['dev']['accuracy'] for r in results)==max(r['dev']['accuracy'] for r in candidates),
            'all_decisions_answers_and_encoder_ledgers_identical':True,'new_encoder_calls':0,'new_cloud_calls':0,
            'original_artifacts_unchanged':True,'test_used':False,'fresh_validation':False,
            'metadata_CPU_timing':'Measured separately during replay; excluded only from deterministic equality checks, never from parameter compute'}
        io.verify(expected['source_files']);io.verify(evidence)
        io.jsonl(args.output/'private-paired-outcomes.jsonl',paired);io.write(args.output/'search.json',results)
        io.write(args.output/'summary.json',result);io.write(marker,seal_payload(args.output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','run'))
    for name in ('data','source','canonical','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');run(parser.parse_args())
