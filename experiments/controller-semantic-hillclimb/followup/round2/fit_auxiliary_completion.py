"""TRAIN-only nested fit of Complete16 on all768 paired science questions.

References are joined only for the512 preselected, fully completed auxiliary
questions. Protected source rows are skipped before JSON reference decoding.
Every prefix of one question stays in the same outer and inner fold.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import auxiliary_completion_features as features
import fit_regularized_completion as previous

io,fixed,nested,runtime=previous.io,previous.fixed,previous.nested,previous.runtime
SCHEMA='expanded768-nested-Complete16-TRAIN-fit-v1'


def reference_join(public,source):
    by_id={r['id']:r for r in public};labels={}
    if len(by_id)!=len(public) or any(r['answer']!='' for r in public):raise ValueError('Require unique reference-blank rows')
    with source.open('rb') as stream:
        for raw in stream:
            qid='supergpqa-'+features.pairs.contract.project(raw,{'uuid'})['uuid']
            if qid not in by_id:continue
            if qid in labels:raise ValueError('Duplicate selected source UUID')
            item=json.loads(raw);letter=item['answer_letter'].strip().upper();options=[s.strip() for s in item['options']]
            public_row=by_id[qid]
            if (features.pairs.contract.auxiliary.question(item)!=public_row['question']
                    or item['field'].casefold()!=public_row['subject'] or len(letter)!=1
                    or letter not in 'ABCDEFGHIJ'[:len(options)]
                    or ' '.join(options[ord(letter)-65].split()).casefold()!=' '.join(item['answer'].split()).casefold()):
                raise ValueError('Official selected TRAIN answer/option identity mismatch')
            labels[qid]=letter
    if set(labels)!=set(by_id):raise ValueError('Missing selected TRAIN reference')
    return [{**r,'answer':labels[r['id']]} for r in public]


def table(rows,trees,metadata,values):
    features.probe.validate(values,len(metadata));records=[];raw=[];dense=[];labels=[];valid=[]
    expected=[(r['id'],p) for r in rows for p in ('','repair','retry')]
    if [(r['id'],r['path']) for r in metadata]!=expected or len(set(expected))!=len(expected):
        raise ValueError('Exact complete question-prefix ordering required')
    i=0
    for row in rows:
        for path in ('','repair','retry'):
            paid={p:trees[row['id']][p] for p in ('',path)};prefix=path+'/' if path else ''
            if metadata[i]['prompt_sha256']!=paid[path]['prompt_sha256']:raise ValueError('Changed paid prompt alignment')
            value=(*[values[k][i] for k in ('state','plain','guided')],int(values['token_slots'][i]))
            x=runtime.core.plain.observables({**row,'answer':''},path,paid,value)
            raw.append(x);dense.append(np.r_[x,values['joint256'][i]])
            paths=[path,*[prefix+a for a in ('repair','retry','complete')]]
            labels.append(runtime.core.state(*[runtime.core.plain.correct(trees[row['id']][p],row['answer']) for p in paths]))
            valid.append(bool(paid[path]['answer']))
            records.append({'id':row['id'],'subject':row['subject'],'path':path,'fold':fixed.refined.fold_for(row['id']),
                'prompt_sha256':paid[path]['prompt_sha256'],'paid_response_sha256':fixed.fingerprint(paid[path])})
            i+=1
    return {'records':records,'features':{nested.FAMILIES[0]:np.asarray(raw),nested.FAMILIES[1]:np.asarray(dense)},
        'labels':np.asarray(labels,dtype=int),'valid':np.asarray(valid,dtype=bool)}


def inputs(args):
    spec=io.read(args.features/'registration.json')
    source=SimpleNamespace(**{k:Path(v) for k,v in spec['paths'].items()},registration_sha256=io.sha(args.features/'registration.json'))
    if source.output.resolve()!=args.features.resolve():raise ValueError('Different feature archive')
    bundle,expected=features.inputs(source)
    if expected!=spec:raise ValueError('Registered auxiliary feature inputs changed')
    io.verify(spec['source_files']);io.verify(spec['input_evidence'])
    values,done=features.verify(source,bundle,spec)
    aux_rows=reference_join(bundle['rows'],bundle['collection_args'].cache/'SuperGPQA-all.jsonl')
    rows=[*bundle['base']['rows'],*aux_rows];trees={**bundle['base']['trees'],**bundle['trees']}
    if len({r['id'] for r in rows})!=768 or Counter(r['subject'] for r in rows)!={'chemistry':384,'physics':384}:
        raise ValueError('Require exactly768 distinct balanced science TRAIN questions')
    all_values={k:np.concatenate([bundle['old_values'][k],values[k]]) for k in values}
    metadata=[*bundle['base']['metadata'],*bundle['metadata']];data=table(rows,trees,metadata,all_values)
    if any(data['features'][f].shape!=(2304,width) for f,width in zip(nested.FAMILIES,(130,386))):
        raise ValueError('Changed expanded feature width/count')
    evidence={**spec['input_evidence'],**{str((args.features/n).resolve()):v for n,v in done['digests'].items()},
        str((args.features/'complete.json').resolve()):io.sha(args.features/'complete.json')}
    registration={'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{k:str(getattr(args,k).resolve()) for k in ('features','output')},'questions':768,'decision_nodes':2304,
        'families':list(nested.FAMILIES),'outer_folds':3,'inner_folds':3,
        'hyperparameters':{f:nested.candidates(f) for f in nested.FAMILIES},
        'records_sha256':fixed.fingerprint(data['records']),'labels_sha256':fixed.array_sha(data['labels']),
        'feature_sha256':{k:fixed.array_sha(v) for k,v in data['features'].items()},
        'valid_sha256':fixed.array_sha(data['valid']),'smoothing':32.,'state_count':16,'exact_finite_Bellman':True,
        'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,'controller_accuracy_claim':False,
        'reference_scope':'Only512 fixed selected auxiliary TRAIN references plus original256 TRAIN',
        'emission_forecasts':'Nested outer-held question predictions, excluding held questions from inner tuning'}
    return rows,trees,bundle['observer'],data,registration


def fit(args,rows,trees,observer,data,record):
    with io.lock(args.output/'.fit.lock'),threadpool_limits(limits=1):
        if any(p.name not in ('registration.json','.fit.lock') for p in args.output.iterdir()):raise ValueError('Preserve prior fit output')
        models={};oof={};reports={}
        for family in nested.FAMILIES:
            predictions=np.full((2304,16),np.nan);receipts=[]
            for fold in range(3):
                held,p,model,receipt=nested.outer_prediction(data,family,fold);predictions[held]=p;receipts.append(receipt)
                io.atomic_bytes(args.output/(family+f'-outer{fold}.pkl'),pickle.dumps(model,protocol=5))
                print({'family':family,'outer_fold_complete':fold,'selected':receipt['selected']},flush=True)
            if not np.isfinite(predictions).all():raise ValueError('Incomplete nested question-held forecasts')
            oof[family]=predictions
            reports[family]={s:{k:float(v[np.asarray([s=='all' or r['subject']==s for r in data['records']])].mean())
                for k,v in fixed.losses(data['labels'],predictions).items()} for s in ('all','chemistry','physics')}
            forecaster,final=previous.final_forecaster(data,family)
            obs=runtime.DenseObserver(observer.directory,observer.temperature,calibration=observer.calibration) if family==nested.FAMILIES[1] else observer
            obs._recipe=observer._recipe
            model=runtime.fit_dynamics(rows,trees,obs,data['records'],predictions,forecaster,family,
                {'registration_sha256':args.registration_sha256,'outer_folds':receipts,'final_forecaster':final})
            path=args.output/(family+'.pkl');io.atomic_bytes(path,pickle.dumps(model,protocol=5))
            models[family]={'model_sha256':io.sha(path),'fit_receipt':model.fit_receipt}
        io.npz(args.output/'private-oof.npz',oof);io.write(args.output/'training-diagnostics.json',reports)
        io.verify(record['source_files']);io.verify(record['input_evidence'])
        io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,'models':models,
            'training_questions':768,'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,'controller_accuracy_claim':False,
            'digests':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
        print({'complete':True,'training_questions':768,'diagnostics':reports,'controller_accuracy_claim':False},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','register','fit'))
    for n in ('features','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--registration-sha256');args=p.parse_args()
    with fixed.training_only_reads(),fixed.guarded.no_model_calls():
        rows,trees,observer,data,record=inputs(args)
        if args.phase=='plan':print({'questions':768,'nodes':2304,'new_neural_calls':0});return
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'fitting_started':False});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Changed expanded fit registration')
        fit(args,rows,trees,observer,data,record)


if __name__=='__main__':main()
