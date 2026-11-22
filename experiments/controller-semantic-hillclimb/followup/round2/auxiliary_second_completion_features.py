"""Collect old frozen observer features for all completed512 paired TRAIN rows.

No labels from the new source are decoded, no DEV/TEST is opened and no cloud
call is made. Capture raw20 and project256 in the same CPU forward. The encoder
is the original V1 model, which excluded every auxiliary question.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

import collect_auxiliary_second_pairs as pairs
import extract_dense_completion as previous

io,probe,fixed=previous.io,previous.probe,previous.fixed
SCHEMA='auxiliary-second512-old-V1-raw20-project256-features-v1'
NODES=1536


def inputs(args):
    spec=io.read(args.collection/'registration.json')
    source=SimpleNamespace(**{k:Path(v) for k,v in spec['paths'].items()},registration_sha256=io.sha(args.collection/'registration.json'))
    if source.output.resolve()!=args.collection.resolve():raise ValueError('Different paired TRAIN archive')
    rows,known,done=pairs.validated_complete(source)
    base=previous.training_bundle(args.base_fit)
    observer=base['observer']
    if Path(observer.directory).resolve()!=source.observer.resolve():raise ValueError('Require the excluded old V1 observer')
    if {r['id'] for r in rows}&{r['id'] for r in base['rows']}:raise ValueError('Old and expanded paired TRAIN overlap')
    old_values,old_done=previous.verify_complete(args.original_neural,io.sha(args.original_neural/'registration.json'),base)
    trees={r['id']:{p:known[r['id'],p]['response'] for p in pairs.contract.PATHS} for r in rows}
    entries,metadata=[],[]
    for row in rows:
        if row['answer']!='':raise ValueError('Public collection must have blank references')
        for path in ('','repair','retry'):
            paid={p:trees[row['id']][p] for p in ('',path)};entry=observer.entry(row,path,paid)
            entries.append(entry);metadata.append({'id':row['id'],'subject':row['subject'],'path':path,
                'prompt_sha256':paid[path]['prompt_sha256'],'input_sha256':fixed.fingerprint(entry)})
    if len(entries)!=NODES:raise ValueError('Require all512×3 paid inputs')
    evidence={**spec['input_evidence']['files'],**base['evidence'],
        **{str((args.collection/n).resolve()):v for n,v in done['digests'].items()},
        **{str((args.original_neural/n).resolve()):v for n,v in old_done['digests'].items()},
        str((args.collection/'complete.json').resolve()):io.sha(args.collection/'complete.json'),
        str((args.original_neural/'neural-complete.json').resolve()):io.sha(args.original_neural/'neural-complete.json')}
    registration={'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{k:str(getattr(args,k).resolve()) for k in ('collection','base_fit','original_neural','output')},
        'entry_sha256':fixed.fingerprint(entries),'metadata_sha256':fixed.fingerprint(metadata),
        'new_questions':512,'new_observations':NODES,'physical_batched_forwards':NODES//4,
        'combined_training_questions':1280,'device':'cpu','threads':2,'batch':4,'chunk':32,'max_length':512,
        'parameters':observer._recipe['parameters'],'observer_directory':str(Path(observer.directory).resolve()),
        'extra_project256_passes':0,'cloud_calls':0,'encoder_training':False,'old_encoder_excluded_auxiliary':True,
        'auxiliary_references_decoded':False,'development_or_test_read':False,
        'future_fit':'All1280 science questions, including the separately sealed first512 feature block, question-grouped nested CV for raw130 and dense386; fixed two-step Complete16 POMDP',
        'resume_rule':'Durable attempted chunk before inference. Any missing result after an attempt blocks automatic repeat.'}
    return {'rows':rows,'trees':trees,'entries':entries,'metadata':metadata,'observer':observer,
        'base':base,'old_values':old_values,'collection_args':source},registration


def identity(bundle,start,registration):
    if type(start)is not int or start<0 or start>=NODES or start%32:raise ValueError('Invalid feature chunk')
    return {'start':start,'count':32,'registration_sha256':registration,
        'input_sha256':fixed.fingerprint(bundle['entries'][start:start+32]),
        'metadata_sha256':fixed.fingerprint(bundle['metadata'][start:start+32])}


def chunks(folder,bundle,registration,forward,guard):
    directory=folder/'chunks';directory.mkdir(exist_ok=True)
    allowed={f'{i:03d}{s}' for i in range(NODES//32) for s in ('.npz','.json')}
    if any(p.name not in allowed for p in directory.iterdir()) or list(folder.glob('*.tmp')):
        raise ValueError('Unsealed or unexpected feature output')
    attempts={}
    for item in pairs.pilot.protocol.strict_records(folder/'attempts.jsonl'):
        start=item['start']
        if start in attempts or item!=identity(bundle,start,registration):raise ValueError('Changed or duplicate forward attempt')
        attempts[start]=item
    values=[];digests={}
    for start in range(0,NODES,32):
        key=identity(bundle,start,registration);path=directory/f'{start//32:03d}.npz';receipt=path.with_suffix('.json')
        if path.exists() or receipt.exists():
            if not path.exists() or not receipt.exists() or start not in attempts:raise ValueError('Unsealed chunk requires inspection')
            record=io.read(receipt)
            if record['identity']!=key or record['arrays_sha256']!=io.sha(path):raise ValueError('Changed chunk identity')
            with np.load(path,allow_pickle=False) as archive:part={k:archive[k].copy() for k in archive.files}
        else:
            if start in attempts:raise ValueError('Unresolved neural attempt; no automatic repetition')
            guard();io.append(folder/'attempts.jsonl',key)
            part=forward(bundle['entries'][start:start+32]);probe.validate(part,32)
            io.npz(path,part);record={'identity':key,'arrays_sha256':io.sha(path),'physical_batched_forwards':8,
                'padded_token_slots':32*512,'unpadded_token_slots':int(part['token_slots'].sum())}
            io.write(receipt,record)
        probe.validate(part,32)
        if (record['physical_batched_forwards']!=8 or record['padded_token_slots']!=32*512
                or record['unpadded_token_slots']!=int(part['token_slots'].sum())):raise ValueError('Changed feature cost receipt')
        values.append(part);digests[str(path.relative_to(folder))]=io.sha(path);digests[str(receipt.relative_to(folder))]=io.sha(receipt)
        print({'completed_auxiliary_rows':start+32,'total':NODES},flush=True)
    return {k:np.concatenate([v[k] for v in values]) for k in values[0]},digests


def verify(args,bundle,record):
    done=io.read(args.output/'complete.json')
    expected={'registration.json','attempts.jsonl','memory.jsonl','neural.npz','metadata.jsonl'}
    expected|={f'chunks/{i:03d}{s}' for i in range(NODES//32) for s in ('.npz','.json')}
    if (done['schema']!=SCHEMA or done['registration_sha256']!=args.registration_sha256
            or done['nodes']!=NODES or done['scored'] or done['cloud_calls'] or done['development_or_test_read']
            or done['physical_batched_forwards']!=NODES//4 or set(done['digests'])!=expected):
        raise ValueError('Require complete ungraded auxiliary feature grid')
    io.verify({str(args.output/n):v for n,v in done['digests'].items()})
    def forbidden(*args):raise AssertionError('Verification cannot perform neural inference')
    combined,_=chunks(args.output,bundle,args.registration_sha256,forbidden,forbidden)
    with np.load(args.output/'neural.npz',allow_pickle=False) as archive:values={k:archive[k].copy() for k in archive.files}
    probe.validate(values,NODES)
    if any(not np.array_equal(values[k],combined[k]) for k in values):raise ValueError('Aggregate differs from sealed chunks')
    metadata=pairs.pilot.protocol.strict_records(args.output/'metadata.jsonl')
    if (metadata!=bundle['metadata'] or fixed.fingerprint(metadata)!=record['metadata_sha256']
            or done['unpadded_token_slots']!=int(values['token_slots'].sum())
            or done['padded_token_slots']!=NODES*512):raise ValueError('Changed alignment or cost ledger')
    return values,done


def extract(args,bundle,record):
    with io.lock(args.output/'.extract.lock'):
        if (args.output/'complete.json').exists():return verify(args,bundle,record)
        if any((args.output/n).exists() for n in ('neural.npz','metadata.jsonl')):raise ValueError('Preserve unsealed aggregate')
        model=tokenizer=None
        def guard():previous.resource.memory_guard(args.output,model is not None)
        def forward(entries):
            nonlocal model,tokenizer
            if model is None:
                torch.set_num_threads(2);model,tokenizer,recipe=previous.network.load(record['observer_directory'])
                if (recipe!=bundle['observer']._recipe or sum(p.numel() for p in model.parameters())!=record['parameters']
                        or any(p.device.type!='cpu' for p in model.parameters())):raise ValueError('Changed frozen observer')
                for p in model.parameters():p.requires_grad_(False)
            parts=[probe.capture(model,previous.network.encode(tokenizer,entries[i:i+4],512)) for i in range(0,len(entries),4)]
            return {k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
        try:values,digests=chunks(args.output,bundle,args.registration_sha256,forward,guard)
        finally:model=tokenizer=None;gc.collect()
        io.npz(args.output/'neural.npz',values);io.jsonl(args.output/'metadata.jsonl',bundle['metadata'])
        digests.update({n:io.sha(args.output/n) for n in ('registration.json','attempts.jsonl','memory.jsonl','neural.npz','metadata.jsonl')})
        io.verify(record['source_files']);io.verify(record['input_evidence'])
        io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
            'nodes':NODES,'digests':digests,'physical_batched_forwards':NODES//4,'padded_token_slots':NODES*512,
            'unpadded_token_slots':int(values['token_slots'].sum()),'scored':False,'cloud_calls':0,
            'development_or_test_read':False,'extra_project256_passes':0})
        verify(args,bundle,record)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','register','extract'))
    for n in ('collection','base-fit','original-neural','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--registration-sha256');args=p.parse_args()
    with fixed.training_only_reads():
        with fixed.guarded.no_model_calls():bundle,record=inputs(args)
        if args.phase=='plan':print({'nodes':NODES,'cloud_calls':0,'new_reference_labels_decoded':False});return
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'inference_started':False});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Changed registered feature acquisition')
        extract(args,bundle,record)


if __name__=='__main__':main()
