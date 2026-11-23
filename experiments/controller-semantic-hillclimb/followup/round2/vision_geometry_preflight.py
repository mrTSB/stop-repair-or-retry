"""Two bounded, generic diagnostics for image geometry/model compatibility.

Use exactly the previous synthetic code pixels: frozen35B on a white-padded
768-square image and frozen4B on the original image. Both run once regardless
of outcomes. No project question, reference, or scientific answer is read.
"""
import argparse
import base64
from pathlib import Path
import time

import vision_binding_preflight as original
import vision_native_preflight as native

read,write,sha,append,digest = original.read,original.write,original.sha,original.append,original.digest
SCHEMA = 'frozen-model-geometry-compatibility-preflight-v3'
COMPACT = 'qwen3.5:4b'
COMPACT_DIGEST = '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd'


def contract(args):
    done = read(args.prior/'complete.json')
    prior = read(args.prior/'registration.json')
    if done['passed'] or prior['source_sha256'] != sha(native.__file__):
        raise ValueError('Require the preserved unsuccessful native transport check')
    for name,value in done['digests'].items():
        if sha(args.prior/name) != value: raise ValueError('Changed native preflight')
    image = Path(prior['prior'])/'code.png'
    if sha(image) != prior['image_sha256']: raise ValueError('Changed synthetic image')
    return {'schema':SCHEMA,'source_sha256':sha(__file__),
        'dependencies':{str(Path(m.__file__).resolve()):sha(m.__file__) for m in (original,native)},
        'prior_complete_sha256':sha(args.prior/'complete.json'),'original_image':str(image.resolve()),
        'original_image_sha256':sha(image),'app':original.APP,'maximum_cloud_gpus':2,
        'variants':[{'name':'reviewer_square_padding','model':original.MODEL,'digest':original.DIGEST,'size':[768,768]},
                    {'name':'compact_original_geometry','model':COMPACT,'digest':COMPACT_DIGEST,'size':[512,736]}],
        'padding':'White border only, centered using integer offsets; no resizing or altered original pixels',
        'both_variants_execute_unconditionally':True,'maximum_physical_generations':2,
        'raw':False,'think':False,'temperature':0,'seed':2026112261,'output_cap':32,
        'benchmark_records_read':False,'benchmark_accuracy_claim':False,
        'selection':'Only exact full code and image-token presence pass; preserve all failures; no benchmark image use otherwise'}


def prepare(args,record):
    from PIL import Image
    args.output.mkdir(parents=True,exist_ok=False)
    old = Image.open(record['original_image']).convert('RGB')
    if old.size != (512,736): raise ValueError('Changed original geometry')
    target = Image.new('RGB',(768,768),'white')
    target.paste(old,(128,16))
    if target.crop((128,16,640,752)).tobytes() != old.tobytes(): raise ValueError('Altered synthetic pixels')
    target.save(args.output/'square.png')
    write(args.output/'registration.json',{**record,'square_image_sha256':sha(args.output/'square.png')})
    print({'registration_sha256':sha(args.output/'registration.json'),'maximum_physical_generations':2},flush=True)


def run(args,record):
    registered = read(args.output/'registration.json')
    if ({k:v for k,v in registered.items() if k != 'square_image_sha256'} != record
            or sha(args.output/'registration.json') != args.registration_sha256
            or sha(args.output/'square.png') != registered['square_image_sha256']):
        raise ValueError('Changed geometry preflight registration')
    if any((args.output/k).exists() for k in ('attempts.jsonl','raw.jsonl','complete.json')):
        raise ValueError('Preserve previous technical attempts')
    import modal
    worker = modal.Cls.from_name(original.APP,'Ollama')()
    results = {}
    try:
        worker.request.autoscaler.update(min_containers=1,max_containers=2)
        append(args.output/'capacity.jsonl',{'min_containers':1,'max_containers':2})
        tags = {r['name']:r['digest'] for r in worker.request.remote('/api/tags')['models']}
        for variant in record['variants']:
            if tags.get(variant['model']) != variant['digest']: raise ValueError('Frozen model changed')
            image = args.output/'square.png' if variant['name'] == 'reviewer_square_padding' else Path(record['original_image'])
            payload = {'model':variant['model'],'prompt':'Return only the eight-character code visible in the supplied image.',
                'raw':False,'think':False,'stream':False,'images':[base64.b64encode(image.read_bytes()).decode()],
                'keep_alive':'30m','options':{'temperature':0,'seed':2026112261,'num_predict':32,'num_ctx':4096}}
            append(args.output/'attempts.jsonl',{'variant':variant,'payload':payload,'payload_sha256':digest(payload)})
            started = time.time()
            response = worker.request.remote('/api/generate',payload)
            append(args.output/'raw.jsonl',{'variant':variant,'started_unix':started,'returned_unix':time.time(),
                'response':response,'response_sha256':digest(response),'payload_sha256':digest(payload)})
            if response.get('done') is not True or response.get('model') != variant['model']:
                raise ValueError('Incomplete technical response')
            results[variant['name']] = response
    finally:
        worker.request.autoscaler.update(min_containers=0,max_containers=2)
        append(args.output/'capacity.jsonl',{'min_containers':0,'max_containers':2})
    checks = {v['name']:{'exact_code_read':original.CODE in results[v['name']].get('response',''),
        'image_tokens_present':results[v['name']]['prompt_eval_count'] >= v['size'][0]*v['size'][1]//1024} for v in record['variants']}
    summary = {'schema':SCHEMA,'checks':checks,'passed_variants':[k for k,v in checks.items() if all(v.values())],
        'physical_calls':2,'benchmark_records_read':False,'benchmark_accuracy_claim':False,
        'metrics':{name:{k:r.get(k) for k in ('prompt_eval_count','prompt_eval_cached_count','eval_count')} for name,r in results.items()}}
    write(args.output/'summary.json',summary)
    names = ('registration.json','square.png','attempts.jsonl','raw.jsonl','summary.json','capacity.jsonl')
    write(args.output/'complete.json',{'registration_sha256':args.registration_sha256,
        'digests':{name:sha(args.output/name) for name in names},'benchmark_records_read':False})
    print(summary,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','run'))
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    record = contract(args)
    if args.phase == 'register': prepare(args,record)
    else: run(args,record)


if __name__ == '__main__': main()
