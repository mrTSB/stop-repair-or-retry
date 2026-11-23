"""Two fixed transport checks after the preserved failed manual-raw OCR test.

Uses the exact same synthetic pixels/code, never a benchmark question. Native
formatting and manual raw formatting with explicit think=False are both tested
once; prefer native if both can read the code. No scientific result is selected.
"""
import argparse
import base64
from pathlib import Path
import time

import vision_binding_preflight as original

read,write,sha,append,digest = original.read,original.write,original.sha,original.append,original.digest
SCHEMA = 'frozen-reviewer-native-image-transport-preflight-v2'


def registration(args):
    prior = read(args.prior/'registration.json')
    done = read(args.prior/'complete.json')
    if done['registration_sha256'] != sha(args.prior/'registration.json') or done['passed']:
        raise ValueError('This diagnostic follows the preserved failed first preflight')
    for name,value in done['digests'].items():
        if sha(args.prior/name) != value: raise ValueError('Changed first technical preflight')
    if prior['source_sha256'] != sha(original.__file__): raise ValueError('Changed original preflight implementation')
    return {'schema':SCHEMA,'source_sha256':sha(__file__),'prior':str(args.prior.resolve()),
        'prior_complete_sha256':sha(args.prior/'complete.json'),'prior_source_sha256':sha(original.__file__),
        'image_sha256':sha(args.prior/'code.png'),'model':original.MODEL,'model_digest':original.DIGEST,
        'app':original.APP,'maximum_cloud_gpus':2,'maximum_calls':2,'output_cap':32,
        'variants':['native_no_think','manual_raw_no_think'],'same_pixels_and_code_as_first_test':True,
        'selection':'Native if exact OCR succeeds, otherwise manual raw if it succeeds, otherwise no benchmark vision',
        'both_variants_execute_unconditionally':True,'benchmark_records_read':False,
        'benchmark_accuracy_claim':False,'vision_patch_slots':prior['vision_input_patch_slots'],
        'vision_encoder_parameters':prior['vision_parameters'],
        'projected_image_slots':prior['projected_language_image_slots']}


def run(args,record):
    if (not args.registration_sha256 or sha(args.output/'registration.json') != args.registration_sha256
            or read(args.output/'registration.json') != record): raise ValueError('Changed native preflight contract')
    if any((args.output/k).exists() for k in ('attempts.jsonl','raw.jsonl','complete.json')):
        raise ValueError('Preserve prior native-format attempts')
    import modal
    worker = modal.Cls.from_name(original.APP,'Ollama')()
    results = {}
    try:
        worker.request.autoscaler.update(min_containers=1,max_containers=2)
        tags = {r['name']:r['digest'] for r in worker.request.remote('/api/tags')['models']}
        if tags.get(original.MODEL) != original.DIGEST: raise ValueError('Frozen reviewer changed')
        for variant in record['variants']:
            raw = variant.startswith('manual')
            text = 'Return only the eight-character code visible in the supplied image.'
            prompt = ('<|im_start|>user\n[img-0]\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n') if raw else text
            payload = {'model':original.MODEL,'prompt':prompt,'raw':raw,'think':False,'stream':False,
                'images':[base64.b64encode((args.prior/'code.png').read_bytes()).decode()],'keep_alive':'30m',
                'options':{'temperature':0,'seed':2026112261,'num_predict':32,'num_ctx':4096}}
            append(args.output/'attempts.jsonl',{'variant':variant,'payload':payload,'payload_sha256':digest(payload)})
            started = time.time()
            response = worker.request.remote('/api/generate',payload)
            append(args.output/'raw.jsonl',{'variant':variant,'started_unix':started,'returned_unix':time.time(),
                'response':response,'response_sha256':digest(response),'payload_sha256':digest(payload)})
            if response.get('done') is not True or response.get('model') != original.MODEL:
                raise ValueError('Incomplete technical response')
            results[variant] = response
    finally:
        worker.request.autoscaler.update(min_containers=0,max_containers=2)
    checks = {name:{'exact_code_read':original.CODE in value.get('response',''),
                    'image_tokens_present':value['prompt_eval_count'] >= record['projected_image_slots']} for name,value in results.items()}
    selected = next((name for name in record['variants'] if all(checks[name].values())),None)
    summary = {'schema':SCHEMA,'checks':checks,'selected_transport':selected,'passed':selected is not None,
        'physical_calls':2,'benchmark_records_read':False,'benchmark_accuracy_claim':False,
        'prior_failure_preserved':True,'metrics':{name:{k:value.get(k) for k in ('prompt_eval_count','prompt_eval_cached_count','eval_count')}
                                                  for name,value in results.items()}}
    write(args.output/'summary.json',summary)
    write(args.output/'complete.json',{'registration_sha256':args.registration_sha256,'passed':summary['passed'],
        'digests':{name:sha(args.output/name) for name in ('registration.json','attempts.jsonl','raw.jsonl','summary.json')}})
    print(summary,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','run'))
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    record = registration(args)
    if args.phase == 'register':
        args.output.mkdir(parents=True,exist_ok=False)
        write(args.output/'registration.json',record)
        print({'registration_sha256':sha(args.output/'registration.json'),'maximum_calls':2},flush=True)
    else: run(args,record)


if __name__ == '__main__': main()
