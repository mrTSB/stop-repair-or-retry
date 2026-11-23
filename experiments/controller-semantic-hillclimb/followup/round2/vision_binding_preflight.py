"""Generic two-call image-binding check on the existing frozen reviewer.

The random-looking code occurs only in synthetic image pixels, never in the
text prompt. This checks image plumbing and token accounting, not benchmark
accuracy. No project question, reference, training record or test is read.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time

MODEL = 'qwen3.5:35b'
DIGEST = '3460ffeede5453ead027dbd2f821b12ad0aa3de54630971993babdb2165221f7'
APP = 'srr-followup-solver-20261120'
CODE = 'R8K2Q7N4'
WIDTH,HEIGHT = 512,736
FONT = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')


def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare(args):
    from PIL import Image,ImageDraw,ImageFont,__version__
    meta = read(args.metadata)
    params = sum(math.prod(t['shape']) for t in meta['tensors'] if t['name'].startswith('v.'))
    info = meta['model_info']
    prefix = 'qwen35moe.vision.'
    patch,merge = info[prefix+'patch_size'],info[prefix+'spatial_merge_size']
    if params != 446571248 or patch != 16 or merge != 2 or 'vision' not in meta['capabilities']:
        raise ValueError('Unexpected frozen reviewer visual architecture')
    if not info[prefix+'shortest_edge'] <= WIDTH*HEIGHT <= info[prefix+'longest_edge']:
        raise ValueError('Synthetic image would be resized by the declared pixel bounds')
    args.output.mkdir(parents=True,exist_ok=False)
    image = Image.new('RGB',(WIDTH,HEIGHT),'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT),72)
    bounds = draw.textbbox((0,0),CODE,font=font)
    draw.text(((WIDTH-(bounds[2]-bounds[0]))/2,(HEIGHT-(bounds[3]-bounds[1]))/2-bounds[1]),CODE,font=font,fill='black')
    image.save(args.output/'code.png')
    slots = (WIDTH//patch)*(HEIGHT//patch)
    record = {'schema':'generic-frozen-reviewer-vision-binding-v1','source_sha256':sha(__file__),
        'metadata_path':str(args.metadata.resolve()),'metadata_sha256':sha(args.metadata),
        'image_sha256':sha(args.output/'code.png'),'font_sha256':sha(FONT),'pillow':__version__,
        'model':MODEL,'model_digest':DIGEST,'app':APP,'maximum_cloud_gpus':2,
        'width':WIDTH,'height':HEIGHT,'vision_parameters':params,'vision_input_patch_slots':slots,
        'projected_language_image_slots':slots//merge**2,
        'vision_encoder_compute_per_phase':params*slots/3.09e9,
        'maximum_calls':2,'output_cap':32,'question_or_reference_read':False,
        'code_only_in_image_pixels':True,'API_image_marker':'[img-0]','raw':True,
        'vision_binding_must_pass_before_benchmark_images':True}
    write(args.output/'registration.json',record)
    print({'registration_sha256':sha(args.output/'registration.json'),'vision_parameters':params,
           'vision_patch_slots':slots,'projected_image_slots':slots//merge**2},flush=True)


def append(path,value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value,sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    record = read(args.output/'registration.json')
    if (not args.registration_sha256 or sha(args.output/'registration.json') != args.registration_sha256
            or record['source_sha256'] != sha(__file__) or record['metadata_sha256'] != sha(record['metadata_path'])
            or record['image_sha256'] != sha(args.output/'code.png')):
        raise ValueError('Changed generic preflight contract')
    if any((args.output/name).exists() for name in ('attempts.jsonl','raw.jsonl','complete.json')):
        raise ValueError('Preserve prior generic preflight; inspect partial attempts')
    import modal
    worker = modal.Cls.from_name(APP,'Ollama')()
    results = {}
    try:
        worker.request.autoscaler.update(min_containers=1,max_containers=2)
        tags = {r['name']:r['digest'] for r in worker.request.remote('/api/tags')['models']}
        if tags.get(MODEL) != DIGEST: raise ValueError('Frozen reviewer changed')
        for arm in ('text_control','image'):
            marker = '[img-0]\n' if arm == 'image' else ''
            prompt = ('<|im_start|>user\n'+marker+'Return only the eight-character code visible in the supplied image.'
                '<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n')
            payload = {'model':MODEL,'prompt':prompt,'raw':True,'stream':False,'keep_alive':'30m',
                       'options':{'temperature':0,'seed':2026112261,'num_predict':32,'num_ctx':4096}}
            if arm == 'image': payload['images'] = [base64.b64encode((args.output/'code.png').read_bytes()).decode()]
            attempt = {'arm':arm,'payload':payload,'payload_sha256':digest(payload)}
            append(args.output/'attempts.jsonl',attempt)
            started = time.time()
            response = worker.request.remote('/api/generate',payload)
            append(args.output/'raw.jsonl',{'arm':arm,'started_unix':started,'returned_unix':time.time(),
                'payload_sha256':digest(payload),'response':response,'response_sha256':digest(response)})
            if response.get('done') is not True or response.get('model') != MODEL:
                raise ValueError('Incomplete generic preflight response')
            results[arm] = response
    finally:
        worker.request.autoscaler.update(min_containers=0,max_containers=2)
    text = results['text_control'].get('response','')
    image = results['image'].get('response','')
    delta = results['image']['prompt_eval_count']-results['text_control']['prompt_eval_count']
    checks = {'visible_code_read':CODE in image,'code_not_in_text_control':CODE not in text,
              'image_token_count_consistent':abs(delta-record['projected_language_image_slots']) <= 16}
    summary = {'schema':record['schema'],'checks':checks,'passed':all(checks.values()),
        'input_count_delta':delta,'projected_image_slots':record['projected_language_image_slots'],
        'vision_parameters':record['vision_parameters'],'vision_patch_slots':record['vision_input_patch_slots'],
        'vision_encoder_compute_per_phase':record['vision_encoder_compute_per_phase'],
        'physical_calls':2,'benchmark_question_read':False,'benchmark_accuracy_claim':False,
        'metrics':{arm:{k:v.get(k) for k in ('prompt_eval_count','prompt_eval_cached_count','eval_count','load_duration','prompt_eval_duration','eval_duration')}
                   for arm,v in results.items()}}
    write(args.output/'summary.json',summary)
    write(args.output/'complete.json',{'registration_sha256':args.registration_sha256,'passed':summary['passed'],
        'digests':{name:sha(args.output/name) for name in ('registration.json','code.png','attempts.jsonl','raw.jsonl','summary.json')}})
    print(summary,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run'))
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    if args.phase == 'prepare': prepare(args)
    else: run(args)


if __name__ == '__main__': main()
