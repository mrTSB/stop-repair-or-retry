"""Bounded full dev/freeze/test collection using private Modal RPC. No public endpoint or hosted token API."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import modal


MODEL='Qwen/Qwen2.5-14B-Instruct'
REVISION='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
MODEL_ID=MODEL+'@'+REVISION
MAX_CALLS=920
WALL_LIMIT_SECONDS=1800
image=modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.6.0','transformers==4.51.3','accelerate==1.6.0','numpy==2.2.5')
app=modal.App('cs238-llm-controller-full-evaluation')

@app.cls(image=image,gpu='A100-80GB',cpu=(4,4),memory=(49152,49152),max_containers=1,timeout=120,startup_timeout=600,scaledown_window=10,retries=0)
class Controller:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
        self.model.eval()

    @modal.method()
    def metadata(self):
        import torch,importlib.metadata
        return {'transport':'authenticated private Modal RPC, not HTTP','model':MODEL,'revision':REVISION,'reported_model':MODEL_ID,'gpu':torch.cuda.get_device_name(),'precision':'bfloat16','decoding':'greedy do_sample=False; no constrained decoding or JSON repair','output_token_limit':128,'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate']}}

    @modal.method()
    def decide(self,body):
        import torch
        require=set(body)=={'model','messages','stream','max_tokens','temperature'}
        if not require or body['model']!=MODEL_ID or body['temperature']!=0 or body['stream'] or body['max_tokens']!=128:
            raise ValueError('Unexpected frozen pilot request.')
        messages=body['messages']
        if len(messages)!=2 or [x['role'] for x in messages]!=['system','user']:
            raise ValueError('Expected system instructions and allowlisted observed history.')
        observation=json.loads(messages[1]['content'])
        if set(observation)!={'problem','remaining_solver_calls','lambda_per_1000_tokens','visited_responses'}:
            raise ValueError('Unexpected observation fields.')
        formatted=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(formatted,return_tensors='pt',add_special_tokens=False).to('cuda')
        if inputs['input_ids'].shape[1]+128>32768:raise ValueError('Context limit exceeded.')
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            generated=self.model.generate(**inputs,max_new_tokens=128,do_sample=False,pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        tokens=generated[0,inputs['input_ids'].shape[1]:].tolist()
        eos=self.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        end=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:end]
        inp=int(inputs['attention_mask'].sum());out=len(tokens)
        text=self.tokenizer.decode(tokens,skip_special_tokens=True)
        return {'id':'modal-'+hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()[:24],'model':MODEL_ID,
                'choices':[{'message':{'role':'assistant','content':text},'finish_reason':'stop' if tokens and tokens[-1] in eos else 'length'}],
                'usage':{'prompt_tokens':inp,'completion_tokens':out,'total_tokens':inp+out},
                'measurement':{'transport':'Modal RPC','formatted_prompt':formatted,'gpu_generation_seconds':elapsed,'input_tokenization_includes_chat_template':True,'output_tokens_include_eos':True,'model_revision':REVISION}}

def main():
    import os
    import subprocess
    import shutil
    ROOT=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(ROOT/'src'))
    from stop_repair_retry import llm_experiment as experiment
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--prior-reserved-usd',required=True,type=float)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute or os.environ.get('MODAL_PROFILE')!='gancone':
        raise SystemExit('Explicit --execute and MODAL_PROFILE=gancone required.')
    output=Path(args.output).resolve()
    if output.exists():raise SystemExit('Refusing overwrite or automatic replay.')
    rate=.000694+4*.0000131+48*.00000222
    # Reserve the watchdog lifetime plus shutdown/call and startup slack, plus fees.
    reservation=(WALL_LIMIT_SECONDS+600+120)*rate+1
    if not 0<=args.prior_reserved_usd<20 or args.prior_reserved_usd+reservation>=20:
        raise SystemExit('Cumulative reservation exceeds $20.')
    config=experiment.load_config(ROOT/'experiments/llm-controller/config.example.json')
    config.update(endpoint='https://modal.com/apps/gancone/main/cs238-llm-controller-full-evaluation',model=MODEL_ID,
        response_format='none',billing_mode='compute',compute_budget_confirmed=True,
        max_requests=800,max_run_seconds=1200,timeout_seconds=120)
    prompt=experiment.DEFAULT_PROMPT.read_text()
    output.mkdir(parents=True)
    experiment.save(output/'config.json',config)
    experiment.save(output/'protocol.json',{
        'created_utc':experiment.now(),'sequence':['all 60 dev','evaluate dev','freeze unchanged settings','all 400 test','evaluate test'],
        'model':MODEL_ID,'driver_sha256':experiment.sha(__file__),'prompt_sha256':experiment.sha(experiment.DEFAULT_PROMPT),
        'max_decisions':MAX_CALLS,'max_metadata_calls':1,'wall_watchdog_seconds':WALL_LIMIT_SECONDS,
        'prior_reserved_usd':args.prior_reserved_usd,'new_reservation_usd':reservation,
        'cumulative_reserved_usd':args.prior_reserved_usd+reservation,'user_cap_usd':20,
        'resources':{'gpu':'A100-80GB','containers':1,'cpu':4,'memory_gib':48,'startup_timeout_seconds':600,'call_timeout_seconds':120,'retries':0,'rate_usd_per_second':rate},
        'changes_from_pilot':'Operational request/runtime limits and descriptive endpoint only. Same weights, prompt, reward, precision, decoding, action parser and visited-history inputs. No tuning from pilot or full dev outcomes.',
        'split_note':'Project splits, not source benchmark split names. Ten dev questions overlap earlier pilot and are recollected once here; no response selection. Existing fixed-baseline test outcomes have been viewed previously.',
        'transport':'private authenticated Modal RPC; descriptive endpoint not used for HTTP',
        'latency_note':'End-to-end controller RPC measured; solver responses replayed from cache.',
    })
    shutil.copyfile(__file__,output/'modal_full_evaluation.frozen.py')
    calls=0;watchdog=None;started=time.monotonic()
    with modal.enable_output(),app.run():
        experiment.save(output/'modal-app.json',{'app_id':app.app_id,'profile':'gancone','started_utc':experiment.now()})
        # Independent subprocess survives parent interruption and stops the app at the deadline.
        watchdog_source="""import subprocess,sys,time
 time.sleep(float(sys.argv[2]))
 for attempt in range(3):
     try:
         result=subprocess.run([sys.executable,'-m','modal','app','stop',sys.argv[1],'--profile','gancone','--yes'],timeout=60)
         if result.returncode==0:break
     except subprocess.TimeoutExpired:pass
     time.sleep(5)
 """.replace('\n ', '\n')
        with (output/'watchdog.log').open('w') as log:
            watchdog=subprocess.Popen([sys.executable,'-c',watchdog_source,app.app_id,str(WALL_LIMIT_SECONDS)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        experiment.save(output/'watchdog.json',{'pid':watchdog.pid,'deadline_seconds':WALL_LIMIT_SECONDS,'action':'modal app stop --yes','independent_process':True})
        worker=Controller()
        experiment.save(output/'backend.json',worker.metadata.remote())
        def client(body):
            nonlocal calls
            if calls>=MAX_CALLS or time.monotonic()-started>WALL_LIMIT_SECONDS-120:
                raise RuntimeError('Global decision or time budget exhausted.')
            calls+=1
            response=worker.decide.remote(body)
            if calls%20==0:print(f'Controller decisions received: {calls}/{MAX_CALLS}',flush=True)
            return response
        dev=output/'dev';test=output/'test';freeze=output/'frozen.json'
        experiment.run(config,prompt,experiment.DEFAULT_DATA,'dev',None,dev,'live',client=client)
        experiment.evaluate(dev)
        experiment.freeze_run(dev,freeze)
        print('All 60 dev complete; settings frozen. Starting test without changing settings.',flush=True)
        experiment.run(config,prompt,experiment.DEFAULT_DATA,'test',None,test,'live',freeze=freeze,client=client)
        # Verify complete evidence before app closure; detailed reporting is offline afterward.
        experiment.completed_records(test)
    if watchdog is not None:
        watchdog.terminate();watchdog.wait(timeout=10)
    experiment.save(output/'completion.json',{'app_context_closed':True,'controller_calls':calls,'client_wall_seconds':time.monotonic()-started,'finished_utc':experiment.now()})
    experiment.evaluate(test)
    print(json.dumps({'status':'complete','controller_calls':calls,'app_id':app.app_id,'output':str(output)}),flush=True)

if __name__=='__main__':main()
