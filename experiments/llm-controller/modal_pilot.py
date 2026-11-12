"""Bounded private Modal RPC pilot. No public endpoint or hosted token API."""
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
LIMIT=10
MAX_CALLS=2*LIMIT
image=modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.6.0','transformers==4.51.3','accelerate==1.6.0','numpy==2.2.5')
app=modal.App('cs238-llm-controller-dev-pilot')

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
    ROOT=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(ROOT/'src'))
    from stop_repair_retry import llm_experiment as experiment
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prior-spend-usd',type=float,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:raise SystemExit('Use --execute only for an authorized paid pilot.')
    if not 0<=args.prior_spend_usd<20:raise SystemExit('Invalid prior spending.')
    output=Path(args.output).resolve()
    if output.exists():raise SystemExit('Refusing automatic paid replay or overwrite. Inspect the original journal before recovery.')
    per_second=.000694+4*.0000131+48*.00000222
    # Pessimistically reserve a full cold start and full timeout for every decision and metadata call.
    conservative_compute=(MAX_CALLS+1)*(600+120+10)*per_second
    if args.prior_spend_usd+conservative_compute+1>=20:raise SystemExit('Conservative total bound exceeds $20 cap.')
    rows=experiment.load_dataset(experiment.DEFAULT_DATA,'dev')[:LIMIT]
    config=experiment.load_config(ROOT/'experiments/llm-controller/config.example.json')
    config.update(endpoint='https://modal.com/apps/gancone/main/cs238-llm-controller-dev-pilot',model=MODEL_ID,
                  response_format='none',billing_mode='compute',compute_budget_confirmed=True,max_requests=MAX_CALLS,max_run_seconds=900,timeout_seconds=120)
    output.mkdir(parents=True)
    experiment.save(output/'config.json',config)
    experiment.save(output/'collection-protocol.json',{'status':'frozen_before_inference','split':'dev','question_ids':[r['id'] for r in rows],
      'selection':'first 10 development questions in existing file order, no outcome selection','domain_counts':{d:sum(r['domain']==d for r in rows) for d in ['math','science']},
      'max_controller_calls':MAX_CALLS,'model':MODEL,'revision':REVISION,'driver_sha256':experiment.sha(__file__),
      'prompt_sha256':experiment.sha(experiment.DEFAULT_PROMPT),'prior_reported_project_usd':args.prior_spend_usd,'user_total_cap_usd':20,
      'resource_bound':{'gpu':'A100-80GB','containers':1,'cpu_cores':4,'ram_gib':48,'startup_timeout_seconds':600,'call_timeout_seconds':120,'scaledown_seconds':10,'automatic_function_retries':0,'rate_usd_per_second':per_second,'pessimistic_compute_usd':conservative_compute,'image_egress_reserve_usd':1},
      'transport':'private authenticated Modal RPC; endpoint field is descriptive, not an HTTP address used for inference',
      'latency':'Experiment latency records end-to-end RPC time. Returned measurement also records GPU generation time. Model loading is recorded as collection startup, not per-call inference.',
      'billing':'standard Modal compute billed on gancone; provider ledger checked before/after; no Shared Endpoint token billing','created_utc':experiment.now()})
    started=time.monotonic()
    with modal.enable_output(),app.run():
        experiment.save(output/'modal-app.json',{'app_id':app.app_id,'started_utc':experiment.now(),'profile':'gancone'})
        worker=Controller();metadata=worker.metadata.remote()
        experiment.save(output/'backend.json',metadata)
        calls=0
        def client(body):
            nonlocal calls
            if calls>=MAX_CALLS:raise RuntimeError('Pilot call limit reached.')
            calls+=1
            response=worker.decide.remote(body)
            print(f'Saved controller response {calls}/{MAX_CALLS} maximum',flush=True)
            return response
        experiment.run(config,experiment.DEFAULT_PROMPT.read_text(),experiment.DEFAULT_DATA,'dev',LIMIT,output,'live',client=client)
    experiment.save(output/'collection-completion.json',{'app_context_closed':True,'controller_calls':calls,'client_wall_seconds':time.monotonic()-started,'finished_utc':experiment.now()})
    print(json.dumps({'status':'complete','output':str(output),'controller_calls':calls,'app_id':app.app_id}),flush=True)

if __name__=='__main__':main()
