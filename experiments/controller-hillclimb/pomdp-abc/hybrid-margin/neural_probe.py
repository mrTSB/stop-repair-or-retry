"""Compare POMDP policies after question-grouped neural verifier training completes."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from margin_data import read
import routing_probe
import neural_pomdp


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verifier',type=Path,required=True)
    parser.add_argument('--leaves',nargs='+',type=int,default=[0])
    parser.add_argument('--smoothing',nargs='+',type=float,default=[16.,64.])
    parser.add_argument('--contextual',action='store_true')
    parser.add_argument('--transition-leaf',type=int,default=24)
    parser.add_argument('--context-weight',type=float,default=.7)
    parser.add_argument('--late-data',action='store_true')
    parser.add_argument('--wait',type=int,default=43200)
    args=parser.parse_args()
    deadline=time.monotonic()+args.wait
    while not (args.verifier/'training-summary.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Verifier training incomplete')
        time.sleep(5)
    metadata=read(args.verifier/'examples.jsonl')
    scores=np.load(args.verifier/'oof-probabilities.npz')['probabilities']
    if len(metadata)!=len(scores) or not np.allclose(scores.sum(axis=1),1.,atol=1e-5):
        raise ValueError('Incomplete out-of-question verifier predictions')
    predictions={(r['id'],r['kind'],r['path']):p for r,p in zip(metadata,scores)}
    args.output.mkdir(parents=True,exist_ok=True)
    receipt=args.output/'neural-configuration.json'
    recipe=json.loads((args.verifier/'full/recipe.json').read_bytes())
    configuration={'weights_sha256':recipe['weights_sha256'],
                   'predictions_sha256':hashlib.sha256((args.verifier/'oof-probabilities.npz').read_bytes()).hexdigest(),
                   'contextual':args.contextual,'context_weight':args.context_weight,'transition_leaf':args.transition_leaf}
    if receipt.exists() and json.loads(receipt.read_bytes())!=configuration:raise ValueError('Changed neural probe; use a new output')
    receipt.write_text(json.dumps(configuration,indent=2)+'\n')
    perception=args.data/'train-neural-observations.jsonl'
    encoded=''.join(json.dumps({'id':r['id'],'kind':r['kind'],'path':r['path'],'fold':r['fold'],
                                'probabilities':p.tolist()})+'\n' for r,p in zip(metadata,scores))
    if perception.exists() and perception.read_text()!=encoded:raise ValueError('Out-of-question perception archive changed')
    perception.write_text(encoded)
    def fit(rows,trees,judges,leaf,smoothing,**kwargs):
        return neural_pomdp.fit(rows,trees,judges,args.verifier/'full',predictions,leaf,smoothing,
                               contextual=args.contextual,transition_leaf=args.transition_leaf,
                               context_weight=args.context_weight,**kwargs)
    def evaluate(row,tree,judges,model,settings,method):
        settings['neural_observer']=True;settings['contextual_transitions']=args.contextual
        settings['context_weight']=args.context_weight
        return neural_pomdp.evaluate(row,tree,judges,model,settings,method)
    def warm(model,rows,trees):
        model.classifier.warm(model,[{**r,'answer':''} for r in rows],trees)
    ordinary_statistics=routing_probe.statistics
    def statistics(items):
        result=ordinary_statistics(items)
        for key in ('verifier_tokens','verifier_token_slots','verifier_calls'):
            result['mean_'+key]=float(np.mean([r.get(key,0.) for r in items]))
        return result
    routing_probe.fit=fit;routing_probe.evaluate=evaluate;routing_probe.warm=warm
    routing_probe.statistics=statistics
    routing_probe.run(args)
