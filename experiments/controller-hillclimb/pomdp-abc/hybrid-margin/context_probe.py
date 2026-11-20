"""Run the same development search with learned contextual fix/damage transitions."""

import argparse
import hashlib
import json
from pathlib import Path

import routing_probe
import contextual_pomdp


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--leaves',nargs='+',type=int,default=[12,32])
    parser.add_argument('--smoothing',nargs='+',type=float,default=[32.])
    parser.add_argument('--transition-leaf',type=int,default=24)
    parser.add_argument('--context-weight',type=float,default=.5)
    parser.add_argument('--late-data',action='store_true')
    parser.add_argument('--wait',type=int,default=43200)
    args=parser.parse_args()
    configuration={'transition_leaf':args.transition_leaf,'context_weight':args.context_weight,
                   'context_code_sha256':hashlib.sha256(Path(contextual_pomdp.__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=True);receipt=args.output/'context-configuration.json'
    if receipt.exists() and json.loads(receipt.read_bytes())!=configuration:
        raise ValueError('Context model settings changed; use a separate experiment output')
    receipt.write_text(json.dumps(configuration,indent=2)+'\n')
    def fit(*a,**kw):
        return contextual_pomdp.fit(*a,**kw,transition_leaf=args.transition_leaf,context_weight=args.context_weight)
    def evaluate(row,tree,judges,model,settings,method):
        settings['contextual_transitions']=True;settings['context_weight']=args.context_weight
        return contextual_pomdp.evaluate(row,tree,judges,model,settings,method)
    # Dispatch only A/C fitting and evaluation; the independently selected LLM B is unchanged.
    routing_probe.fit=fit;routing_probe.evaluate=evaluate
    routing_probe.run(args)
