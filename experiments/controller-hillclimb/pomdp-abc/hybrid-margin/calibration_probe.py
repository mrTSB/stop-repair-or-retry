"""Search C with a trained review observation channel; leave the LLM B independent."""

import argparse
import hashlib
import json
from pathlib import Path

import routing_probe
import review_calibration


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--leaves',nargs='+',type=int,default=[32])
    parser.add_argument('--smoothing',nargs='+',type=float,default=[16.,64.])
    parser.add_argument('--review-leaf',type=int,default=24)
    parser.add_argument('--transition-leaf',type=int,default=24)
    parser.add_argument('--context-weight',type=float,default=.7)
    parser.add_argument('--late-data',action='store_true')
    parser.add_argument('--wait',type=int,default=21600)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    configuration={'review_leaf':args.review_leaf,'transition_leaf':args.transition_leaf,
                   'context_weight':args.context_weight,
                   'review_source_sha256':hashlib.sha256(Path(review_calibration.__file__).read_bytes()).hexdigest()}
    receipt=args.output/'calibration-configuration.json'
    if receipt.exists() and json.loads(receipt.read_bytes())!=configuration:raise ValueError('Changed calibration trial')
    receipt.write_text(json.dumps(configuration,indent=2)+'\n')
    def fit(*a,**kw):
        return review_calibration.fit(*a,**kw,review_leaf=args.review_leaf,
                   transition_leaf=args.transition_leaf,context_weight=args.context_weight)
    from contextual_pomdp import evaluate as contextual
    def evaluate(row,tree,judges,model,settings,method):
        settings.update(review_calibration=True,contextual_transitions=True,context_weight=args.context_weight)
        return contextual(row,tree,judges,model,settings,method)
    routing_probe.fit=fit;routing_probe.evaluate=evaluate
    routing_probe.run(args)
