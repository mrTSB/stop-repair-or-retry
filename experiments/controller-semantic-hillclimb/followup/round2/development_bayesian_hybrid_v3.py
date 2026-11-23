"""Same36-policy C comparison with a complete sealed frozen-observer bank.

Preserve both earlier incomplete cache-only attempts. The additional144 local
offline forwards are disclosed separately; policy replay makes no new forward
and charges every actually used observation at its measured full input size.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import bayesian_observer_features as features

original,io = features.original,features.io
SCHEMA = 'Bayesian-POMDP-hybrid-completed-observer-reused-DEV-v3'


@contextmanager
def configured():
    old_inputs,old_schema,old_write = original.inputs,original.SCHEMA,io.write
    def inputs(args):
        bundle,summary,source,record = old_inputs(args)
        folder = args.previous.parent/'bayesian-observer-features-v1'
        done = io.read(folder/'complete.json')
        if done['schema'] != features.SCHEMA or done['registration_sha256'] != io.sha(folder/'registration.json'):
            raise ValueError('Require the completed registered frozen-observer bank')
        io.verify({str(folder/k):v for k,v in done['digests'].items()})
        registered = io.read(folder/'registration.json');io.verify(registered['source_files'])
        if (registered['previous'] != str(args.previous.resolve()) or registered['output'] != str(folder.resolve())
                or registered['prior_complete_sha256'] != io.sha(args.previous/'complete.json')
                or registered['model_fitting'] or registered['test_used']):
            raise ValueError('Changed observer-bank public prefix or model provenance')
        expanded = original.canonical.previous.previous.cache_values((folder/'private-neural-caches.pkl').read_bytes())
        for asset,entries in bundle[4].items():
            for key,value in entries.items():
                if any(not np.array_equal(a,b) for a,b in zip(expanded[asset][key],value)):
                    raise ValueError('Never change an inherited observation to improve a result')
        bundle = (*bundle[:4],expanded,*bundle[5:])
        receipt = io.read(folder/'summary.json')
        record.update(schema=SCHEMA,source_files=io.sources(sys.modules[__name__]),
            observer_bank_complete_sha256=io.sha(folder/'complete.json'),
            offline_feature_extraction=receipt,new_neural_calls_scope='Only this policy-replay process; offline feature extraction disclosed separately',
            prior_incomplete_attempts_preserved=['development-bayesian-model-hybrid-v1','development-bayesian-model-hybrid-v2'])
        return bundle,summary,source,record
    def write(path,value):
        if Path(path).name == 'summary.json' and isinstance(value,dict) and value.get('schema') == SCHEMA:
            folder = Path(value['C_candidate']['model_path']).parent.parent/'bayesian-observer-features-v1'
            value = {**value,'offline_feature_extraction':io.read(folder/'summary.json'),
                'new_neural_calls_scope':'This cache-only policy replay; additional frozen local offline forwards are disclosed separately'}
        return old_write(path,value)
    original.inputs,original.SCHEMA,io.write = inputs,SCHEMA,write
    try:yield
    finally:original.inputs,original.SCHEMA,io.write = old_inputs,old_schema,old_write


if __name__ == '__main__':
    with configured():original.main()
