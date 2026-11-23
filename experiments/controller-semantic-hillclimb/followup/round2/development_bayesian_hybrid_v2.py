"""Cache-identity extension after preserving the incomplete first C replay.

ScienceOptionObserver depends on public question/answer input, not path name.
An exactly identical serialized input may reuse its existing physical neural
output under another path key. No approximate match, label-based source choice,
new forward, changed model, changed policy grid or free inference bill is used.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

import development_bayesian_hybrid as original
from prospective_freeze import observer_slots

SCHEMA = 'Bayesian-POMDP-hybrid-exact-input-cache-alias-reused-DEV-v2'


def alias_entries(entries,requests):
    by_input = {}
    ambiguous = set()
    for key,value in sorted(entries.items()):
        identity = key[1],key[3]
        if identity in by_input and any(not np.array_equal(a,b) for a,b in zip(by_input[identity],value)):
            ambiguous.add(identity)
        else: by_input[identity] = value
    result,aliases,missing = dict(entries),[],[]
    for key in requests:
        if key in result: continue
        identity = key[1],key[3]
        if identity not in by_input or identity in ambiguous:
            missing.append(key);continue
        result[key] = by_input[identity]
        aliases.append(key)
    return result,{'aliases':aliases,'unavailable_inputs':missing,'conflicting_input_identities':sorted(ambiguous)}


def augment(bundle,source):
    rows,trees,judges,_,caches,*_ = bundle
    result = dict(caches)
    observers = {}
    for name in original.PARENTS:
        model = pickle.loads((source/'models'/(name+'.pkl')).read_bytes())
        for _,observer in observer_slots(model):
            asset = original.canonical.previous.previous.share_key(observer)
            if asset[1:] == ('science_option_observer','ScienceOptionObserver'):
                observers[asset] = observer
    report = []
    for asset,observer in sorted(observers.items()):
        observer._recipe = original.io.read(Path(observer.directory)/'recipe.json')
        requests = []
        for row in rows:
            if row['subject'] not in ('chemistry','physics'):continue
            safe = {**row,'answer':''}
            for path in ('','repair','retry','guided'):
                if path not in trees[row['id']]:continue
                entry = observer.entry(safe,path,original.policy.core.prefix(trees[row['id']],path))
                fingerprint = hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest()
                requests.append(('solver',row['id'],path,fingerprint))
        result[asset],receipt = alias_entries(caches[asset],requests)
        report.append({'asset':list(asset),'original_entries':len(caches[asset]),'requested_entries':len(requests),
            'alias_count':len(receipt['aliases']),'unavailable_inputs':len(receipt['unavailable_inputs']),
            'conflicting_inputs':len(receipt['conflicting_input_identities']),
            'receipt_sha256':original.canonical.previous.common.json.dumps(receipt,sort_keys=True)})
        report[-1]['receipt_sha256'] = hashlib.sha256(report[-1]['receipt_sha256'].encode()).hexdigest()
    return (*bundle[:4],result,*bundle[5:]),report


@contextmanager
def configured():
    previous_inputs,previous_schema = original.inputs,original.SCHEMA
    def inputs(args):
        bundle,summary,source,record = previous_inputs(args)
        bundle,aliases = augment(bundle,source)
        record.update(schema=SCHEMA,source_files=original.io.sources(sys.modules[__name__]),
            exact_input_cache_aliases=aliases,
            prior_incomplete_replay='development-bayesian-model-hybrid-v1 preserved; stopped on forbidden missing neural-cache entry',
            cache_aliasing_rule='Same frozen recipe, class, question and serialized neural input; all duplicate outputs must be bit-identical; original full inference charge retained')
        return bundle,summary,source,record
    original.inputs,original.SCHEMA = inputs,SCHEMA
    try:yield
    finally:original.inputs,original.SCHEMA = previous_inputs,previous_schema


if __name__ == '__main__':
    with configured():original.main()
