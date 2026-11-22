"""Construct expanded C candidates around a separately selected strongest A.

Only the unqueried A component changes. The320-root mentor/transfer and448-row
metadata head remain numerically identical to their sealed TRAIN artifacts.
"""
import copy
from pathlib import Path
import pickle

import expanded_mentor_io as io
import fit_expanded_mentor as fitting
import normalize_expanded_metadata as normalized
import stronger_base_hybrid as plain_runtime
import stronger_metadata_review as metadata_builder
import development_stronger_metadata as previous
# These saved classes are invoked through pickle or subject/component routing.
# Explicit imports keep every supported runtime in the static source closure.
import factorized_repairability
import nonlinear_repairability
import subject_science_pomdp
import science_fusion_pomdp
import science_option_observer

SCHEMA = 'round2-expanded-strongest-A-construction-v1'


def retained_state(model):
    return io.structural_digest((model.mentor, model.transfer, getattr(model, 'metadata_adapter', None)))


def build(base, selected, parent, record, provenance):
    before = retained_state(parent)
    settings = selected['candidate'].get('settings', {})
    if record['metadata_review']:
        if record['policy_module'] != 'metadata_review_hybrid':
            raise ValueError('Metadata parent requires its explicit runtime')
        model = metadata_builder.build(base, selected['module'], settings, parent, provenance)
        module = 'metadata_review_hybrid'
    else:
        if record['policy_module'] != 'hybrid_repairability':
            raise ValueError('Raw mentor parent requires its explicit runtime')
        model = plain_runtime.build(base, selected['module'], settings, parent, provenance)
        module = 'stronger_base_hybrid'
    model.fit_receipt.update(builder_module=__name__, policy_module=module,
        expanded_training_questions=512, expanded_reviewed_roots=320,
        expanded_metadata_records=448 if record['metadata_review'] else 0,
        mentor_refitted=False, transfer_refitted=False, metadata_adapter_refitted=False)
    if retained_state(model) != before or retained_state(parent) != before:
        raise ValueError('Strongest-A construction changed a trained mentor/transfer/adapter')
    return model, module, before


def construct(output, parent_directory, selected, references, pending, registration_sha256):
    parent_complete = normalized.verify_complete(parent_directory)
    io.verify({selected['summary']: selected['summary_sha256'],
        str(Path(selected['summary']).parent/'identity.json'): selected['identity_sha256'],
        selected['model_path']: selected['model_sha256']})
    directory = output/'models'; directory.mkdir(exist_ok=True)
    identity = {'schema': SCHEMA, 'builder_module': __name__, 'source_sha256': io.sha(__file__),
        'registration_sha256': registration_sha256, 'base_selection': selected,
        'A_references': references, 'pending_optional_A': pending,
        'normalized_parent_directory': str(parent_directory.resolve()),
        'normalized_parent_complete_sha256': io.sha(parent_directory/'complete.json'),
        'training_questions': 512, 'reviewed_roots': 320, 'metadata_records': 448,
        'mentor_refitted': False, 'transfer_refitted': False, 'metadata_adapter_refitted': False,
        'test_used': False, 'neural_calls': 0, 'cloud_calls': 0}
    fitting.save_json(directory/'identity.json', identity)
    if (directory/'complete.json').exists():
        complete = io.read(directory/'complete.json')
        if complete['identity_sha256'] != io.sha(directory/'identity.json'):
            raise ValueError('Changed completed construction identity')
        for record in complete['models']:
            path = directory/(record['key']+'.pkl')
            if io.sha(path) != record['sha256']:
                raise ValueError('Changed constructed expanded model')
        for asset in complete['observer_assets']:
            io.verify({asset['recipe_path']: asset['recipe_sha256'], **asset['weights']})
        return complete
    base = pickle.loads(Path(selected['model_path']).read_bytes())
    models, assets = [], {}
    for record in parent_complete['models']:
        source = Path(record['model_path'])
        if io.sha(source) != record['sha256']: raise ValueError('Changed expanded TRAIN parent model')
        parent = pickle.loads(source.read_bytes())
        model, module, preserved = build(copy.deepcopy(base), selected, parent, record,
            {'schema': SCHEMA, 'construction_identity_sha256': io.sha(directory/'identity.json'),
             'expanded_parent_model_path': str(source), 'expanded_parent_sha256': record['sha256']})
        assets.update(previous.validate_assets(model))
        key = record['key']; path = directory/(key+'.pkl')
        saved = fitting.save_model(path, model)
        loaded = pickle.loads(path.read_bytes())
        if retained_state(loaded) != preserved: raise ValueError('Saved construction changed retained learned state')
        models.append({'key': key, **saved, 'parent_model_path': str(source), 'parent_sha256': record['sha256'],
            'smoothing': record['smoothing'], 'semantic_review': record['semantic_review'],
            'metadata_review': record['metadata_review'], 'policy_module': module,
            'retained_state_sha256': preserved, 'builder_module': __name__})
    complete = {'schema': SCHEMA, 'models': models, 'base_selection': selected,
        'base_A': selected['candidate'], 'identity_sha256': io.sha(directory/'identity.json'),
        'parent_complete_sha256': io.sha(parent_directory/'complete.json'),
        'observer_assets': list(assets.values()), 'mentor_refitted': False, 'transfer_refitted': False,
        'metadata_adapter_refitted': False, 'neural_calls': 0, 'cloud_calls': 0, 'test_used': False}
    io.write(directory/'complete.json', complete)
    return complete
