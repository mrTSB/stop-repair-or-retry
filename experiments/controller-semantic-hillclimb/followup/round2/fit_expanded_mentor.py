"""Register and CPU-fit the expanded weighted320-root POMDP candidates.

Requires the fully sealed960-row neural extraction. No encoder or LLM call
can run here. The original A8 is a construction anchor, not a claim to be
the strongest current A; a later explicitly reviewed development wrapper
must substitute/retain the strongest completed A before prospective use.
"""
import argparse
import copy
import json
from pathlib import Path
import pickle
import platform
import sys

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

import expanded_mentor_core as core
import expanded_mentor_inputs as inputs
import expanded_mentor_io as io
import expanded_mentor_spec as spec
import expanded_review_metadata as metadata_fit
import extract_expanded_mentor as extractor
import fit_metadata_review as previous_metadata
import metadata_review_hybrid as metadata_runtime

SCHEMA = 'round2-expanded-weighted-controller-fit-v1'
PATHS = (*extractor.ARG_PATHS, 'neural', 'diagnostic', 'old_data', 'parent_models')


def bind_registered_input_identity(bundle, registered):
    """Separate the frozen128 loader's ambient import inventory from inputs.

That historical helper includes every loaded project module in its returned
source_files map. The extraction and fitting entrypoints necessarily load
different callers. Validate both maps, compare every other input/provenance
field exactly, and retain both maps in an explicit normalization receipt.
No source file is removed from this fit's own static dependency seal.
"""
    actual, expected = copy.deepcopy(bundle['identity']), registered['input_identity']
    observed_sources = actual['validated_parent_provenance']['source_files']
    registered_sources = expected['validated_parent_provenance']['source_files']
    io.verify(observed_sources); io.verify(registered_sources)
    actual['validated_parent_provenance']['source_files'] = registered_sources
    if actual != expected:
        raise ValueError('Expanded extraction inputs changed beyond the ambient caller source inventory')
    receipt = {'field': 'validated_parent_provenance.source_files',
        'reason': 'Frozen historical loader inventories ambient loaded modules, including its caller',
        'registered_sources': registered_sources, 'fitting_entrypoint_sources': observed_sources,
        'all_other_input_fields_exactly_equal': True, 'both_source_maps_hash_verified': True,
        'current_fit_static_source_seal_retained': True}
    return {**bundle, 'identity': actual, 'ambient_source_manifest_normalization': receipt}


def completed_inputs(args):
    bundle = inputs.load_inputs(args)
    neural_registration = io.read(args.neural/'registration.json')
    if neural_registration['paths'] != {name: str(getattr(args, name).resolve()) for name in extractor.ARG_PATHS}:
        raise ValueError('Expanded extraction and fitting use different input paths')
    bundle = bind_registered_input_identity(bundle, neural_registration)
    arrays, completion = extractor.verify_complete(args.neural, bundle, io.sha(args.neural/'registration.json'))
    original, _, _, evidence, _, _ = previous_metadata.load_diagnostic(args)
    records = metadata_fit.merge_records(original, bundle)
    evidence.update({str((args.neural/name).resolve()): io.sha(args.neural/name) for name in
        ('registration.json', 'neural-complete.json', 'neural-receipt.json', 'metadata.jsonl', 'neural.npz')})
    evidence.update(bundle['identity']['input_evidence'])
    return bundle, arrays, records, evidence, completion


def contract(args, bundle, records, evidence, completion):
    return {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': evidence, 'paths': {name: str(getattr(args, name).resolve()) for name in PATHS},
        'output': str(args.output.resolve()), 'input_identity': bundle['identity'],
        'neural_complete_sha256': io.sha(args.neural/'neural-complete.json'),
        'metadata_records_sha256': inputs.fingerprint(records),
        'ambient_source_manifest_normalization': bundle['ambient_source_manifest_normalization'],
        'proposal_sha256': io.sha(args.proposal), 'smoothing': list(spec.SMOOTHING),
        'review_observations': ['reported', 'semantic', 'nested-paid-metadata'],
        'fit_questions': 512, 'paired_root_questions': 320, 'metadata_records': 448,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'calibration_weight_normalization': 'Total mass equals independent eligible questions',
        'pooled_dynamics_subject_weights': spec.POOLED, 'subject_local_question_weight': 1.,
        'plain_only_dynamics': True, 'OOF_emissions': True, 'pure_more_data_ablation': False,
        'neural_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False,
        'base_is_strongest_completed_A': False, 'base_role': 'Frozen construction anchor only',
        'prospective_support': 'Requires a separate expansion-aware provenance extension; old guards stay unchanged'}


def save_json(path, value):
    if path.exists():
        if io.read(path) != value: raise ValueError('Preserved fitted receipt differs; use a new variant')
    else: io.write(path, value)


def save_arrays(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if (set(old.files) != set(arrays) or any(old[name].dtype != value.dtype
                or old[name].shape != value.shape or old[name].tobytes() != value.tobytes()
                for name, value in arrays.items())):
                raise ValueError('Preserved CPU fit arrays differ; never overwrite completed outputs')
    else: io.npz(path, arrays)


def save_model(path, model):
    expected = io.structural_digest(model)
    if path.exists():
        restored = pickle.loads(path.read_bytes())
        if io.structural_digest(restored) != expected:
            raise ValueError('Preserved fitted model differs numerically/structurally')
    else:
        data = pickle.dumps(model, protocol=5)
        if io.structural_digest(pickle.loads(data)) != expected:
            raise ValueError('Pickle roundtrip changed learned state')
        io.atomic_bytes(path, data)
    return {'sha256': io.sha(path), 'saved_structural_sha256': expected,
            'historical_byte_identity_claim': False}


def verify_complete(output, registration_sha256):
    completed = io.read(output/'complete.json')
    registered = io.read(output/'registration.json')
    if (completed['schema'] != SCHEMA or completed['registration_sha256'] != registration_sha256
        or io.sha(output/'registration.json') != registration_sha256 or completed['fit_questions'] != 512
        or completed['paired_review_questions'] != 320 or completed['metadata_questions'] != 320
        or completed['metadata_records'] != 448 or completed['development_or_test_used']
        or completed['neural_calls'] or completed['cloud_calls'] or len(completed['models']) != 6):
        raise ValueError('Invalid completed expanded fit')
    io.verify(registered['source_files']); io.verify(registered['input_evidence'])
    io.verify({str(output/name): value for name, value in completed['artifacts'].items()})
    for record in [*completed['models'], completed['regular_A4_artifact']]:
        path = output/(record['key']+'.pkl')
        if io.sha(path) != record['sha256'] or io.structural_digest(pickle.loads(path.read_bytes())) != record['saved_structural_sha256']:
            raise ValueError('Changed saved controller or learned-state identity')
    return completed


def fit_models(output, registration_sha256, bundle, outputs, records):
    """Only cached features and TRAIN labels enter these deterministic CPU fits."""
    rows, trees, judges = bundle['rows'], bundle['trees'], bundle['judges']
    # Actual recipe/temperature are already hash-bound in the input identity.
    base = pickle.loads(Path(bundle['base_path']).read_bytes())
    directory = Path(base.classifier.directory)
    recipe = io.read(directory/'recipe.json')
    if io.sha(directory/'recipe.json') != bundle['identity']['encoder_recipe_sha256']:
        raise ValueError('Construction base and expanded child observer assets differ')
    observer = core.perception.Observer(str(directory.resolve()), bundle['identity']['inherited_temperature'], device='cpu')
    observer, oof, calibration, calibration_rows, probabilities = core.fit_observer(
        rows, trees, bundle['metadata'], outputs, observer)
    save_json(output/'calibration.json', calibration)
    alignment_path = output/'calibration-alignment.jsonl'
    if alignment_path.exists():
        if [json.loads(line) for line in alignment_path.read_text().splitlines()] != calibration_rows:
            raise ValueError('Preserved calibration alignment changed')
    else: io.jsonl(alignment_path, calibration_rows)
    save_arrays(output/'calibration-oof.npz', {'probabilities': probabilities})
    encoder_price = recipe['parameters']/1e9*recipe['max_length']/3.09
    regular = core.plain_regular(rows, trees, observer, oof, encoder_price, calibration)
    a4_key = 'regular-plain-A4'
    a4 = {'key': a4_key, **save_model(output/(a4_key+'.pkl'), regular),
          'policy_module': 'dynamic_grounded_pomdp', 'role': 'Unevaluated pure-A candidate',
          'review_observer': False, 'never_development_selected_here': True}
    learned = metadata_fit.fit(records, registration_sha256)
    save_json(output/'metadata-fit.json', learned['receipt'])
    save_json(output/'trained-linear.json', {'state': learned['adapter'].state(), 'receipt': learned['adapter'].receipt})
    save_model(output/'adapter.pkl', learned['adapter'])
    save_arrays(output/'metadata-oof.npz', {'raw': learned['raw'], 'nested': learned['nested']})
    fold_models = []
    for fold, outer in enumerate(learned['outer_models']):
        name = f'metadata-outer{fold}.pkl'; fold_models.append({'file': name, **save_model(output/name, outer)})
        for inner, model in enumerate(learned['inner_models'][fold]):
            name = f'metadata-outer{fold}-inner{inner}.pkl'
            fold_models.append({'file': name, **save_model(output/name, model)})
    save_json(output/'metadata-fold-models.json', fold_models)
    signals, aligned = metadata_fit.root_signals(records, learned['nested'], bundle)
    save_json(output/'metadata-root-alignment.json', aligned)
    models = []
    for smoothing in spec.SMOOTHING:
        for semantic in (False, True):
            fitted, pair_records = core.fit_mentor(rows, trees, judges, copy.deepcopy(regular), oof,
                bundle['metadata'], outputs, encoder_price, smoothing, semantic)
            model = core.build(copy.deepcopy(base), fitted, rows, trees, judges, smoothing)
            key = f'smooth{smoothing:g}-review{int(semantic)}'
            model.fit_receipt['registration_sha256'] = registration_sha256
            models.append({'key': key, **save_model(output/(key+'.pkl'), model),
                'smoothing': smoothing, 'semantic_review': semantic, 'metadata_review': False,
                'policy_module': 'hybrid_repairability', 'fit_module': core.__name__,
                'paired_records_sha256': inputs.fingerprint(pair_records)})
        fitted, pair_records = core.fit_mentor(rows, trees, judges, copy.deepcopy(regular), oof,
            bundle['metadata'], outputs, encoder_price, smoothing, False, review_signals=signals)
        parent = core.build(copy.deepcopy(base), fitted, rows, trees, judges, smoothing)
        model = metadata_runtime.build(parent, copy.deepcopy(learned['adapter']), bundle['base_spec']['settings'],
            {'schema': SCHEMA, 'registration_sha256': registration_sha256,
             'weighted_core_module': core.__name__, 'expanded_metadata_module': metadata_fit.__name__})
        model.fit_receipt.update(expanded_fit_schema=SCHEMA, fit_module=__name__,
            registration_sha256=registration_sha256, base_is_strongest_completed_A=False)
        key = f'metadata-smooth{smoothing:g}'
        models.append({'key': key, **save_model(output/(key+'.pkl'), model),
            'smoothing': smoothing, 'semantic_review': False, 'metadata_review': True,
            'policy_module': 'metadata_review_hybrid', 'fit_module': __name__,
            'paired_records_sha256': inputs.fingerprint(pair_records),
            'review_neural_calls': 0, 'regular_child_neural_calls_retained': True})
    return models, a4


def run(args):
    with inputs.parent_fit.no_neural_inference(), threadpool_limits(limits=1):
        bundle, arrays, records, evidence, completion = completed_inputs(args)
        expected = contract(args, bundle, records, evidence, completion)
        if args.mode == 'register':
            if args.output.exists(): raise ValueError('Use a new controller-fitting directory')
            args.output.mkdir(parents=True)
            io.write(args.output/'registration.json', expected)
            print({'registered': True, 'registration_sha256': io.sha(args.output/'registration.json')}, flush=True)
            return
        if not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256:
            raise ValueError('Fit requires the explicit reviewed registration SHA')
        if io.read(args.output/'registration.json') != expected:
            raise ValueError('Changed CPU fit registration/evidence/runtime')
        with io.lock(args.output/'.fit.lock'):
            if (args.output/'complete.json').exists():
                verify_complete(args.output, args.registration_sha256); return
            identity = {'schema': SCHEMA, 'source_sha256': io.sha(__file__),
                'registration_sha256': args.registration_sha256, 'input_identity': bundle['identity'],
                'ambient_source_manifest_normalization': bundle['ambient_source_manifest_normalization'],
                'base_A': bundle['base_spec'], 'neural_complete_sha256': io.sha(args.neural/'neural-complete.json'),
                'metadata_records_sha256': inputs.fingerprint(records), 'policy_modules': [
                    'hybrid_repairability', 'metadata_review_hybrid'], 'fit_questions': 512,
                'paired_review_questions': 320, 'neural_calls': 0, 'cloud_calls': 0,
                'development_or_test_used': False, 'historical_byte_identity_claim': False}
            save_json(args.output/'identity.json', identity)
            feature_path = args.output/'private-metadata-features.jsonl'
            if feature_path.exists():
                if [json.loads(line) for line in feature_path.read_text().splitlines()] != records:
                    raise ValueError('Preserved metadata training features changed')
            else: io.jsonl(feature_path, records)
            models, a4 = fit_models(args.output, args.registration_sha256, bundle, arrays, records)
            io.verify(expected['source_files']); io.verify(evidence)
            artifacts = {p.name: io.sha(p) for p in sorted(args.output.iterdir())
                if p.is_file() and not p.name.startswith('.') and p.name != 'complete.json'}
            completed = {'schema': SCHEMA, 'models': models, 'regular_A4_artifact': a4,
                'registration_sha256': args.registration_sha256, 'identity_sha256': io.sha(args.output/'identity.json'),
                'fit_questions': 512, 'paired_review_questions': 320, 'metadata_questions': 320,
                'metadata_records': 448, 'base_A': bundle['base_spec'], 'base_is_strongest_completed_A': False,
                'neural_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False,
                'pure_more_data_ablation': False, 'artifacts': artifacts}
            io.write(args.output/'complete.json', completed)
            verify_complete(args.output, args.registration_sha256)
            print({'complete': True, 'models': len(models), 'fit_questions': 512, 'paired_roots': 320}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('register', 'fit'))
    for name in (*PATHS, 'output'): p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--registration-sha256')
    run(p.parse_args())


if __name__ == '__main__': main()
