"""Reconstruct selected A from recorded training features, without neural inference.

This is a current-artifact reproducibility audit, not proof of historical pickle
identity. It never evaluates a policy, opens a development/test question file,
or estimates a generalization metric. The neural arrays are verified cached
inputs; neural weights themselves are outside this deterministic refit.
"""

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import platform
import sys

import numpy as np
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import repairability_pomdp as core
import factorized_repairability as factorized
from dynamic_grounded_pomdp import Observer
from evidence import Retriever, DEFAULT as CORPUS

PATHS = ('', 'repair', 'retry')
PLAIN = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
BASE_RECEIPT = ('state_count', 'state_bits', 'paired_questions', 'decision_nodes',
                'grouped_forecast_folds', 'calibration_C', 'smoothing',
                'actual_counterfactuals_accessed_at_inference', 'solver_budget', 'future_observations')
SUPPORTED_RUNTIME_MODULES = ('repairability_pomdp', 'factorized_repairability', 'nonlinear_repairability',
                             'continuous_repairability', 'science_option_observer', 'subject_science_pomdp',
                             'science_fusion_pomdp', 'development_refined_observer')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def project_sources(declare_supported=False):
    """Static import closure rooted at the auditor and supported runtime modules.

    Already-loaded modules resolve import locations, but are never themselves
    roots. Thus a combined test process cannot pull unrelated tests/drivers or
    an in-repository virtual environment into the sealed dependency graph.
    Function-local imports are included without executing their modules.
    """
    del declare_supported  # All explicitly supported branches are always roots.

    def allowed(filename):
        if not filename:
            return None
        path = Path(filename).resolve()
        if path.suffix != '.py' or not path.is_relative_to(PROJECT) or not path.is_file():
            return None
        if any(part.startswith('.') or part in ('site-packages', '__pycache__')
               for part in path.relative_to(PROJECT).parts):
            return None
        return path

    def locate(name):
        if not name:
            return None
        module = sys.modules.get(name)
        if module is not None:
            return allowed(getattr(module, '__file__', None))
        relative = Path(*name.split('.'))
        for directory in dict.fromkeys([str(HERE), *sys.path]):
            if not directory:
                directory = str(Path.cwd())
            for candidate in (Path(directory) / relative.with_suffix('.py'),
                              Path(directory) / relative / '__init__.py'):
                if candidate.is_file():
                    return allowed(candidate)
        return None

    pending = [(Path(__file__).resolve(), 'audit_a_refit')]
    for name in SUPPORTED_RUNTIME_MODULES:
        path = locate(name)
        if path is None:
            raise ValueError('Missing declared training-audit dependency: ' + name)
        pending.append((path, name))
    sources = set()
    while pending:
        path, module_name = pending.pop()
        if path in sources:
            continue
        sources.add(path)
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = module_name.split('.')[:-node.level]
                    name = '.'.join([*package, *([node.module] if node.module else [])])
                else:
                    name = node.module or ''
                names = [name, *[name + '.' + alias.name for alias in node.names if alias.name != '*']]
            for name in names:
                dependency = locate(name)
                if dependency is not None:
                    pending.append((dependency, name))
    return sorted(sources)


class Inputs:
    """Track every explicit read and refuse held-out question/response files."""
    def __init__(self):
        self.files = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        if path.name.startswith(('test-', 'dev-')) or path.name in ('test.jsonl', 'dev.jsonl'):
            raise ValueError('Training-only audit refuses held-out records: ' + str(path))
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and expected != digest:
            raise ValueError('Changed recorded input: ' + str(path))
        previous = self.files.get(str(path))
        if previous is not None and previous['sha256'] != digest:
            raise ValueError('Input changed during audit: ' + str(path))
        self.files[str(path)] = {'sha256': digest, 'bytes': len(raw)}
        return raw

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def lines(self, path, expected=None):
        return [json.loads(line) for line in self.read(path, expected).splitlines()]

    def sha(self, path, expected=None):
        self.read(path, expected)
        return self.files[str(Path(path).resolve())]['sha256']


def training_data(data, inputs):
    receipt = inputs.json(data / 'train-plain-complete.json')
    # Only the immutable sealed prefix is an input; collectors may append rows.
    with (data / 'train-observations.jsonl').open('rb') as stream:
        prefix = b''.join(next(stream) for _ in range(receipt['base_records']))
    if hashlib.sha256(prefix).hexdigest() != receipt['base_bank_sha256']:
        raise ValueError('Sealed training response prefix changed')
    rows = inputs.lines(data / 'splits/train-questions.jsonl')
    if len(rows) != 512 or len({r['id'] for r in rows}) != 512:
        raise ValueError('Require the complete registered 512-question training split')
    trees = {r['id']: {} for r in rows}
    for line in prefix.splitlines():
        item = json.loads(line)
        if item['kind'] == 'solver' and item['path'] in PLAIN:
            tree = trees[item['id']]
            if item['path'] in tree:
                raise ValueError('Duplicate sealed training response')
            tree[item['path']] = item['response']
    if any(set(tree) != set(PLAIN) for tree in trees.values()):
        raise ValueError('Incomplete paired training branches')
    return rows, trees, {'path': str((data / 'train-observations.jsonl').resolve()),
                         'records': receipt['base_records'], 'bytes': len(prefix),
                         'sha256': receipt['base_bank_sha256'], 'scope': 'sealed prefix only'}


def resolve_selection(path, registry, inputs, seen=None):
    seen = set() if seen is None else seen
    path = Path(path).resolve()
    if path in seen:
        raise ValueError('Cyclic selected-A summary chain')
    seen.add(path)
    summary = inputs.json(path)
    identity = inputs.json(path.parent / 'identity.json')
    if summary.get('test_used') or summary.get('fresh_data_used') or identity.get('test_used'):
        raise ValueError('Require a development-selected, test-unused A artifact')
    candidate = summary['selected_A']
    if candidate.get('source_summary'):
        source = Path(candidate['source_summary'])
        inputs.sha(source, candidate['source_summary_sha256'])
        return resolve_selection(source, registry, inputs, seen)
    module = candidate.get('policy_module')
    if module is None:
        if candidate.get('pickle') == 'selected-A.pkl':
            module = 'subject_science_pomdp'
        elif identity.get('policy_sha256') == inputs.sha(HERE / 'continuous_repairability.py'):
            module = 'continuous_repairability'
        else:
            module = 'repairability_pomdp'
    if module not in ('repairability_pomdp', 'continuous_repairability', 'subject_science_pomdp', 'science_fusion_pomdp'):
        raise ValueError('Unsupported A role; proposal pilots are not fitted A controllers')
    filename = candidate.get('pickle') or candidate.get('model', '') + '.pkl'
    model_path = Path(candidate.get('model_path', path.parent / filename))
    expected = identity.get('model_sha256', {}).get(model_path.name)
    if not model_path.exists():
        matches = [p for p in registry.rglob(filename) if expected and inputs.sha(p) == expected]
        if not matches:
            raise ValueError('Selected A model cannot be resolved from a recorded digest')
        model_path = sorted(matches)[0]
    model = pickle.loads(inputs.read(model_path, expected))
    return model, {'summary': str(path), 'summary_sha256': inputs.sha(path),
                   'model_path': str(model_path.resolve()), 'model_sha256': inputs.sha(model_path),
                   'policy_module': module, 'settings_sha256': fingerprint(candidate.get('settings', {})),
                   'historical_pickle_digest_available': expected is not None}, identity


@contextmanager
def forbid_neural_calls():
    """Fail closed if future input-building code accidentally loads/runs a NN."""
    from science_option_observer import ScienceOptionObserver
    from science_fusion_pomdp import FusionObserver
    classes = (Observer, ScienceOptionObserver, FusionObserver)
    saved = []
    def refused(*args, **kwargs):
        raise AssertionError('Neural loading/inference is forbidden in the training refit audit')
    for cls in classes:
        for name in ('load', 'values', 'warm'):
            if name in cls.__dict__:
                saved.append((cls, name, cls.__dict__[name]))
                setattr(cls, name, refused)
    try:
        yield
    finally:
        for cls, name, original in reversed(saved):
            setattr(cls, name, original)


def leaf_recipe(observer, inputs):
    if getattr(observer, 'components', None):
        recipes = [leaf_recipe(c, inputs) for c in observer.components]
        if (len(recipes) != 2 or [r['input_profile'] for r in recipes] !=
                ['full-options-openstax-v1', 'supergpqa-option-correctness-v1'] or
                recipes[0]['parameters'] != recipes[1]['parameters'] or
                [r['max_length'] for r in recipes] != [512, 512]):
            raise ValueError('Unsupported fusion recipe/cost composition')
        observer._recipe = {'parameters': sum(r['parameters'] for r in recipes), 'max_length': 512}
        return observer._recipe
    observer._recipe = inputs.json(Path(observer.directory) / 'recipe.json')
    observer.device = 'cpu'
    if observer._recipe.get('input_profile') == 'full-options-openstax-v1':
        inputs.sha(CORPUS / 'passages.jsonl', observer._recipe['corpus_sha256'])
        # Retriever writes only when its cached index is absent/stale. Refuse
        # that path so this audit cannot modify a live grounding asset.
        cached = pickle.loads(inputs.read(CORPUS / 'tfidf.pkl'))
        if cached['sha256'] != observer._recipe['corpus_sha256']:
            raise ValueError('Read-only corpus index is stale')
        observer._retriever = Retriever(CORPUS)
    return observer._recipe


def validate_cache(metadata, outputs, rows, trees, observer, all_ids):
    expected_keys = {(qid, path) for qid in all_ids for path in PATHS}
    keys = [(r['id'], r['path']) for r in metadata]
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError('Neural cache has missing, duplicate, or unexpected training nodes')
    for name, width in (('state', 4), ('plain', 8), ('guided', 8)):
        if outputs[name].shape != (len(keys), width) or not np.isfinite(outputs[name]).all():
            raise ValueError('Invalid cached neural output shape/values')
    indices = {key: i for i, key in enumerate(keys)}
    lookup, alignment = {}, []
    for row in rows:
        for path in PATHS:
            key = row['id'], path
            i = indices[key]
            paid = {p: trees[row['id']][p] for p in ('', path)}
            entry = observer.entry({**row, 'answer': ''}, path, paid)
            digest = fingerprint(entry)
            if (metadata[i]['input_sha256'], metadata[i]['prompt_sha256']) != (digest, paid[path]['prompt_sha256']):
                raise ValueError('Exact paid input or solver prompt changed')
            lookup[key] = (*[outputs[name][i] for name in ('state', 'plain', 'guided')], 0)
            alignment.append({'id': key[0], 'path': path, 'index': i, 'input_sha256': digest,
                              'prompt_sha256': metadata[i]['prompt_sha256']})
    return lookup, alignment


def learned_state(model):
    """All learned second-stage state; exclude NN assets and runtime caches."""
    arrays, structure = {}, {}
    def forecast(value, prefix):
        structure[prefix + '/class'] = type(value).__module__ + '.' + type(value).__name__
        if isinstance(value, Pipeline):
            structure[prefix + '/steps'] = [name for name, _ in value.steps]
            for name, step in value.steps:
                forecast(step, prefix + '/' + name)
        elif isinstance(value, factorized.Forecaster):
            arrays[prefix + '/classes_'] = np.asarray(value.classes_)
            for i, binary in enumerate(value.binary):
                forecast(binary, prefix + '/binary/' + str(i))
            for key, prior in sorted(value.priors.items()):
                arrays[prefix + '/priors/' + repr(key)] = np.asarray(prior)
        elif type(value).__module__ == 'nonlinear_repairability' and type(value).__name__ == 'Forecaster':
            arrays[prefix + '/temperature'] = np.asarray(value.temperature)
            arrays[prefix + '/classes_'] = np.asarray(value.classes_)
            forecast(value.estimator, prefix + '/estimator')
        elif type(value).__module__ == 'nonlinear_repairability' and type(value).__name__ == 'ConstantModel':
            arrays[prefix + '/label'] = np.asarray(value.label)
            arrays[prefix + '/classes_'] = np.asarray(value.classes_)
        elif type(value).__name__ == 'HistGradientBoostingClassifier':
            structure[prefix + '/parameters'] = value.get_params(deep=False)
            structure[prefix + '/loss_class'] = type(value._loss).__name__
            structure[prefix + '/link_class'] = type(value._loss.link).__name__
            if value._preprocessor is not None:
                raise ValueError('Nonlinear audit only supports the registered numeric-only features')
            for name in ('classes_', 'n_features_in_', 'n_trees_per_iteration_', 'n_iter_',
                         '_baseline_prediction', 'train_score_', 'validation_score_'):
                arrays[prefix + '/' + name] = np.asarray(getattr(value, name))
            mapper = value._bin_mapper
            for name in ('is_categorical_', 'n_bins_non_missing_', 'missing_values_bin_idx_'):
                arrays[prefix + '/bin_mapper/' + name] = np.asarray(getattr(mapper, name))
            for i, thresholds in enumerate(mapper.bin_thresholds_):
                arrays[prefix + '/bin_mapper/thresholds/' + str(i)] = np.asarray(thresholds)
            if mapper.known_categories is not None:
                raise ValueError('Unexpected categorical binning in registered nonlinear model')
            for iteration, predictors in enumerate(value._predictors):
                for tree_index, predictor in enumerate(predictors):
                    key = f'{prefix}/trees/{iteration}/{tree_index}'
                    for field in predictor.nodes.dtype.names:
                        arrays[key + '/nodes/' + field] = np.asarray(predictor.nodes[field])
                    for field in ('binned_left_cat_bitsets', 'raw_left_cat_bitsets'):
                        arrays[key + '/' + field] = np.asarray(getattr(predictor, field))
        elif isinstance(value, factorized.ConstantBinary):
            arrays[prefix + '/probability'] = np.asarray(value.probability)
            arrays[prefix + '/classes_'] = np.asarray(value.classes_)
        else:
            if type(value).__name__ not in ('StandardScaler', 'LogisticRegression'):
                raise ValueError('Unsupported learned forecaster type: ' + str(type(value)))
            structure[prefix + '/parameters'] = value.get_params(deep=False)
            for name in ('classes_', 'coef_', 'intercept_', 'n_iter_', 'mean_', 'scale_', 'var_',
                         'n_features_in_', 'n_samples_seen_'):
                if hasattr(value, name):
                    arrays[prefix + '/' + name] = np.asarray(getattr(value, name))
    forecast(model.forecaster, 'forecaster')
    for subject, dynamics in sorted(model.dynamics.items()):
        for name in ('prior', 'transition', 'emission', 'cost'):
            arrays['dynamics/' + subject + '/' + name] = np.asarray(getattr(dynamics, name))
    receipt_keys = BASE_RECEIPT + (('prior_smoothing', 'ipf_marginal_clip', 'emission_folds',
                                  'interaction_odds_ratios_preserved')
                                 if isinstance(model.forecaster, factorized.Forecaster) else ())
    if model.fit_receipt.get('fit_module') == 'nonlinear_repairability':
        receipt_keys = tuple(k for k in BASE_RECEIPT if k != 'calibration_C') + (
            'fit_module', 'forecaster_hyperparameters', 'temperature_grid', 'calibration',
            'feature_sha256', 'training_question_ids_sha256',
            'forecaster_and_temperature_exclude_own_question_in_emission_fit',
            'structural_transitions_and_costs_unchanged')
    structure['fit_receipt'] = {key: model.fit_receipt[key] for key in receipt_keys}
    return arrays, structure


def state_digest(arrays, structure):
    digest = hashlib.sha256(json.dumps(structure, sort_keys=True, allow_nan=False).encode())
    for key, value in sorted(arrays.items()):
        array = np.asarray(value, dtype='<f8')
        digest.update(json.dumps([key, list(array.shape)]).encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def compare_learned(actual, rebuilt, atol=1e-10, rtol=1e-10):
    left, ls = learned_state(actual); right, rs = learned_state(rebuilt)
    if ls != rs or set(left) != set(right):
        raise ValueError('Reconstructed learned-state structure differs')
    differences = {}
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Reconstructed parameter shape/finiteness differs: ' + key)
        difference = float(np.max(np.abs(a - b))) if a.size else 0.
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            raise ValueError(f'Reconstructed parameter differs: {key}; max absolute error={difference}')
        differences[key] = difference
    return {'passed': True, 'arrays_compared': len(left), 'scalar_values_compared': sum(a.size for a in left.values()),
            'max_absolute_error': max(differences.values(), default=0.), 'atol': atol, 'rtol': rtol,
            'exact_numeric_equality': all(value == 0. for value in differences.values()),
            'learned_state_sha256': state_digest(left, ls), 'reconstructed_state_sha256': state_digest(right, rs),
            'per_array_max_absolute_error': differences}


class RefitAudit:
    def __init__(self, args, inputs, rows, trees):
        self.args, self.inputs, self.rows, self.trees = args, inputs, rows, trees
        self.components, self.seen, self.cache = [], {}, {}

    def ordinary_lookup(self, observer, rows, auxiliary=False):
        folder = self.args.science if auxiliary else self.args.base
        identity = self.inputs.json(folder / 'identity.json')
        expected_train = identity.get('train_question_sha256', identity.get('new_train_questions_sha256'))
        self.inputs.sha(self.args.data / 'splits/train-questions.jsonl', expected_train)
        if auxiliary:
            summary = self.inputs.json(folder / 'summary.json')
            expected_recipe = summary['fit_receipt']['auxiliary_recipe_sha256']
            expected_neural = summary['fit_receipt']['science_neural_sha256']
            all_ids = {r['id'] for r in self.rows if r['subject'] in ('chemistry', 'physics')}
        else:
            expected_recipe = identity['encoder_recipe_sha256']; expected_neural = None
            all_ids = {r['id'] for r in self.rows}
        self.inputs.sha(Path(observer.directory) / 'recipe.json', expected_recipe)
        metadata = self.inputs.lines(folder / 'training-neural.jsonl')
        outputs = np.load(io.BytesIO(self.inputs.read(folder / 'training-neural.npz', expected_neural)), allow_pickle=False)
        lookup, alignment = validate_cache(metadata, outputs, rows, self.trees, observer, all_ids)
        return lookup, {'mode': 'disjoint frozen observer cached outputs', 'directory': str(folder.resolve()),
                        'alignment_sha256': fingerprint(alignment), 'aligned_nodes': len(alignment),
                        'metadata_sha256': self.inputs.sha(folder / 'training-neural.jsonl'),
                        'neural_sha256': self.inputs.sha(folder / 'training-neural.npz')}

    def v4_lookup(self, observer, rows, fit_receipt):
        # This variant rebuilds all new TRAIN metadata only. Unlike the original
        # development driver it does not open protected question records.
        from development_refined_observer import aligned_lookup, key
        from examples import examples
        from evidence import left_text
        folder = self.args.refined
        provenance = self.inputs.json(folder / 'oof-provenance.json')
        if provenance != fit_receipt['neural_observer_oof_provenance']:
            raise ValueError('V4 model and saved OOF provenance differ')
        verifier = Path(observer.directory).parent
        metadata = self.inputs.lines(verifier / 'train.jsonl', provenance['metadata_sha256'])
        contract = self.inputs.json(verifier / 'contract.json', provenance['training_contract_sha256'])
        self.inputs.json(verifier / 'training-summary.json', provenance['training_summary_sha256'])
        self.inputs.sha(self.args.data / 'splits/train-questions.jsonl', contract['question_split_sha256'])
        if contract['data_sha256'] != provenance['metadata_sha256'] or contract['folds'] != 3:
            raise ValueError('V4 input contract mismatch')
        for fold in (0, 1, 2, 'full'):
            recipe = self.inputs.json(verifier / str(fold) / 'recipe.json', provenance['fold_recipe_sha256'][str(fold)])
            if recipe['fold'] != fold or any(recipe.get(k) != v for k, v in contract.items()):
                raise ValueError('V4 question-excluded fold recipe mismatch')
        saved_alignment = self.inputs.lines(folder / 'oof-alignment.jsonl', provenance['alignment_sha256'])
        complete = self.inputs.json(self.args.data / 'train-ordinary-complete.json')
        prompts = {}
        for part, name, kind in (('base', 'train-observations.jsonl', 'original-review'),
                                  ('late', 'train-late-observations.jsonl', 'late-review')):
            for row in self.inputs.lines(self.args.data / name, complete[part + '_bank_sha256']):
                k = row['id'], 'solver' if row['kind'] == 'solver' else kind, row['path']
                if k in prompts:
                    raise ValueError('Duplicate V4 training prompt')
                prompts[k] = row['response']['prompt_sha256']
        expected = examples(self.args.data, 'train')
        by_id = {r['id']: r for r in rows}
        contexts = {r['id']: observer._retriever.context({**r, 'answer': ''})[0] for r in rows}
        for item in expected:
            item['left'] = left_text(by_id[item['id']], contexts[item['id']])
        outputs = np.load(io.BytesIO(self.inputs.read(verifier / 'oof.npz', provenance['oof_sha256'])), allow_pickle=False)
        lookup, alignment = aligned_lookup(metadata, outputs, expected, prompts, set(by_id))
        if {key(r): r for r in alignment} != {key(r): r for r in saved_alignment}:
            raise ValueError('Rebuilt V4 alignment differs from saved full training receipt')
        if len(alignment) != 17 * len(rows):
            raise ValueError('Incomplete V4 all-node training alignment')
        return lookup, {'mode': 'actual question-excluded V4 OOF outputs', 'alignment_sha256': fingerprint(alignment),
                        'aligned_nodes': len(alignment), 'neural_question_exclusion': True,
                        'second_stage_crossfit': 'Separate grouped forecaster crossfit on the fixed OOF feature table',
                        'nested_crossfit': False, 'neural_weights_recomputed': False}

    def lookup(self, observer, rows, receipt):
        cache_key = (id(observer), tuple(r['id'] for r in rows))
        if cache_key in self.cache:
            return self.cache[cache_key]
        if getattr(observer, 'components', None):
            components = [self.lookup(c, rows, {}) for c in observer.components]
            if any(set(v[0]) != set(components[0][0]) for v in components):
                raise ValueError('Fusion training components have different node identities')
            lookup = {k: (*[np.concatenate([part[0][k][i] for part in components]) for i in range(3)], 0)
                      for k in components[0][0]}
            result = lookup, {'mode': 'concatenated physical observer components', 'components': [c[1] for c in components]}
        elif 'neural_observer_oof_provenance' in receipt:
            result = self.v4_lookup(observer, rows, receipt)
        else:
            profile = observer._recipe.get('input_profile')
            if profile not in ('full-options-openstax-v1', 'supergpqa-option-correctness-v1'):
                raise ValueError('Unsupported observer input profile')
            result = self.ordinary_lookup(observer, rows, profile == 'supergpqa-option-correctness-v1')
        self.cache[cache_key] = result
        return result

    def reconstruct(self, model, name='A'):
        if id(model) in self.seen:
            return {'reuse_component': self.seen[id(model)]}
        self.seen[id(model)] = name
        if hasattr(model, 'models'):
            if type(model).__name__ != 'SubjectPlanner' or set(model.models) != {'math', 'chemistry', 'physics'}:
                raise ValueError('Unsupported recursive A wrapper')
            return {'role': 'subject-routed POMDP', 'settings_sha256': fingerprint(model.settings),
                    'children': {s: self.reconstruct(child, name + '/' + s) for s, child in sorted(model.models.items())}}
        if not isinstance(model, core.Planner):
            raise ValueError('Only fitted A8 POMDP artifacts can be reconstructed')
        receipt = model.fit_receipt
        rows = self.rows if receipt['paired_questions'] == 512 else [r for r in self.rows if r['subject'] in ('chemistry', 'physics')]
        if len(rows) != receipt['paired_questions'] or receipt['paired_questions'] not in (256, 512):
            raise ValueError('Unexpected A training population')
        recipe = leaf_recipe(model.classifier, self.inputs)
        if 'component_alignment_sha256' in receipt:
            alignment = self.inputs.json(self.args.fusion / 'component-alignment.json', receipt['component_alignment_sha256'])
            for component, directory in (('old', self.args.base), ('option', self.args.science)):
                self.inputs.sha(directory / 'training-neural.jsonl', alignment[component + '_metadata_sha256'])
                self.inputs.sha(directory / 'training-neural.npz', alignment[component + '_neural_sha256'])
        lookup, provenance = self.lookup(model.classifier, rows, receipt)
        if receipt.get('fit_module') == 'nonlinear_repairability':
            from nonlinear_repairability import fit
            kwargs = {'smoothing': receipt['smoothing'], 'max_leaf_nodes': receipt['forecaster_hyperparameters']['max_leaf_nodes']}
        else:
            fit = factorized.fit if isinstance(model.forecaster, factorized.Forecaster) else core.fit
            kwargs = {'smoothing': receipt['smoothing'], 'regularization': receipt['calibration_C']}
            if fit is factorized.fit:
                kwargs['prior_smoothing'] = receipt['prior_smoothing']
        with threadpool_limits(limits=1):
            rebuilt = fit(rows, self.trees, model.classifier, lookup, **kwargs)
        comparison = compare_learned(model, rebuilt, self.args.atol, self.args.rtol)
        record = {'component': name, 'fit_function': fit.__module__ + '.fit', 'training_questions': len(rows),
                  'training_ids_sha256': fingerprint([r['id'] for r in rows]), 'hyperparameters': kwargs,
                  'conservative_encoder_cost_recipe': {'parameters': recipe['parameters'], 'max_length': recipe['max_length']},
                  'feature_provenance': provenance, 'numeric_comparison': comparison}
        self.components.append(record)
        print(json.dumps({'component': name, 'passed': True, 'max_absolute_error': comparison['max_absolute_error']}), flush=True)
        return {'component': name}


def check_identity(identity, inputs, args):
    mappings = {'pomdp_sha256': 'repairability_pomdp.py', 'core_sha256': 'repairability_pomdp.py',
                'factorized_sha256': 'factorized_repairability.py', 'policy_sha256': 'continuous_repairability.py',
                'nonlinear_sha256': 'nonlinear_repairability.py',
                'fusion_sha256': 'science_fusion_pomdp.py', 'numeric_consistency_sha256': 'numeric_consistency.py'}
    for key, filename in mappings.items():
        if key in identity:
            inputs.sha(HERE / filename, identity[key])
    sources = identity.get('source_sha256')
    if isinstance(sources, dict):
        for filename, digest in sources.items():
            inputs.sha(HERE / filename, digest)
    elif isinstance(sources, str):
        matches = [p for p in HERE.glob('development_*.py') if hashlib.sha256(p.read_bytes()).hexdigest() == sources]
        if not matches:
            raise ValueError('Development driver no longer matches its recorded source identity')
        inputs.sha(matches[0], sources)
    for key in ('new_train_questions_sha256', 'train_question_sha256', 'train_questions_sha256'):
        if key in identity:
            inputs.sha(args.data / 'splits/train-questions.jsonl', identity[key])
    for key, name in (('base_identity_sha256', 'identity.json'), ('training_neural_sha256', 'training-neural.npz'),
                       ('training_neural_metadata_sha256', 'training-neural.jsonl')):
        if key in identity:
            inputs.sha(args.base / name, identity[key])


def run(args):
    if args.output.exists():
        raise ValueError('Use a new audit output path')
    inputs = Inputs()
    rows, trees, prefix = training_data(args.data, inputs)
    model, selection, identity = resolve_selection(args.summary, args.registry, inputs)
    check_identity(identity, inputs, args)
    base_identity = inputs.json(args.base / 'identity.json')
    check_identity(base_identity, inputs, args)
    audit = RefitAudit(args, inputs, rows, trees)
    with forbid_neural_calls():
        layout = audit.reconstruct(model)
    source_files = {str(path): inputs.sha(path) for path in project_sources()}
    result = {'schema': 'a-training-refit-audit-v1', 'passed': True, 'training_only': True,
              'no_test_question_response_values_read': True,
              'no_development_question_response_values_read': True, 'no_neural_inference': True,
              'historical_identity_claim': False, 'nested_crossfit_claim': False,
              'scope': 'Current fitted second-stage A state matches deterministic refitting from recorded training inputs',
              'limitations': ['Does not prove historical pickle byte identity or reconstruct neural training.',
                             'Does not validate development selection scores or provide fresh-test evidence.',
                             'V4 neural question exclusion and second-stage forecaster crossfit are separate, not nested CV.'],
              'excluded_state': ['Neural weights/tokenizer tensors', 'Runtime caches and object identity',
                                 'Development-selected policy settings (bound by digest, not refitted)'],
              'selection': selection, 'layout': layout, 'components': audit.components,
              'sealed_training_response_prefix': prefix, 'input_files': inputs.files, 'source_files': source_files,
              'runtime': {'python': sys.version, 'platform': platform.platform(), 'cpu_thread_limit': 1,
                          'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn', 'scipy', 'torch')},
                          'environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': True, 'output': str(args.output.resolve()), 'components': len(audit.components)}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'summary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('registry', 'base', 'science', 'refined', 'fusion'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--rtol', type=float, default=1e-10)
    args = parser.parse_args()
    args.registry = args.registry or args.data.parent
    args.base = args.base or args.registry / 'development-repairability-v1'
    args.science = args.science or args.registry / 'development-science-option-v1'
    args.refined = args.refined or args.registry / 'development-refined-observer-v1'
    args.fusion = args.fusion or args.registry / 'development-science-fusion-v1'
    if min(args.atol, args.rtol) < 0 or max(args.atol, args.rtol) > 1e-8:
        parser.error('Numerical tolerances must be between zero and 1e-8')
    run(args)
