"""CPU-only word-POMDP fitting and separate cached development comparison."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits

import training_sparse_text as diagnostic
import sparse_text_repairability as text
import word_repairability_pomdp as policy
import development_nonlinear as cached
from development_refined_observer import reference_candidate
from prepare_mixed_mentor import plain_data
from study import fixed
from study_common import BASELINES, statistics

sha, read, write = diagnostic.sha, diagnostic.read, diagnostic.write
PENALTIES, WEIGHTS, SMOOTHING = cached.PENALTIES, cached.WEIGHTS, cached.SMOOTHING


def verify_evidence(mapping):
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise ValueError('Registered source or input changed: ' + str(path))


def training_inputs(args):
    data, alignment, provenance = diagnostic.load_inputs(args)
    registration = read(args.diagnostic / 'registration.json')
    summary = read(args.diagnostic / 'summary.json')
    for file, field in (('registration.json', 'registration_sha256'), ('private-oof.npz', 'predictions_sha256'),
                        ('fold-receipts.json', 'fold_receipts_sha256'), ('training-alignment.json', 'alignment_sha256')):
        if sha(args.diagnostic / file) != summary[field]:
            raise ValueError('Completed training diagnostic changed')
    verify_evidence(registration['source_files']); verify_evidence(registration['input_evidence'])
    for key, value in provenance.items():
        if registration[key] != value:
            raise ValueError('Word runtime uses different training inputs: ' + key)
    if registration['temperature'] != 1. or registration['case_preserved'] is not True or summary['test_used']:
        raise ValueError('Preserve fixed case-sensitive T1 grouped diagnostic')
    rows, trees = plain_data(args.data, 'train')
    data['answer_present'] = np.asarray([bool(trees[qid][path]['answer']) for qid, path in zip(data['ids'], data['paths'])])
    receipts = read(args.diagnostic / 'fold-receipts.json')['word12_C1']
    ids = set(data['ids'])
    for fold, receipt in enumerate(receipts):
        held = sorted(q for q in ids if text.fold_for(q) == fold)
        fit = sorted(ids - set(held))
        if (receipt['fold'] != fold or receipt['question_overlap'] != 0
                or receipt['held_ids_sha256'] != text.fingerprint(held)
                or receipt['fit_ids_sha256'] != text.fingerprint(fit)):
            raise ValueError('Changed whole-question excluded emission forecasts')
    if len(receipts) != 3:
        raise ValueError('Need all three excluded-question fold receipts')
    with np.load(args.diagnostic / 'private-oof.npz', allow_pickle=False) as archive:
        oof = archive['word12_C1'].copy()
    evidence = {**registration['input_evidence'], **{str((args.diagnostic / name).resolve()): sha(args.diagnostic / name)
        for name in ('registration.json', 'summary.json', 'private-oof.npz', 'fold-receipts.json', 'training-alignment.json')}}
    for smoothing in SMOOTHING:
        path = args.base / f'C1-smooth{smoothing:g}.pkl'; evidence[str(path.resolve())] = sha(path)
    return data, alignment, oof, provenance, evidence


def fit(args):
    if args.output.exists():
        raise ValueError('Use a new word-POMDP area; do not overwrite a registered run')
    with diagnostic.no_model_calls(), threadpool_limits(limits=1):
        data, alignment, oof, provenance, evidence = training_inputs(args)
        sources = diagnostic.source_files()
        registration = {'source_files': sources, 'input_evidence': evidence,
            'training_diagnostic_sha256': sha(args.diagnostic / 'summary.json'),
            'base_directory': str(args.base.resolve()), 'fitting_questions': 512, 'decision_nodes': 1536,
            'train_questions_sha256': provenance['train_question_sha256'],
            'dev_questions_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
            'encoder_recipe_sha256': provenance['encoder_recipe_sha256'],
            'variant': 'word12_C1', 'smoothing': list(SMOOTHING), 'temperature': 1.,
            'posterior_weights': list(WEIGHTS), 'penalties': list(PENALTIES),
            'policy_module': 'word_repairability_pomdp', 'test_used': False,
            'development_values_read_during_fit': False, 'neural_calls': 0, 'cloud_calls': 0,
            'reference_summaries': [str(p.resolve()) for p in args.reference],
            'global_A_selection': 'Pending other registered A candidates; preserve every supplied completed reference'}
        args.output.mkdir(parents=True)
        write(args.output / 'fitting-registration.json', registration)
        write(args.output / 'training-alignment.json', alignment)
        print('Registered word POMDP before full-training fit', sha(args.output / 'fitting-registration.json'), flush=True)
        forecaster = text.fit_forecaster(data['texts'], data['numeric'], data['labels'], 'word12_C1')
        if forecaster.convergence_warnings:
            raise ValueError('Word full-fit did not converge: ' + repr(forecaster.convergence_warnings))
        models = []
        for smoothing in SMOOTHING:
            source = args.base / f'C1-smooth{smoothing:g}.pkl'
            parent = pickle.loads(source.read_bytes())
            model = policy.build(parent, forecaster, data, oof, smoothing,
                {'parent_model_sha256': sha(source), 'registration_sha256': sha(args.output / 'fitting-registration.json'),
                 'training_diagnostic_sha256': sha(args.diagnostic / 'summary.json')})
            key = f'word12-C1-smooth{smoothing:g}'; path = args.output / (key + '.pkl')
            path.write_bytes(pickle.dumps(model, protocol=5))
            models.append({'key': key, 'sha256': sha(path), 'smoothing': smoothing, 'parent_sha256': sha(source)})
        verify_evidence(evidence); verify_evidence(sources)
        identity = {**registration, 'model_sha256': {m['key'] + '.pkl': m['sha256'] for m in models},
            'training_alignment_sha256': sha(args.output / 'training-alignment.json'),
            'fitting_registration_sha256': sha(args.output / 'fitting-registration.json')}
        write(args.output / 'identity.json', identity)
        write(args.output / 'complete.json', {'models': models, 'identity_sha256': sha(args.output / 'identity.json'),
            'training_only': True, 'neural_calls': 0, 'cloud_calls': 0, 'test_used': False})
        print('Completed word-POMDP CPU fit', flush=True)


def development_cache(args, identity):
    receipt = read(args.features / 'complete.json')
    panel_sha = sha(args.data / 'splits/dev-questions.jsonl')
    if (identity['dev_questions_sha256'] != panel_sha or receipt['dev_questions_sha256'] != panel_sha or receipt['encoder_recipe_sha256'] != identity['encoder_recipe_sha256']
            or receipt['neural_sha256'] != sha(args.features / 'neural.npz')
            or receipt['metadata_sha256'] != sha(args.features / 'metadata.jsonl')):
        raise ValueError('Development feature cache does not match the exact registered panel/observer')
    rows, trees = plain_data(args.data, 'dev')
    if len(rows) != 256:
        raise ValueError('Use all256 registered development questions')
    original = cached.observer_from(args.base)
    metadata = [json.loads(line) for line in (args.features / 'metadata.jsonl').read_text().splitlines()]
    with np.load(args.features / 'neural.npz', allow_pickle=False) as values:
        lookup, alignment = cached.verify_features(rows, trees, original, metadata, values, True)
    observer = cached.CachedObserver(original.directory, original.temperature, calibration=original.calibration)
    observer._recipe, observer._retriever, observer._backgrounds = original._recipe, original._retriever, original._backgrounds
    observer.saved = lookup
    observer.input_sha256 = {(r['id'], r['path']): r['input_sha256'] for r in alignment}
    return rows, trees, observer, alignment, panel_sha


def develop(args):
    identity = read(args.output / 'identity.json'); complete = read(args.output / 'complete.json')
    if complete['identity_sha256'] != sha(args.output / 'identity.json'):
        raise ValueError('Changed completed word fit')
    verify_evidence(identity['source_files']); verify_evidence(identity['input_evidence'])
    if (args.output / 'development-registration.json').exists() or args.features is None:
        raise ValueError('Require a fresh development registration and an existing complete feature cache')
    with diagnostic.no_model_calls(), threadpool_limits(limits=1):
        rows, trees, observer, alignment, panel_sha = development_cache(args, identity)
        references = [reference_candidate(Path(p), panel_sha) for p in identity['reference_summaries']]
        if not references:
            raise ValueError('Retain the strongest completed A reference')
        sources = diagnostic.source_files()
        evidence = {str((args.features / name).resolve()): sha(args.features / name)
                    for name in ('complete.json', 'metadata.jsonl', 'neural.npz')}
        evidence.update({str(Path(p).resolve()): sha(p) for p in identity['reference_summaries']})
        registration = {'dev_questions_sha256': panel_sha, 'development_questions': 256, 'source_files': sources,
            'input_evidence': evidence, 'fit_complete_sha256': sha(args.output / 'complete.json'),
            'penalties': list(PENALTIES), 'posterior_weights': list(WEIGHTS), 'test_used': False,
            'neural_calls': 0, 'cloud_calls': 0, 'runtime_policy_module': 'word_repairability_pomdp'}
        write(args.output / 'development-registration.json', registration)
        write(args.output / 'development-alignment.json', alignment)
        baselines = {name: statistics([fixed(r, trees[r['id']], *s) for r in rows]) for name, s in BASELINES.items()}
        candidates = []
        for record in complete['models']:
            path = args.output / (record['key'] + '.pkl')
            if sha(path) != record['sha256']:
                raise ValueError('Changed saved word model')
            model = pickle.loads(path.read_bytes()); model.classifier = observer
            groups = {s: [] for s in ('math', 'chemistry', 'physics')}
            def result(settings):
                values = [policy.evaluate(r, trees[r['id']], {}, model, settings, 'A') for r in rows]
                return values, {'model': path.stem, 'model_path': str(path.resolve()), 'policy_module': 'word_repairability_pomdp',
                    'settings': settings, 'dev': statistics(values),
                    'mean_sparse_head_calls': float(np.mean([v['sparse_head_calls'] for v in values])),
                    'mean_sparse_head_cpu_ms': float(1000 * np.mean([v['sparse_head_cpu_seconds'] for v in values]))}
            for weight in WEIGHTS:
                for penalty in PENALTIES:
                    settings = {'penalty': penalty, 'posterior_weight': weight}
                    values, candidate = result(settings); candidates.append(candidate)
                    for subject in groups:
                        groups[subject].append({'settings': settings, 'dev': statistics([v for r, v in zip(rows, values) if r['subject'] == subject])})
            settings = {'by_subject': {s: max(group, key=lambda x: (x['dev']['accuracy'], -x['dev']['mean_compute']))['settings']
                                       for s, group in groups.items()}}
            _, candidate = result(settings); candidates.append(candidate)
            write(args.output / 'search.json', candidates)
            print('Completed full256 development search for', path.stem, flush=True)
        best = max(candidates, key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
        retained = max([best, *references], key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
        verify_evidence(evidence); verify_evidence(sources)
        write(args.output / 'summary.json', {'selected_A': retained, 'word_A': best, 'existing_A_references': references,
            'baselines': baselines, 'development_questions': 256, 'fitting_questions': 512, 'test_used': False,
            'dev_questions_sha256': panel_sha, 'development_registration_sha256': sha(args.output / 'development-registration.json'),
            'global_A_selection_pending': True, 'A_above_highest_baseline': retained['dev']['accuracy'] - max(x['accuracy'] for x in baselines.values()),
            'no_new_neural_or_cloud_calls': True,
            'cost_accounting': 'Same neural/solver parameter-token proxy; measured added sparse-head CPU time reported separately.'})
        print('Completed word-POMDP development comparison', json.dumps({'word_A': best, 'retained_A': retained}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'develop'))
    for name in ('data', 'base', 'diagnostic', 'legacy-examples', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--reference', type=Path, action='append', default=[])
    args = parser.parse_args()
    {'fit': fit, 'develop': develop}[args.phase](args)
