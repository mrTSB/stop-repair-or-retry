"""Report nonlinear development diagnostics before every A reference finishes.

This uses the immutable trained models and exactly the registered development
grid. It reads cached paid observations only, never runs a neural model, and
never writes selected_A or a final selection summary. The registered final
development driver and registry-wide strongest-A freeze remain authoritative.
"""

import argparse
import json
from pathlib import Path
import pickle
import time

from threadpoolctl import threadpool_limits

import development_nonlinear as d
import prospective_freeze as freeze
from prospective_freeze import nonlinear_fitting_evidence


def report(candidates, references, pending, baselines):
    key = lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute'])
    nonlinear = max(candidates, key=key)
    best_completed = max([nonlinear, *references], key=key)
    return {'stage': 'development diagnostic only; final A selection remains open',
        'nonlinear_A': nonlinear, 'best_completed_A_preview': best_completed,
        'completed_A_references': references, 'pending_A_references': pending,
        'baselines': baselines, 'test_used': False, 'final_selection_performed': False,
        'nonlinear_above_highest_baseline': nonlinear['dev']['accuracy'] - max(v['accuracy'] for v in baselines.values()),
        'nonlinear_above_best_completed_reference': (nonlinear['dev']['accuracy'] - max(
            r['dev']['accuracy'] for r in references)) if references else None,
        'fitting_questions': 512, 'development_questions': 256,
        'neural_inference': False, 'LLM_calls': 0,
        'selection_note': 'Run the immutable registered final development driver after pending references finish; prospective freeze independently retains the strongest completed A across the full registry.'}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    identity = d.read(args.models / 'identity.json')
    sources = {str(Path(path).resolve()): d.sha(path) for path in
        (__file__, d.__file__, d.nonlinear.__file__, d.nonlinear.core.__file__, d.policy.__file__, freeze.__file__)}
    write = lambda name, value: d.write(args.output / name, value)
    write('registration.json', {'source_files': sources,
        'fitting_identity_sha256': d.sha(args.models / 'identity.json'),
        'fitting_complete_sha256': d.sha(args.models / 'fit-complete.json'),
        'models_directory': str(args.models.resolve()), 'features_directory': str(args.features.resolve()),
        'registered_reference_summaries': identity['reference_summaries'],
        'test_used': False, 'final_selection_performed': False})
    while not (args.features / 'complete.json').exists():
        if not args.wait:
            raise ValueError('Complete the development feature cache or enable --wait')
        print('Waiting for complete paid development feature cache', flush=True)
        time.sleep(15)
    if any(d.sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Preview runtime changed while waiting')
    for field, path in (('source_sha256', d.__file__), ('nonlinear_sha256', d.nonlinear.__file__),
                        ('core_sha256', d.nonlinear.core.__file__), ('policy_sha256', d.policy.__file__)):
        if identity[field] != d.sha(path):
            raise ValueError('Immutable nonlinear runtime differs from fitting registration')
    nonlinear_fitting_evidence(args.models, {'fit_module': 'nonlinear_repairability'}, identity,
        args.data, args.models / next(iter(identity['model_sha256'])),
        {'development_feature_directory': str(args.features),
         'development_feature_complete_sha256': d.sha(args.features / 'complete.json')})
    rows, trees, _, _ = d.wait_plain(args.data, 'dev')
    if len(rows) != 256 or d.sha(args.data / 'splits/dev-questions.jsonl') != identity['dev_questions_sha256']:
        raise ValueError('Preserve the whole registered development panel')
    original = d.observer_from(args.base)
    metadata = [json.loads(line) for line in (args.features / 'metadata.jsonl').read_text().splitlines()]
    lookup, checked = d.verify_features(rows, trees, original, metadata, d.np.load(args.features / 'neural.npz'), True)
    observer = d.CachedObserver(original.directory, original.temperature, calibration=original.calibration)
    observer._recipe, observer._retriever, observer._backgrounds = original._recipe, original._retriever, original._backgrounds
    observer.saved = lookup
    observer.input_sha256 = {(r['id'], r['path']): r['input_sha256'] for r in checked}
    baselines = {name: d.statistics([d.fixed(r, trees[r['id']], *settings) for r in rows])
                 for name, settings in d.BASELINES.items()}
    candidates = []
    for filename, digest in identity['model_sha256'].items():
        path = args.models / filename
        if d.sha(path) != digest:
            raise ValueError('Immutable nonlinear fitted model changed')
        model = pickle.loads(path.read_bytes()); model.classifier = observer
        by_subject = {s: [] for s in ('math', 'chemistry', 'physics')}
        with threadpool_limits(limits=1):
            for weight in d.WEIGHTS:
                for penalty in d.PENALTIES:
                    settings = {'penalty': penalty, 'posterior_weight': weight}
                    values = [d.policy.evaluate(r, trees[r['id']], {}, model, settings, 'A') for r in rows]
                    candidates.append({'model': path.stem, 'model_path': str(path.resolve()),
                        'policy_module': 'continuous_repairability', 'fit_module': 'nonlinear_repairability',
                        'settings': settings, 'dev': d.statistics(values)})
                    for subject in by_subject:
                        by_subject[subject].append({'settings': settings,
                            'dev': d.statistics([v for r, v in zip(rows, values) if r['subject'] == subject])})
            settings = {'by_subject': {s: max(group, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))['settings']
                                      for s, group in by_subject.items()}}
            values = [d.policy.evaluate(r, trees[r['id']], {}, model, settings, 'A') for r in rows]
        candidates.append({'model': path.stem, 'model_path': str(path.resolve()),
            'policy_module': 'continuous_repairability', 'fit_module': 'nonlinear_repairability',
            'settings': settings, 'dev': d.statistics(values)})
        write('search.json', candidates)
        print('Previewed nonlinear development model', path.stem, flush=True)
    references, pending = [], []
    for name in identity['reference_summaries']:
        path = Path(name)
        if not path.exists():
            pending.append(name)
        else:
            references.append(d.reference_candidate(path, identity['dev_questions_sha256']))
    result = report(candidates, references, pending, baselines)
    result['development_feature_complete_sha256'] = d.sha(args.features / 'complete.json')
    result['registration_sha256'] = d.sha(args.output / 'registration.json')
    if any(d.sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Preview runtime changed during development scoring')
    write('preview.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'base', 'models', 'features', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    run(parser.parse_args())
