"""Evaluate continuous repairability beliefs on the registered development set."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from development_repairability import wait_plain
from study_common import statistics
import continuous_repairability as policy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.base / 'summary.json').read_bytes())
    rows, trees, _, _ = wait_plain(args.data, 'dev')
    models = sorted(args.base.glob('C*-smooth*.pkl'))
    identity = {'source_sha256': sha(__file__), 'policy_sha256': sha(policy.__file__),
                'base_identity_sha256': sha(args.base / 'identity.json'),
                'model_sha256': {p.name: sha(p) for p in models},
                'weights': [0., .5, 1.], 'test_used': False,
                'development_question_sha256': sha(args.data / 'splits/dev-questions.jsonl')}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    candidates, shared = [], None
    for path in models:
        model = pickle.loads(path.read_bytes())
        if shared is None:
            model.classifier.device = args.device
            model.classifier.warm(rows, trees, {})
            shared = model.classifier
        else:
            for attr in ('_model', '_tokenizer', '_recipe', '_cache', 'device', '_retriever', '_backgrounds'):
                setattr(model.classifier, attr, getattr(shared, attr))
        by_subject = {s: [] for s in ('math', 'chemistry', 'physics')}
        for weight in (0., .5, 1.):
            for penalty in (0., .01, .02, .04, .06, .08, .10, .16, .24):
                settings = {'penalty': penalty, 'posterior_weight': weight}
                values = [policy.evaluate(r, trees[r['id']], {}, model, settings, 'A') for r in rows]
                candidates.append({'model': path.stem, 'settings': settings, 'dev': statistics(values)})
                for s in by_subject:
                    by_subject[s].append({'settings': settings, 'dev': statistics([v for r, v in zip(rows, values) if r['subject'] == s])})
        settings = {'by_subject': {s: max(v, key=lambda x: (x['dev']['accuracy'], -x['dev']['mean_compute']))['settings'] for s, v in by_subject.items()}}
        values = [policy.evaluate(r, trees[r['id']], {}, model, settings, 'A') for r in rows]
        candidates.append({'model': path.stem, 'settings': settings, 'dev': statistics(values)})
        (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
        print('Completed continuous belief candidate', path.stem, json.dumps(candidates[-1]), flush=True)
    best = max(candidates, key=lambda x: (x['dev']['accuracy'], -x['dev']['mean_compute']))
    highest = max(v['accuracy'] for v in source['baselines'].values())
    result = {'selected_A': best, 'baselines': source['baselines'], 'test_used': False,
              'development_questions': len(rows), 'A_above_highest_baseline': best['dev']['accuracy'] - highest,
              'approximation': 'Discriminative history posterior blend; future planning uses finite POMDP observation distribution.'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'base', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    run(p.parse_args())
