"""Training-only grouped calibration diagnostic, never a fresh accuracy result."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks  # Initialize the archived module search paths.
from semantic_data import read
from numeric_consistency import observations


def run(args):
    samples = read(args.examples)
    outputs = np.load(args.verifier / 'oof.npz')
    temperature = json.loads((args.verifier / 'training-summary.json').read_bytes())['temperature_selected_on_training_oof']
    rows = {r['id']: r for r in read(args.questions)}
    selected = [i for i, r in enumerate(samples) if r['kind'] == 'solver' and r['train'] and r['valid']]
    features = {'original': [], 'arithmetic': [], 'numeric_arithmetic': []}
    labels, folds = [], []
    for i in selected:
        r = samples[i]
        logits = outputs['state'][i].astype(float) / temperature
        logits -= logits.mean()
        base = [*logits, float(bool(r['path'])),
                *[float(rows[r['id']]['subject'] == s) for s in ('math', 'chemistry', 'physics', 'biology')]]
        arith = observations(r['right'].partition('\nSolution:\n')[2])
        features['original'].append(base)
        features['arithmetic'].append([*base, *arith])
        features['numeric_arithmetic'].append([*base, *r['numeric'], *arith])
        labels.append(r['label'])
        folds.append(r['fold'])
    y, folds = np.asarray(labels), np.asarray(folds)
    report = {'cohort': 'Training-only grouped calibration diagnostic; not a full-pipeline held-out estimate.', 'nodes': len(y), 'models': {}}
    for name, values in features.items():
        x = np.asarray(values)
        probability = np.zeros((len(y), 4))
        for fold in (0, 1, 2):
            fit, held = folds != fold, folds == fold
            scaler = StandardScaler().fit(x[fit])
            model = LogisticRegression(C=1., max_iter=1000).fit(scaler.transform(x[fit]), y[fit])
            probability[np.ix_(held, model.classes_.astype(int))] = model.predict_proba(scaler.transform(x[held]))
        pc = probability[:, 1] + probability[:, 3]
        report['models'][name] = {'binary_accuracy': float(np.mean((pc >= .5) == y % 2)),
                                  'brier': float(np.mean((pc - y % 2) ** 2)),
                                  'auc': float(roc_auc_score(y % 2, pc))}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('examples', 'verifier', 'questions', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
