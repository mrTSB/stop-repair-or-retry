"""Build review statistics and development-only figures from completed artifacts.

This is postprocessing only; it does not read the runtime or choose a new model.
Requires matplotlib solely for rendering. Run: python analyze_accuracy.py RESULT.
"""
import argparse
import json
import math
from pathlib import Path


def analyze(folder):
    summary = json.loads((folder / 'summary.json').read_text())
    candidates = json.loads((folder / 'candidate-metrics.json').read_text())
    changes = json.loads((folder / 'changed-questions.json').read_text())
    # Independently compose the minimum cost for each integer accuracy count.
    dp = {0: 0.0}
    subjects = {}
    for subject, rows in candidates.items():
        best = {}
        for row in rows:
            metric = row['metrics']
            correct, cost = metric['correct'], metric['mean_compute'] * metric['n']
            best[correct] = min(cost, best.get(correct, math.inf))
        subjects[subject] = {'candidate_count': len(rows), 'max_correct': max(best),
                             'n': rows[0]['metrics']['n']}
        combined = {}
        for n, cost in dp.items():
            for m, extra in best.items():
                combined[n + m] = min(cost + extra, combined.get(n + m, math.inf))
        dp = combined
    frontier, cheapest = [], math.inf
    for correct in sorted(dp, reverse=True):
        cost = dp[correct] / 256
        if cost < cheapest:
            frontier.append({'correct': correct, 'accuracy': correct / 256, 'mean_compute': cost})
            cheapest = cost
    tuned, baseline, b = (summary[k] for k in ('tuned_C', 'baseline_C', 'B'))
    assert max(dp) == tuned['correct']
    assert math.isclose(dp[max(dp)] / 256, tuned['mean_compute'], rel_tol=1e-12)
    folds = summary['selection_sensitivity_four_dev_folds']
    review = {
        'passed': True,
        'subject_results': subjects,
        'accuracy_gain_correct_vs_original_C': tuned['correct'] - baseline['correct'],
        'accuracy_gain_correct_vs_prior_tuned_C': tuned['correct'] - 178,
        'accuracy_difference_correct_vs_B': tuned['correct'] - b['correct'],
        'compute_reduction_percent_vs_original_C': 100 * (1 - tuned['mean_compute'] / baseline['mean_compute']),
        'compute_reduction_percent_vs_B': 100 * (1 - tuned['mean_compute'] / b['mean_compute']),
        'wins_vs_original_C': sum(c['new_correct'] > c['old_correct'] for c in changes),
        'losses_vs_original_C': sum(c['new_correct'] < c['old_correct'] for c in changes),
        'changed_paths_or_bills': len(changes),
        'fold_selected_correct': sum(f['tuned']['correct'] for f in folds),
        'fold_reference_correct': sum(f['baseline']['correct'] for f in folds),
        'fold_selected_mean_compute': math.fsum(f['tuned']['mean_compute'] for f in folds) / 4,
        'fresh_validation': False,
        'test_used': False,
        'pareto_frontier': sorted(frontier, key=lambda r: r['mean_compute']),
    }
    (folder / 'analysis.json').write_text(json.dumps(review, indent=2))
    return summary, review


def figures(folder, summary, review):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    points = review['pareto_frontier']
    ax.plot([p['mean_compute'] for p in points], [100 * p['accuracy'] for p in points],
            color='#68798a', linewidth=1.5, label='Search frontier (reused DEV)')
    for label, row, color, offset in [
        ('Original C', summary['baseline_C'], '#777777', (7, -17)),
        ('Prior tuned C', {'accuracy': 178 / 256, 'mean_compute': 3330.645424500263}, '#b26b00', (-100, -18)),
        ('Accuracy-first C', summary['tuned_C'], '#007f78', (-112, 12)),
        ('Frozen B', summary['B'], '#6f42a1', (8, 8)),
    ]:
        ax.scatter(row['mean_compute'], 100 * row['accuracy'], color=color, s=52, zorder=4)
        ax.annotate(label, (row['mean_compute'], 100 * row['accuracy']), xytext=offset,
                    textcoords='offset points', color=color)
    ax.set(xlabel='Mean compute proxy (parameter-weighted tokens)', ylabel='Development accuracy (%)',
           title='C hyperparameter search: development accuracy / compute')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=.18)
    fig.suptitle('Exploratory selection; not reserved-test performance', fontsize=10, color='#555555')
    fig.savefig(folder / 'accuracy-compute.png', dpi=180)
    fig.savefig(folder / 'accuracy-compute.pdf')
    plt.close(fig)
    folds = summary['selection_sensitivity_four_dev_folds']
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    positions = list(range(len(folds)))
    ax.bar([p - .18 for p in positions], [f['baseline']['correct'] for f in folds], .36, label='Original C', color='#999999')
    ax.bar([p + .18 for p in positions], [f['tuned']['correct'] for f in folds], .36, label='Selected on other folds', color='#007f78')
    ax.set(xticks=positions, xticklabels=[str(i + 1) for i in positions], ylim=(0, 64),
           xlabel='Held development fold', ylabel='Correct out of 64',
           title='Selection sensitivity on the already reused development panel')
    ax.legend(frameon=False)
    fig.suptitle('Historical development reuse prevents independent validation', fontsize=10, color='#555555')
    fig.savefig(folder / 'fold-sensitivity.png', dpi=180)
    fig.savefig(folder / 'fold-sensitivity.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    summary, review = analyze(args.result)
    figures(args.result, summary, review)
    print(json.dumps({k: v for k, v in review.items() if k != 'pareto_frontier'}, indent=2))
