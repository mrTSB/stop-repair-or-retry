"""Publish an audited, text-free confirmation report with an explicit allowlist."""

import argparse
import json
from pathlib import Path
import shutil

from confirmation_common import BASELINES, evidence, sha, verify_fixed_prefix

TOP = {'policy', 'panel', 'id', 'domain', 'subject', 'correct', 'path', 'solver_tokens',
       'judge_tokens', 'compute', 'total_parameter_compute', 'solver_calls', 'judge_calls',
       'decisions', 'verifier_tokens', 'verifier_calls', 'verifier_token_slots',
       'verifier_parameters_billions', 'judge_physical_calls',
       'verifier_total_parameters_billions', 'verifier_components'}
DECISION = {'path', 'belief', 'action', 'queried', 'information_value', 'continuous_filter',
            'contextual_transition', 'root_medium_review', 'tracked_mentor'}
LABELS = {'A': 'Controller A (POMDP)', 'B': 'Controller B (LLM)', 'C': 'Controller C (hybrid)'}
METHODS = [*BASELINES, 'A', 'B', 'C']
PATHS = {'', 'repair', 'retry', 'guided', 'repair/repair', 'repair/retry',
         'retry/repair', 'retry/retry', 'guided/repair', 'guided/retry',
         'repair/reviewed', 'retry/reviewed', 'guided/reviewed', 'guided/reuse', 'retry/reuse'}


def validate_outcome(item, ids):
    if set(item) - TOP or item['id'] not in ids or item['policy'] not in METHODS:
        raise ValueError('Unexpected or potentially text-bearing outcome field')
    if item['domain'] not in ('math', 'science') or item['subject'] not in ('math', 'chemistry', 'physics'):
        raise ValueError('Unexpected task label')
    if item['path'] not in PATHS or item['correct'] not in (0, 1):
        raise ValueError('Unexpected path or correctness')
    for d in item.get('decisions', []):
        if set(d) - DECISION or d['path'] not in PATHS or d['action'] not in ('stop', 'repair', 'retry'):
            raise ValueError('Unexpected decision field')
        if 'belief' in d and (len(d['belief']) not in (2, 4, 8) or not all(isinstance(x, (int, float)) for x in d['belief'])):
            raise ValueError('Unexpected belief encoding')
    # Fixed policy and source identifiers are the only unstructured top-level
    # strings; all model text, answer strings, options and prompts are absent.
    strings = {'policy', 'panel', 'id', 'domain', 'subject', 'path'}
    for key, value in item.items():
        if key not in strings and key != 'decisions' and not isinstance(value, (int, float)):
            raise ValueError('Unexpected nonnumeric result value')


def table(report):
    lines = ['| Method | Correct | Accuracy (%) | Avg. LLM tokens | Compute proxy |',
             '|---|---:|---:|---:|---:|']
    n = report['questions']
    for name in METHODS:
        value = report['methods'][name]
        tokens = value['mean_solver_tokens'] + value['mean_judge_tokens']
        lines.append(f'| {LABELS.get(name, name)} | {value["correct"]}/{n} | '
                     f'{100*value["accuracy"]:.2f} | {tokens:,.1f} | {value["mean_compute"]:,.1f} |')
    return '\n'.join(lines)


def report_text(summary, frozen):
    joint = summary['joint']
    target = 'met' if joint['all_required_passed'] else 'not met'
    protocol = frozen['selected']['C']['protocol']
    design = ('C retains the exact ordinary brief root review and its worked solution for '
              '128-token answer-first repairs; its hidden state includes mentor correctness.'
              if protocol == 'full-packet' else
              'The selected medium-review variant declares its deeper reasoning budget and '
              '128-token answer-first guided format.' if protocol in ('medium', 'root-medium', 'mentor-reuse') else
              'C uses the ordinary review and guided-repair protocols, with every acquired review charged.')
    lines = ['# Frozen controller confirmation', '',
             f'The requested joint accuracy and compute targets are **{target}** on all 912 reserved questions.', '',
             table(joint), '',
             'The baselines are the five fixed stop/repair/retry policies. Every method uses the same '
             'frozen compact solver and initial response, with at most two additional solver calls. '
             'A uses a POMDP with learned observations, B uses the independently selected ordinary LLM judge, '
             'and C uses the POMDP with paid LLM observations.', '',
             '| Required comparison | Observed | Target | Result |',
             '|---|---:|---:|---|']
    for key, title in (
        ('A_above_highest_baseline', 'A above highest baseline'),
        ('B_above_highest_baseline', 'B above highest baseline'),
        ('C_above_highest_A_B', 'C above higher A/B')):
        value = joint['margins'][key]
        passed = value['point'] >= .1 - 1e-12
        lines.append(f'| {title} | {100*value["point"]:+.2f} pp | +10 pp | {"Pass" if passed else "Miss"} |')
    savings = joint['margins']['C_compute_fraction_saved_vs_B']['point']
    lines.append(f'| C compute saved versus B | {100*savings:+.2f}% | Above 0% | {"Pass" if savings>0 else "Miss"} |')
    lines += ['', 'Paired 95% intervals:', '']
    for key, title in (
        ('A_above_highest_baseline', 'A minus highest baseline'),
        ('B_above_highest_baseline', 'B minus highest baseline'),
        ('C_above_highest_A_B', 'C minus higher A/B'),
        ('C_above_B', 'C minus B')):
        value = joint['margins'][key]
        low, high = value['ci95']
        lines.append(f'- {title}: {100*value["point"]:+.2f} pp [{100*low:+.2f}, {100*high:+.2f}].')
    value = joint['margins']['C_compute_fraction_saved_vs_B']
    lines.append(f'- C compute saved versus B: {100*value["point"]:+.2f}% '
                 f'[{100*value["ci95"][0]:+.2f}, {100*value["ci95"][1]:+.2f}].')
    for name, panel in summary.items():
        if name == 'joint':
            continue
        lines += ['', '## ' + name + f' panel ({panel["questions"]} questions)', '', table(panel)]
    lines += ['', '## Interpretation and reproducibility', '',
              'Both source panels were reserved before their scores were examined. The original 192 panel '
              'and the additional 720 panel are reported separately as well as jointly. Bootstrap draws '
              'preserve the subject-by-source composition and recompute the highest baseline and higher '
              'A/B score inside every paired draw.', '',
              'Fresh means unused in this study\'s fitting and model selection. Pretraining '
              'contamination has not been measured.', '',
              'Average LLM tokens include every solver and physical review input/output phase. The '
              'compute proxy additionally charges each actual encoder token slot at its parameter count: '
              '`(4*solver_tokens +3*review_tokens +encoder_billions*actual_slots)/3.09`. '
              'Offline training and response collection are outside this per-question inference proxy. '
              'Raw tokens/calls and the 35B total parameter sensitivity are retained in the JSON metrics.', '',
              f'Selected C protocol: `{protocol}`. ' + design + ' The original fixed '
              'references and strongest ordinary B are retained.', '',
              'Where the frozen settings include `by_subject`, the POMDP cost penalty and '
              'review threshold were selected separately by observed subject on development '
              'data; individual questions never select their own setting.', '',
              'The audit checks independent cost accounting, blinded replay using paid prefixes only, '
              'training-only model reconstruction, frozen source and weight hashes, and regenerated '
              'neural observations. The fixed baseline response prefix remains unchanged.', '',
              'Raw GPQA questions, options, answer strings, generated rationales, and text-bearing '
              'neural caches remain private. This artifact contains aggregate metrics, sanitized '
              'correctness/cost decisions, model provenance and audit receipts. Trained observer '
              'weights remain in the private Modal volume named in the frozen manifest.']
    return '\n'.join(lines) + '\n'


def run(args):
    frozen = json.loads((args.study / 'frozen.json').read_bytes())
    if evidence(args.data) != frozen['evidence']:
        raise ValueError('Changed frozen evidence')
    verify_fixed_prefix(frozen, args.data)
    for name in ('development-audit.json', 'test-audit.json'):
        value = json.loads((args.study / name).read_bytes())
        if not value['passed'] or value['freeze_sha256'] != sha(args.study / 'frozen.json'):
            raise ValueError('Publish only after both audit phases pass')
        if value['independent_encoder_regenerations'] <= 0:
            raise ValueError('Independently regenerate neural observations in both audit phases')
        if name == 'development-audit.json' and value['training_only_refitted_outcomes'] != 480:
            raise ValueError('Both selected planners must be independently refitted')
        if name == 'test-audit.json' and (value['scored_summary_sha256'] != sha(args.study / 'test/summary.json')
                                         or value['scored_outcomes_sha256'] != sha(args.study / 'test/per-question.jsonl')):
            raise ValueError('Scored results changed after audit')
    summary = json.loads((args.study / 'test/summary.json').read_bytes())
    manifest = json.loads((args.data / 'splits/selection.json').read_bytes())
    ids = {qid for values in manifest['strata'].values() for qid in values}
    outcomes = [json.loads(s) for s in (args.study / 'test/per-question.jsonl').read_text().splitlines()]
    expected = {(qid, policy) for qid in ids for policy in METHODS}
    if len(outcomes) != len(expected) or {(r['id'], r['policy']) for r in outcomes} != expected:
        raise ValueError('Incomplete or duplicate published outcomes')
    for item in outcomes:
        validate_outcome(item, ids)
    if args.output.exists():
        raise ValueError('Use a new artifact path; do not overwrite results')
    args.output.mkdir(parents=True)
    # Explicit allowlist. Never walk the raw test data or neural-cache trees.
    for name in ('frozen.json', 'development-audit.json', 'test-audit.json'):
        shutil.copy2(args.study / name, args.output / name)
    for name in ('summary.json', 'per-question.jsonl'):
        shutil.copy2(args.study / 'test' / name, args.output / name)
    (args.output / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'REPORT.md').write_text(report_text(summary, frozen))
    receipt = {'raw_question_text_present': False, 'answer_strings_present': False,
               'rationales_present': False, 'neural_cache_present': False,
               'source_freeze_sha256': sha(args.study / 'frozen.json'),
               'packager_sha256': sha(__file__),
               'files': {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / 'artifact-manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'artifact': str(args.output), 'questions': len(ids),
                      'all_required_passed': summary['joint']['all_required_passed']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
