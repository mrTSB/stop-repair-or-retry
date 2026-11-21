"""Publish an audited, text-free confirmation report with an explicit allowlist."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from confirmation_common import HERE, BASELINES, evidence, sha, verify_fixed_prefix

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


def report_text(summary, frozen, registration=None, diagnosis=None):
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
    if registration is not None:
        lines[4:4] = [
            'This separately frozen follow-up uses the **same 912 questions** as the retained '
            'original confirmation. Its candidates were selected only on retired development '
            'data and frozen before their own replay and before any test grades were opened. '
            'Exact ungraded solver/judge responses and compatible raw encoder observations '
            'were reused, with every virtual policy call still charged. These are not an '
            'additional independent 912 test examples.', '']
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
    if diagnosis is not None:
        counts = diagnosis['panels']['joint']['families']['C']
        lines += ['', 'Post-grading failure diagnosis: of C\'s '
                  f'{counts["errors"]} errors, {counts["errors_without_a_paid_review"]} '
                  'had no paid review, '
                  f'{counts["errors_with_a_correct_paid_review"]} had a correct paid review '
                  'available, and '
                  f'{counts["errors_with_paid_reviews_all_wrong"]} had paid reviews that '
                  'were all wrong. These are hindsight categories, not estimates of '
                  'achievable controller gains.']
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
              'An additional independent encoder check samples every source-by-subject '
              'stratum using fixed question hashes, including the GPQA transfer panel. It '
              'replays only paid inputs with reference answers removed.', '',
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
    extra = json.loads((args.study / 'encoder-panel-audit.json').read_bytes())
    allowed = {'passed', 'freeze_sha256', 'source_sha256', 'neural_cache_sha256', 'sampling',
               'per_stratum', 'sampled_questions', 'strata', 'independent_encoder_regenerations',
               'references_provided_to_policy', 'checks'}
    if set(extra) != allowed or not extra['passed'] or extra['references_provided_to_policy']:
        raise ValueError('Invalid stratified encoder audit')
    if extra['freeze_sha256'] != sha(args.study / 'frozen.json') or extra['source_sha256'] != sha(HERE / 'audit_encoder_panels.py'):
        raise ValueError('Changed stratified audit source or controller freeze')
    if extra['neural_cache_sha256'] != sha(args.study / 'test-neural-cache.jsonl'):
        raise ValueError('Changed stratified audit inputs')
    expected_strata = {p + '/' + s for p in manifest['strata'] for s in ('math', 'chemistry', 'physics')}
    if set(extra['strata']) != expected_strata or min(extra['strata'].values()) <= 0:
        raise ValueError('Audit must cover every source-by-subject stratum')
    fields = {'family', 'id', 'panel', 'subject', 'path', 'kind', 'tokens', 'maximum_logit_difference'}
    covered = set()
    for item in extra['checks']:
        if set(item) != fields or item['id'] not in ids or item['family'] not in ('A', 'C'):
            raise ValueError('Unexpected stratified audit record')
        if item['path'] not in PATHS or item['kind'] not in ('solver', 'review'):
            raise ValueError('Unexpected stratified audit observation')
        if not isinstance(item['tokens'], int) or not isinstance(item['maximum_logit_difference'], (int, float)):
            raise ValueError('Unexpected numeric audit receipt')
        covered.add((item['family'], item['panel'] + '/' + item['subject']))
    if covered != {(f, s) for f in ('A', 'C') for s in expected_strata}:
        raise ValueError('Both observers must be checked in all source-by-subject strata')
    outcomes = [json.loads(s) for s in (args.study / 'test/per-question.jsonl').read_text().splitlines()]
    expected = {(qid, policy) for qid in ids for policy in METHODS}
    if len(outcomes) != len(expected) or {(r['id'], r['policy']) for r in outcomes} != expected:
        raise ValueError('Incomplete or duplicate published outcomes')
    for item in outcomes:
        validate_outcome(item, ids)
    diagnosis = None
    diagnostic = args.study / 'error-analysis.json'
    if diagnostic.exists():
        diagnosis = json.loads(diagnostic.read_bytes())
        expected = {'freeze_sha256', 'scored_outcomes_sha256', 'source_sha256',
                    'raw_examples_present', 'interpretation', 'panels'}
        if set(diagnosis) != expected or diagnosis['raw_examples_present']:
            raise ValueError('Unexpected diagnostic payload')
        if diagnosis['freeze_sha256'] != sha(args.study / 'frozen.json') or diagnosis['scored_outcomes_sha256'] != sha(args.study / 'test/per-question.jsonl'):
            raise ValueError('Wrong diagnostic result set')
        if diagnosis['source_sha256'] != sha(HERE / 'error_causes.py'):
            raise ValueError('Changed diagnostic code')
        if set(diagnosis['panels']) != set(summary):
            raise ValueError('Diagnostic source panels differ')
        categories = {'errors', 'errors_without_a_paid_review', 'errors_with_a_correct_paid_review',
                      'errors_with_paid_reviews_all_wrong', 'errors_with_a_correct_earlier_visited_answer',
                      'errors_where_initial_answer_was_correct'}
        for panel, item in diagnosis['panels'].items():
            if set(item) != {'questions', 'families', 'C_vs_B'} or item['questions'] != summary[panel]['questions']:
                raise ValueError('Unexpected diagnostic structure')
            if set(item['families']) != {'A', 'B', 'C'}:
                raise ValueError('Unexpected diagnostic controller')
            for family, values in item['families'].items():
                if set(values) != categories or not all(isinstance(v, int) and v >= 0 for v in values.values()):
                    raise ValueError('Unexpected diagnostic counts')
                if values['errors'] != item['questions'] - summary[panel]['methods'][family]['correct']:
                    raise ValueError('Diagnostic errors disagree with audited results')
            if set(item['C_vs_B']) != {'C_only_correct', 'B_only_correct', 'both_correct', 'both_wrong'}:
                raise ValueError('Unexpected paired diagnostic fields')
            if not all(isinstance(v, int) and v >= 0 for v in item['C_vs_B'].values()) or sum(item['C_vs_B'].values()) != item['questions']:
                raise ValueError('Invalid paired diagnostic counts')
    if args.output.exists():
        raise ValueError('Use a new artifact path; do not overwrite results')
    args.output.mkdir(parents=True)
    # Explicit allowlist. Never walk the raw test data or neural-cache trees.
    for name in ('frozen.json', 'development-audit.json', 'test-audit.json', 'encoder-panel-audit.json'):
        shutil.copy2(args.study / name, args.output / name)
    if diagnosis is not None:
        shutil.copy2(diagnostic, args.output / diagnostic.name)
    imported = args.study / 'neural-cache-import.json'
    if imported.exists():
        value = json.loads(imported.read_bytes())
        fields = {'source_freeze_sha256', 'target_freeze_sha256', 'source_cache_sha256',
                  'target_cache_sha256', 'target_cache_bytes', 'compatible_observations', 'rule', 'grades_opened'}
        if set(value) != fields or value['grades_opened'] or value['target_freeze_sha256'] != sha(args.study / 'frozen.json'):
            raise ValueError('Invalid label-free neural-cache import receipt')
        prefix = (args.study / 'test-neural-cache.jsonl').read_bytes()[:value['target_cache_bytes']]
        if hashlib.sha256(prefix).hexdigest() != value['target_cache_sha256']:
            raise ValueError('Imported neural receipt prefix changed')
        shutil.copy2(imported, args.output / imported.name)
    for name in ('summary.json', 'per-question.jsonl'):
        shutil.copy2(args.study / 'test' / name, args.output / name)
    (args.output / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'REPORT.md').write_text(report_text(summary, frozen, manifest.get('secondary_registration'), diagnosis))
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
