"""Fit the critical-review state extension from every completed training pair."""

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import committed
from prepare_mixed_mentor import plain_data, sha, fingerprint
import mentor_pairs
import second_review_pilot as pilot
import second_review_pomdp as policy
import stronger_base_hybrid as primary_core


def read(path):
    return json.loads(Path(path).read_bytes())


def wait(path, allowed):
    while not path.exists():
        if not allowed:
            raise ValueError('Finish all registered pairs/models or use --wait: ' + str(path))
        time.sleep(30)
    return read(path)


def training_records(args):
    summary = wait(args.pilot / 'summary.json', args.wait)
    contract = read(args.pilot / 'contract.json')
    prefix_receipt = read(args.pilot / 'prefix-receipt.json')
    parent = read(args.mentor_bank / 'contract.json')
    completion = read(args.mentor_bank / 'complete.json')
    if (contract['source_sha256'], contract['training_question_sha256'], contract['test_used']) != (
            sha(pilot.__file__), sha(args.data / 'splits/train-questions.jsonl'), False):
        raise ValueError('Changed pilot source, training questions or split')
    if summary['questions'] != 64 or summary['test_used'] or summary['bank_sha256'] != sha(args.pilot / 'train.jsonl'):
        raise ValueError('Use the whole completed64-question training pilot')
    if (summary['contract_sha256'], summary['prefix_receipt_sha256'], prefix_receipt['parent_bank_sha256']) != (
            sha(args.pilot / 'contract.json'), sha(args.pilot / 'prefix-receipt.json'), sha(args.mentor_bank / 'train.jsonl')):
        raise ValueError('Pilot evidence changed')
    if (completion['bank_sha256'], contract['parent_contract_sha256']) != (
            sha(args.mentor_bank / 'train.jsonl'), sha(args.mentor_bank / 'contract.json')):
        raise ValueError('Original mentor fitting bank changed')
    if contract['selection'] != pilot.selection(parent):
        raise ValueError('Critical fit changed its registered science cohort')
    rows, trees = plain_data(args.data, 'train'); by_id = {r['id']: r for r in rows}
    strong = {(r['id'], r['kind'], r['path']): r['response'] for r in committed(args.mentor_bank / 'train.jsonl')}
    raw = committed(args.pilot / 'train.jsonl')
    if any(r['path'] != ('guided' if r['kind'] == 'judge' else 'guided/critical-repair') for r in raw):
        raise ValueError('Unexpected critical pilot path')
    extra = {(r['id'], r['kind']): r['response'] for r in raw}
    if len(extra) != len(raw) or len(extra) != 128:
        raise ValueError('Critical pilot is missing or repeats a paired outcome')
    result = []
    for selected in contract['selection']:
        qid = selected['id']; row = by_id[qid]; safe = {**row, 'answer': ''}
        root_review = strong[qid, 'judge', '']; current = strong[qid, 'solver', 'guided']
        critical = extra[qid, 'judge']; repair = extra[qid, 'solver']; retry = strong[qid, 'solver', 'guided/retry']
        expected_prefix = {'root_solver_sha256': fingerprint(trees[qid]['']),
                           'root_review_sha256': fingerprint(root_review), 'guided_solver_sha256': fingerprint(current)}
        if prefix_receipt['controls'][qid] != expected_prefix:
            raise ValueError('The critical pilot used another paid prefix')
        mentor_pairs.verify_review(safe, trees[qid][''], root_review, 1024, 512)
        pilot.verify_review(safe, root_review, current, critical)
        import hashlib
        prompt = pilot.short_repair(safe, current, critical)
        if (repair['prompt_sha256'], repair['seed'], repair['output_cap'], repair['model_digest'],
                repair['temperature'], repair['context_length']) != (
                hashlib.sha256(prompt.encode()).hexdigest(), mentor_pairs.seed_for(qid, 'guided/reuse'),
                128, pilot.old.MODEL_DIGEST, .65, 4096):
            raise ValueError('Critical repair was not generated from its paid review')
        hard = int(sum(policy.mentor_core.correct(trees[qid][p], row['answer']) for p in policy.mentor_core.core.FIXED) / 5 < .6)
        current_correct = int(policy.mentor_core.correct(current, row['answer']))
        first_correct = int(policy.mentor_core.correct(root_review, row['answer']))
        result.append({'id': qid, 'subject': row['subject'],
                       'state8': policy.mentor_core.state(2 * hard + current_correct, first_correct),
                       'critical_correct': int(policy.mentor_core.correct(critical, row['answer'])),
                       'signal': policy.signal(critical, current),
                       'outcomes': [int(policy.mentor_core.correct(n, row['answer'])) for n in (repair, retry)],
                       'solver_compute': [4 * (n['input_tokens'] + n['output_tokens']) / 3.09 for n in (repair, retry)],
                       'review_compute': 3 * (critical['input_tokens'] + critical['output_tokens']) / 3.09})
    return result


def run(args):
    records = training_records(args)
    source = wait(args.primary_models / 'complete.json', args.wait)
    if source['identity_sha256'] != sha(args.primary_models / 'identity.json') or source.get('test_used', False):
        raise ValueError('Changed primary hybrid fitting provenance')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'source_sha256': sha(__file__), 'policy_sha256': sha(policy.__file__),
                'pilot_source_sha256': sha(pilot.__file__), 'primary_runtime_sha256': sha(primary_core.__file__),
                'parent_models_directory': str(args.primary_models.resolve()),
                'critical_parent_complete_sha256': sha(args.primary_models / 'complete.json'),
                'parent_identity_sha256': sha(args.primary_models / 'identity.json'),
                'pilot_directory': str(args.pilot.resolve()), 'pilot_summary_sha256': sha(args.pilot / 'summary.json'),
                'pilot_contract_sha256': sha(args.pilot / 'contract.json'), 'pilot_bank_sha256': sha(args.pilot / 'train.jsonl'),
                'mentor_bank_sha256': sha(args.mentor_bank / 'train.jsonl'),
                'train_question_sha256': sha(args.data / 'splits/train-questions.jsonl'),
                'records_sha256': fingerprint(records), 'questions': 64,
                'subjects': {'chemistry': 32, 'physics': 32}, 'smoothing': args.smoothing,
                'new_parameter_fit_uses_development_or_test': False, 'neural_calls': 0, 'cloud_calls': 0,
                'prior_mentor_role': 'Original192-pair dynamics are fixed; critical-specific parameters fit only64complete paired science questions.',
                'critical_state': 'H/current/original_mentor/critical_mentor,16states; teacher truths hidden at inference'}
    path = args.output / 'identity.json'
    if path.exists() and read(path) != identity:
        raise ValueError('Changed second-review fitting inputs')
    path.write_text(json.dumps(identity, indent=2) + '\n')
    models = []
    for spec in source['models']:
        source_path = args.primary_models / (spec['key'] + '.pkl')
        if sha(source_path) != spec['sha256']:
            raise ValueError('Changed primary hybrid model')
        for smoothing in args.smoothing:
            prior = pickle.loads(source_path.read_bytes())
            if not isinstance(prior, primary_core.Planner):
                prior = primary_core.build(prior.base, 'repairability_pomdp', {}, prior,
                                            {'original_hybrid_sha256': spec['sha256']})
            critical = policy.fit_critical(records, smoothing)
            receipt = {'critical_state_count': 16, 'maximum_reviews': 2,
                       'conditional_state_counts': {'unqueried_plain': 8, 'root_reviewed': 8, 'critical_reviewed': 16},
                       'critical_state_bits': ['difficulty', 'current_correct', 'original_mentor_correct', 'critical_mentor_correct'],
                       'primary': prior.fit_receipt, 'critical_training_questions': 64,
                       'critical_smoothing': smoothing, 'fit_identity_sha256': sha(path),
                       'critical_review_availability': 'Only after guided first continuation, with one solver call remaining',
                       'critical_observation': 'Parsed current-confidence bucket and proposal/current agreement; no additional neural pass',
                       'original_mentor_bit_preserved': True, 'maximum_additional_solver_calls': 2,
                       'root_planning': 'Finite-observation Bellman backup includes optional second critical review',
                       'no_critical_labels_imputed_to_original_192': True}
            model = policy.Planner(prior, critical, receipt)
            key = spec['key'] + f'-critical{smoothing:g}'
            output = args.output / (key + '.pkl'); output.write_bytes(pickle.dumps(model, protocol=5))
            models.append({'key': key, 'sha256': sha(output), 'parent_sha256': spec['sha256'],
                           'smoothing': smoothing, 'semantic_review': spec['semantic_review']})
    complete = {'models': models, 'identity_sha256': sha(path), 'base_A': source.get('base_A'),
                'critical_training_questions': 64, 'test_used': False, 'neural_calls': 0, 'cloud_calls': 0}
    (args.output / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'mentor-bank', 'pilot', 'primary-models', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--smoothing', type=float, nargs='+', default=[8., 32.])
    p.add_argument('--wait', action='store_true')
    run(p.parse_args())
