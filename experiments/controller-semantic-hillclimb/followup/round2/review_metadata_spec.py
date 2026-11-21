"""Prespecified training-only paid-review diagnostic; no model work on import."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SEED = 2026112229
FEATURES = [
    'subject_math', 'subject_chemistry', 'subject_physics',
    'protocol_root', 'protocol_critical', 'protocol_independent',
    'action_stop', 'action_repair', 'action_retry',
    'answer_present', 'parse_valid', 'json_complete', 'agreement', 'current_answer_present',
    'p_current', 'log_phase_count', 'log_review_input_tokens', 'log_review_output_tokens',
    'total_output_fraction', 'first_output_fraction', 'last_output_fraction',
    'first_at_cap', 'last_at_cap', 'log_thought_chars', 'log_thought_words',
    'log_check_words', 'log_current_words', 'log_answer_chars',
    'contrast_markers_per100', 'uncertainty_markers_per100',
    'error_markers_per100', 'assumption_markers_per100']
MARKERS = {
    'contrast': r'\b(?:however|but|although|instead|alternatively)\b',
    'uncertainty': r'\b(?:uncertain|unsure|cannot|unclear|ambiguous)\b',
    'error': r'\b(?:contradicts?|contradiction|contradictory|inconsistent|mistake|incorrect|wrong)\b',
    'assumption': r'\b(?:assume|assumes|assuming|assumption|approximate|approximation)\b'}
RULE_FEATURES = FEATURES[:15]
TEMPERATURES = [.5, .75, 1., 1.5, 2., 3.]
MODELS = {
    'subject_protocol_prior': {'beta_prior': [1., 1.], 'global_shrinkage_weight': 2., 'temperature': 1.},
    'confidence_agreement_logistic': {'C': 1., 'max_iter': 1000, 'standardize': True, 'features': RULE_FEATURES},
    'metadata_logistic': {'C': 1., 'max_iter': 1000, 'standardize': True, 'features': FEATURES},
    'metadata_hist7': {'max_leaf_nodes': 7, 'max_iter': 100, 'min_samples_leaf': 15,
        'l2_regularization': 5., 'learning_rate': .1, 'early_stopping': False,
        'random_state': SEED, 'max_bins': 255, 'features': FEATURES}}
REPORTS = ['new_root', 'new_science_root', 'all_root', 'old_root', 'all_reviews',
           'root_math', 'root_chemistry', 'root_physics', 'critical', 'independent']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(args):
    if args.output.exists():
        raise ValueError('Use a new diagnostic registration area')
    root = json.loads((args.root / 'contract.json').read_bytes())
    critical = json.loads((args.critical / 'contract.json').read_bytes())
    independent = json.loads((args.independent / 'contract.json').read_bytes())
    science = {r['id'] for r in root['selection'] if r['bank'] == 'new' and r['subject'] != 'math'}
    if (len(root['selection']) != 192 or len(science) != 64
            or {r['id'] for r in critical['selection']} != science
            or {r['id'] for r in independent['selection']} != science):
        raise ValueError('Require original root192 and both exact new-science64 siblings')
    # Collection status only; no reference answers or correctness values read.
    bank = args.independent / 'train.jsonl'
    status = [] if not bank.exists() else [json.loads(line) for line in bank.read_text().splitlines()]
    completed = sum(r['kind'] == 'judge' for r in status)
    recipe = {'schema': 'paid-review-metadata-diagnostic-v1',
        'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'specification_sha256': sha(__file__),
        'source_directories': {name: str(getattr(args, name).resolve()) for name in ('root', 'critical', 'independent')},
        'source_contract_sha256': {name: sha(getattr(args, name) / 'contract.json') for name in ('root', 'critical', 'independent')},
        'root_complete_sha256': sha(args.root / 'complete.json'),
        'critical_summary_sha256': sha(args.critical / 'summary.json'),
        'questions': 192, 'review_records_after_completion': 320,
        'root_records': 192, 'critical_records': 64, 'independent_records': 64,
        'independent_reviews_completed_at_registration': completed,
        'independent_summary_existed_at_registration': (args.independent / 'summary.json').exists(),
        'features': FEATURES, 'marker_patterns': MARKERS, 'models': MODELS,
        'feature_interpretation': 'Surface markers only; not proof of reasoning contradictions. All features from paid review/current prefix; no learned logits or reference values.',
        'training_weights': 'Each question has total weight1 across all its sibling review records.',
        'outer_folds': 'int(SHA256(question_id)[:8],16)%3; all sibling protocols held together',
        'inner_folds': 'int(SHA256(review-meta-inner-v1:question_id)[:8],16)%3 within each outer training subset',
        'temperature_grid': TEMPERATURES,
        'calibration': 'For each learned model and outer fold, choose temperature by weighted BCE on inner question-excluded predictions from outer-training questions only; then refit outer classifier on all outer training questions.',
        'invalid_answer': 'Retain every record; force all model probabilities to1e-6 when teacher answer is empty, using observed parse state only.',
        'report_panels': REPORTS, 'primary_panels': ['new_root', 'new_science_root'],
        'metrics': ['Brier', 'binary_log_loss', 'AUROC', 'accuracy_at_half', 'coverage_precision_at_25_50_75_100_percent'],
        'reporting_weights': 'Equal total question weight within each reported panel.',
        'confidence_intervals': '2000 paired question-cluster bootstrap draws for Brier/log-loss changes versus subject/protocol prior and confidence/agreement logistic on the two primary panels.',
        'no_hyperparameter_or_feature_selection': True, 'no_runtime_policy_changes': True,
        'full_scoring_requires_all320_verified_reviews': True,
        'no_project_development_test_values': True, 'neural_calls': 0, 'cloud_calls': 0, 'seed': SEED}
    args.output.mkdir(parents=True)
    path = args.output / 'registration.json'
    path.write_text(json.dumps(recipe, indent=2) + '\n')
    print(json.dumps({'registration_sha256': sha(path), 'independent_completed': completed,
                      'diagnostic_labels_scored': False, 'neural_calls': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'critical', 'independent', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    register(parser.parse_args())
