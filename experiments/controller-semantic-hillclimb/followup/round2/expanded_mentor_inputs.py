"""Training-only provenance and exact cache alignment for320 mentor pairs.

No observer loading, model forward pass, cloud call, or heldout-data access is
permitted. The caller separately schedules the960 genuinely missing inputs.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import fit_new_source_mentor as parent_fit
import prepare_mixed_mentor as prepare
import science_mentor_expansion as expansion

read, sha, fingerprint = parent_fit.read, prepare.sha, prepare.fingerprint
WIDTHS = {'state': 4, 'plain': 8, 'guided': 8}
PAIR_COUNTS = {'math': 64, 'chemistry': 128, 'physics': 128}
SCHEMA = 'expanded-newsource-mentor-inputs-v1'


def verify_files(mapping):
    for path, value in mapping.items():
        if sha(path) != value:
            raise ValueError('Changed registered input/source: ' + str(path))


def complete_expansion(directory, data, mentor):
    """Validate the entire completed acquisition without opening dev values."""
    complete = read(directory / 'complete.json')
    record = read(directory / 'registration.json')
    required = {'train.jsonl', 'raw-generations.jsonl', 'registration.json', 'prerequisites.json'}
    optional = {'transport-errors.jsonl'} if (directory / 'transport-errors.jsonl').exists() else set()
    if (complete.get('schema') != expansion.SCHEMA or complete.get('questions') != 192
            or complete.get('records') != 960 or complete.get('source_matched_total_pairs') != 320
            or complete.get('references_supplied') is not False
            or complete.get('training_labels_scored') is not False
            or complete.get('development_or_test_values_decoded') is not False
            or complete.get('neural_calls') != 0 or complete.get('outcome_filtering') is not False
            or set(complete.get('digests', {})) != required | optional):
        raise ValueError('Require the sealed complete192-question science expansion')
    verify_files({str(directory / name): value for name, value in complete['digests'].items()})
    if (complete['registration_sha256'] != sha(directory / 'registration.json')
            or complete['bank_sha256'] != sha(directory / 'train.jsonl')
            or complete['fitting_contract'] != record['fitting_contract']):
        raise ValueError('Expansion seal and fitting contract disagree')
    expansion.verify_sources(record)
    saved = {name: Path(path) for name, path in record['paths'].items()}
    if (saved['data'].resolve() != data.resolve() or saved['parent_bank'].resolve() != mentor.resolve()
            or saved['output'].resolve() != directory.resolve()):
        raise ValueError('Expansion belongs to another training reservation or mentor parent')
    args = SimpleNamespace(**saved, proposal_sha256=record['proposal_sha256'],
        priority_completion=[Path(p) for p in record['priority_completions']])
    if expansion.registration(args) != record:
        raise ValueError('Frozen expansion registration cannot be reconstructed')
    rows, plain, chosen, proposal = expansion.inputs(args)
    records = expansion.protocol.strict_records(directory / 'train.jsonl')
    raw = expansion.protocol.strict_records(directory / 'raw-generations.jsonl')
    _, reviews, known, coverage = expansion.restore(rows, plain, records, raw)
    expected = {(r['id'], kind, path) for r in rows for kind, path in expansion.PATHS}
    physical = 192 * 3 + sum(len(j['calls']) for j in reviews.values())
    if (set(known) != expected or len(records) != 960 or coverage != complete['raw_coverage']
            or complete['unique_physical_generations'] != physical
            or len(raw) != physical or coverage['unique_raw_generations_covered'] != physical
            or coverage['raw_generations_stored'] != physical):
        raise ValueError('Incomplete or unmatched expansion physical/logical grid')
    evidence = {str((directory / name).resolve()): sha(directory / name)
                for name in ['complete.json', *sorted(complete['digests'])]}
    for name in ('proposal.json', 'selection.json'):
        path = saved['proposal'] / name; evidence[str(path.resolve())] = sha(path)
    # Prerequisites are hash-bound here. Do not call prerequisite_receipts(): it
    # reads completed DEV/B summaries, which are irrelevant to training fitting.
    return records, chosen, {'complete_sha256': sha(directory / 'complete.json'),
        'registration_sha256': sha(directory / 'registration.json'), 'raw_coverage': coverage,
        'physical_generations': physical, 'input_evidence': evidence,
        'prerequisites_rechecked_as_bytes_only': True,
        'proposal_sha256': record['proposal_sha256'], 'selection_sha256': proposal['selection_sha256']}


def alignment(metadata, inputs, cached_metadata, outputs, expected_missing_ids, expected_sizes=(5184, 4224, 960)):
    """Exact all-field metadata matching; never relabel cached prompt outputs."""
    if len(metadata) != len(inputs):
        raise ValueError('Expected input/metadata alignment differs')
    key = lambda r: (r['id'], r['kind'], r['path'])
    expected_keys, cached_keys = [key(r) for r in metadata], [key(r) for r in cached_metadata]
    if len(set(expected_keys)) != len(expected_keys) or len(set(cached_keys)) != len(cached_keys):
        raise ValueError('Duplicate expected or cached observation key')
    if not set(cached_keys) <= set(expected_keys):
        raise ValueError('Cached parent observation absent from expanded training input')
    if set(outputs) != set(WIDTHS):
        raise ValueError('Unknown or missing cached neural output heads')
    for name, width in WIDTHS.items():
        array = np.asarray(outputs[name])
        if array.shape != (len(cached_metadata), width) or not np.isfinite(array).all():
            raise ValueError('Malformed cached neural outputs: ' + name)
    lookup = {key(r): i for i, r in enumerate(cached_metadata)}
    reused, missing = {}, []
    required_fields = {'id', 'kind', 'path', 'fold', 'train', 'valid', 'label', 'prompt_sha256', 'input_sha256'}
    for i, (item, entry) in enumerate(zip(metadata, inputs)):
        if (set(item) != required_fields or item['fold'] != 'full' or item['input_sha256'] != fingerprint(entry)
                or item['kind'] not in ('solver', 'original-review')
                or type(item['train']) is not bool or type(item['valid']) is not bool
                or type(item['label']) is not int or item['label'] not in range(4)):
            raise ValueError('Changed expected input/fold/label/valid metadata semantics')
        cached = lookup.get(key(item))
        if cached is not None:
            if item != cached_metadata[cached]:
                raise ValueError('Parent cache changed id/kind/path/prompt/input/fold/label/valid alignment')
            reused[i] = cached
        else:
            missing.append(i)
    allowed = {(qid, 'original-review', '') for qid in expected_missing_ids}
    allowed |= {(qid, 'solver', path) for qid in expected_missing_ids
                for path in ('guided', 'guided/reuse', 'guided/retry', 'retry/reuse')}
    if {expected_keys[i] for i in missing} != allowed:
        raise ValueError('Missing inputs differ from the exact five new observations per science question')
    if (len(metadata), len(reused), len(missing)) != tuple(expected_sizes):
        raise ValueError('Expected5184 inputs,4224 exact reuses and960 missing observations')
    if set(reused.values()) != set(range(len(cached_metadata))):
        raise ValueError('Every original cached observation must be retained exactly once')
    return reused, missing


def extraction_plan(metadata, inputs, reused_indexes, missing_indexes, encoder_recipe_sha256):
    if sorted([*reused_indexes, *missing_indexes]) != list(range(len(metadata))):
        raise ValueError('Extraction plan must partition every expected observation')
    rows = [{'expected_index': i, **metadata[i]} for i in missing_indexes]
    if any(r['fold'] != 'full' for r in rows):
        raise ValueError('New training questions require the old full unseen encoder')
    return {'schema': SCHEMA, 'expected_rows': len(metadata), 'reused_rows': len(reused_indexes),
        'missing_rows': len(missing_indexes), 'missing': rows,
        'encoder_recipe_sha256': encoder_recipe_sha256,
        'expected_metadata_sha256': fingerprint(metadata), 'expected_inputs_sha256': fingerprint(inputs),
        'reuse_alignment_sha256': fingerprint([[i, j] for i, j in sorted(reused_indexes.items())]),
        'missing_inputs_sha256': fingerprint([inputs[i] for i in missing_indexes]),
        'missing_metadata_sha256': fingerprint([metadata[i] for i in missing_indexes]),
        'missing_kind_path_counts': dict(Counter(r['kind'] + ':' + r['path'] for r in rows)),
        'no_inference_started': True, 'runtime_terminal_observations_required': False,
        'terminal_note': 'Terminal node arrays are a fitting cache only; forced terminal actions require no runtime emission or NN pass',
        'training_only': True, 'development_or_test_used': False}


def load_inputs(args):
    """Args: new_data,mentor,expansion,parent,verifier,legacy_examples,base."""
    with parent_fit.no_neural_inference():
        expansion_records, new_pairs, acquisition = complete_expansion(args.expansion, args.new_data, args.mentor)
        complete, identity = read(args.parent / 'complete.json'), read(args.parent / 'identity.json')
        receipt = read(args.parent / 'neural-receipt.json')
        if (complete['identity_sha256'] != sha(args.parent / 'identity.json')
                or complete.get('development_or_test_used') is not False
                or complete.get('fit_questions') != 512 or complete.get('paired_review_questions') != 128
                or complete.get('new_source_only') is not True
                or identity['source_sha256'] != sha(parent_fit.__file__)):
            raise ValueError('Require the completed newsource128 parent')
        verify_files(identity['source_files'])
        for model in complete['models']:
            if sha(args.parent / (model['key'] + '.pkl')) != model['sha256']:
                raise ValueError('Changed completed newsource parent model')
        original_parent = Path(identity['parent_models_directory'])
        if (sha(original_parent / 'complete.json') != identity['parent_complete_sha256']
                or sha(original_parent / 'identity.json') != identity['parent_identity_sha256']):
            raise ValueError('Original mixed-parent ancestry changed')
        inherited_args = SimpleNamespace(**{**vars(args), 'parent': original_parent})
        rows, parent_trees, parent_judges, prior_metadata, prior_outputs, base_path, base_spec, validated = parent_fit.load_inputs(inherited_args)
        if (receipt['metadata_sha256'] != sha(args.parent / 'metadata.jsonl')
                or receipt['outputs_sha256'] != sha(args.parent / 'neural.npz')
                or receipt['input_sha256'] != validated['input_sha256'] or receipt['n'] != 4224
                or identity['input_sha256'] != validated['input_sha256']
                or identity['base_model_sha256'] != validated['base_model_sha256']
                or identity['inherited_temperature'] != validated['inherited_temperature']
                or complete['base_A'] != base_spec):
            raise ValueError('Newsource128 cache/temperature/base ancestry changed')
        cached_metadata = [json.loads(line) for line in (args.parent / 'metadata.jsonl').read_text().splitlines()]
        if cached_metadata != prior_metadata:
            raise ValueError('Stored newsource128 metadata differs from reconstructed training inputs')
        with np.load(args.parent / 'neural.npz') as stored:
            cached_outputs = {name: stored[name].copy() for name in stored.files}
        if set(cached_outputs) != set(prior_outputs) or any(
                not np.array_equal(cached_outputs[name], prior_outputs[name]) for name in prior_outputs):
            raise ValueError('Newsource128 neural cache is not the exact original-parent subset')
        recipe_path = args.verifier / 'full/recipe.json'; recipe = read(recipe_path)
        verify_files({str(recipe_path.parent / name): value for name, value in recipe['weights'].items()})
        if identity['encoder_recipe_sha256'] != sha(recipe_path):
            raise ValueError('Full unseen encoder changed')
        old_pairs = validated['selected_pairs']
        if ({v['id'] for v in old_pairs} & {v['id'] for v in new_pairs}
                or Counter(v['subject'] for v in [*old_pairs, *new_pairs]) != PAIR_COUNTS):
            raise ValueError('Expanded320 pairs must retain128 and add every192 science question')
        selected = [*old_pairs, *new_pairs]
        selected_ids = {v['id'] for v in old_pairs}
        original_records = [v for v in prepare.committed(args.mentor / 'train.jsonl') if v['id'] in selected_ids]
        plain = {qid: {p: n for p, n in tree.items() if p in prepare.PLAIN} for qid, tree in parent_trees.items()}
        trees, judges = prepare.overlay(rows, plain, [*original_records, *expansion_records], selected, validate=True)
        if len(judges) != 320 or any(judges[key] != value for key, value in parent_judges.items()):
            raise ValueError('Expanded reviews changed an original retained pair')
        if any(trees[v['id']]['guided/retry'] != trees[v['id']]['repair/retry'] for v in selected):
            raise ValueError('The logical guided/retry alias must retain the exact paid pure-retry response')
        observer = parent_fit.mentor.perception.Observer(str((args.verifier / 'full').resolve()),
            validated['inherited_temperature'], device='cpu')
        metadata, inputs = prepare.entries(rows, trees, judges, set(), observer)
        reused, missing = alignment(metadata, inputs, cached_metadata, cached_outputs, {v['id'] for v in new_pairs})
        plan = extraction_plan(metadata, inputs, reused, missing, sha(recipe_path))
        evidence = dict(acquisition['input_evidence'])
        evidence.update({str((args.parent / name).resolve()): sha(args.parent / name) for name in
            ('complete.json', 'identity.json', 'neural-receipt.json', 'metadata.jsonl', 'neural.npz')})
        evidence.update({str(recipe_path.resolve()): sha(recipe_path), str(base_path.resolve()): sha(base_path)})
        result_identity = {'schema': SCHEMA, 'source_sha256': sha(__file__), 'input_evidence': evidence,
            'newsource_parent_directory': str(args.parent.resolve()), 'newsource_parent_identity': identity,
            'validated_parent_provenance': validated, 'expansion_receipt': acquisition,
            'selected_pairs': selected, 'review_subject_counts': PAIR_COUNTS,
            'training_questions': 512, 'paired_review_questions': 320, 'input_rows': 5184,
            'cached_rows': 4224, 'missing_rows': 960, 'encoder_recipe_sha256': sha(recipe_path),
            'inherited_temperature': validated['inherited_temperature'],
            'tree_sha256': fingerprint(trees), 'metadata_sha256': fingerprint(metadata),
            'inputs_sha256': fingerprint(inputs), 'extraction_plan_sha256': fingerprint(plan),
            'ordinary_dynamics_paths': list(prepare.PLAIN), 'old_question_ids_included': False,
            'neural_inference_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False}
        return {'rows': rows, 'trees': trees, 'judges': judges, 'metadata': metadata, 'inputs': inputs,
            'cached_outputs': cached_outputs, 'reused_indexes': reused, 'missing_indexes': missing,
            'base_path': base_path, 'base_spec': base_spec, 'identity': result_identity, 'extraction_plan': plan}


def training_headroom(rows, trees, judges):
    """Complete training-only science availability, never a policy score."""
    science = [r for r in rows if r['subject'] in ('chemistry', 'physics')]
    if len(science) != 256 or Counter(r['subject'] for r in science) != {'chemistry': 128, 'physics': 128}:
        raise ValueError('Headroom requires all256 training science questions')
    paths = [*prepare.PLAIN, 'guided', 'guided/reuse', 'guided/retry', 'retry/reuse']
    panels = {}
    for subject, panel in [('all_science', science), *[(s, [r for r in science if r['subject'] == s]) for s in ('chemistry', 'physics')]]:
        if any(set(paths) - set(trees[r['id']]) or (r['id'], '') not in judges for r in panel):
            raise ValueError('Incomplete per-question expanded science action grid')
        if any(trees[r['id']]['guided/retry'] != trees[r['id']]['repair/retry'] for r in panel):
            raise ValueError('Changed guided/retry alias: no extra independent solver response exists')
        correctness = {p: [bool(prepare.correct(trees[r['id']][p], r['answer'])) for r in panel] for p in paths}
        plain = [any(correctness[p][i] for p in prepare.PLAIN) for i in range(len(panel))]
        expanded = [any(correctness[p][i] for p in paths) for i in range(len(panel))]
        panels[subject] = {'questions': len(panel), 'paths': {p: {'available': len(panel), 'correct': sum(v),
            'accuracy': sum(v)/len(panel)} for p, v in correctness.items()},
            'original_seven_path_oracle_correct': sum(plain), 'expanded_eleven_path_oracle_correct': sum(expanded),
            'expanded_ten_distinct_response_oracle_correct': sum(expanded),
            'logical_paths_per_question': 11, 'distinct_solver_responses_per_question': 10,
            'new_correct_beyond_plain_oracle': sum(b and not a for a, b in zip(plain, expanded)),
            'teacher_correct_not_a_submittable_solver_output': sum(prepare.correct(judges[r['id'], ''], r['answer']) for r in panel)}
    return {'training_only': True, 'development_or_test_used': False, 'panels': panels,
        'interpretation': 'Hindsight availability diagnostic, not achievable controller accuracy; all original and expanded science questions retained',
        'alias': 'guided/retry equals repair/retry exactly:11 logical paths,10 distinct paid solver responses',
        'neural_calls': 0, 'cloud_calls': 0}
