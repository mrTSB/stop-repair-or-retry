"""Read-only source/reference/parser audit of the512 Round2 training questions.

Decode only selected training source rows. Never read protected split records,
score solver correctness, run inference, or alter a bank. Private outputs contain
aggregate counts, hashes and issue IDs, without question/option/reference text.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
import extended_acquire as old
from semantic_data import previous
from examples import text_pair
from science_auxiliary_data import option_entry
from scoring import correct

LETTERS = 'ABCDEFGHIJ'
PLAIN = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split()).casefold()


def science_rows(path, ids):
    wanted = {qid.removeprefix('supergpqa-') for qid in ids}
    pattern = re.compile(rb'"uuid"\s*:\s*"([a-zA-Z0-9-]+)"')
    found = {}
    with path.open('rb') as stream:
        for line in stream:
            match = pattern.search(line)
            if match is None or match[1].decode() not in wanted:
                continue
            row = json.loads(line)
            if row['uuid'] not in wanted or row['uuid'] in found:
                raise ValueError('Duplicate or inconsistent selected science source UUID')
            found[row['uuid']] = row
    if set(found) != wanted:
        raise ValueError('Missing selected science source UUID')
    return {'supergpqa-' + qid: row for qid, row in found.items()}


def math_rows(path, ids):
    import pyarrow as pa
    indices = {int(qid.removeprefix('gsm8k-train-')): qid for qid in ids}
    if any(qid != f'gsm8k-train-{i:05d}' for i, qid in indices.items()):
        raise ValueError('Unexpected GSM8K training source identifier')
    found = {}; offset = 0
    with pa.memory_map(str(path), 'r') as stream:
        for batch in pa.ipc.open_stream(stream):
            local = sorted(i - offset for i in indices if offset <= i < offset + len(batch))
            # Materialize only selected training rows, not the entire source.
            for i, row in zip(local, batch.take(pa.array(local, type=pa.int64())).to_pylist()):
                found[indices[offset + i]] = row
            offset += len(batch)
    if set(found) != set(ids):
        raise ValueError('Missing selected GSM8K source index')
    return found


def check_science(row, source, revision):
    errors, warnings = [], []
    options = [value.strip() for value in source['options']]
    letter = source['answer_letter'].strip().upper()
    stem = source['question'].strip()
    expected = stem + '\n\nOptions:\n' + '\n'.join(f'{LETTERS[i]}. {v}' for i, v in enumerate(options)) + old.SCIENCE_SUFFIX
    if row['question'] != expected:
        errors.append('science_question_or_option_order_differs_from_official_source')
    if row['answer'] != letter:
        errors.append('science_reference_letter_differs_from_official_source')
    if (row['domain'], row['subject'], row.get('source')) != ('science', source['field'].casefold(), 'm-a-p/SuperGPQA@' + revision):
        errors.append('science_domain_subject_or_source_revision_mismatch')
    if source['discipline'] != 'Science' or source['field'].casefold() not in ('chemistry', 'physics'):
        errors.append('unsupported_science_subject')
    if not 4 <= len(options) <= 10 or not all(options) or len(set(options)) != len(options):
        errors.append('invalid_official_choice_structure')
    if len(letter) != 1 or letter not in LETTERS[:len(options)]:
        errors.append('official_reference_outside_options')
    elif normalize(source['answer']) != normalize(options[LETTERS.index(letter)]):
        errors.append('official_answer_text_and_letter_disagree')
    if len({normalize(value) for value in options}) != len(options):
        warnings.append('choices_duplicate_after_unicode_case_whitespace_normalization')
    safe = {**row, 'answer': ''}
    for letter, value in zip(LETTERS, options):
        if old.parse_answer('science', 'Final: ' + letter) != letter:
            errors.append('solver_parser_rejects_official_option_letter')
        try:
            entry = option_entry(safe, letter, 98)
            if entry['right'] != 'Proposed answer: ' + letter + '. ' + value:
                errors.append('option_observer_extracts_different_candidate_text')
            _, right = text_pair(safe, {'answer': letter, 'text': ''})
            if right != 'Proposed answer: ' + letter + ': ' + value + '\nSolution:\n':
                errors.append('rationale_observer_extracts_different_candidate_text')
        except (KeyError, ValueError, IndexError):
            errors.append('candidate_input_parser_rejects_stored_options')
    return sorted(set(errors)), sorted(set(warnings)), len(options)


def check_math(row, source):
    errors = []
    if row['question'] != source['question']:
        errors.append('math_question_differs_from_official_source_index')
    if '####' not in source['answer']:
        errors.append('math_source_reference_marker_missing')
    elif row['answer'] != source['answer'].rsplit('####', 1)[1].strip():
        errors.append('math_reference_differs_from_official_final_value')
    if (row['domain'], row['subject']) != ('math', 'math'):
        errors.append('math_domain_or_subject_mismatch')
    return errors


def prompt_audit(data, rows):
    complete = json.loads((data / 'train-plain-complete.json').read_bytes())
    with (data / 'train-observations.jsonl').open('rb') as stream:
        prefix = b''.join(next(stream) for _ in range(complete['base_records']))
    if hashlib.sha256(prefix).hexdigest() != complete['base_bank_sha256']:
        raise ValueError('Sealed training response prefix changed')
    trees = {r['id']: {} for r in rows}
    for line in prefix.splitlines():
        record = json.loads(line)
        if record['kind'] == 'solver' and record['path'] in PLAIN:
            if record['path'] in trees[record['id']]:
                raise ValueError('Duplicate plain response')
            trees[record['id']][record['path']] = record['response']
    errors, possible_window_risks, stats, checked = [], [], defaultdict(list), []
    for row in rows:
        tree = trees[row['id']]
        if set(tree) != set(PLAIN):
            raise ValueError('Require all seven plain training responses')
        safe = {**row, 'answer': ''}
        for path in PLAIN:
            text = old.solver_prompt(safe, path, tree, {})
            node = tree[path]; cap = 256 if path else 128
            issues = []
            if hashlib.sha256(text.encode()).hexdigest() != node['prompt_sha256']:
                issues.append('recorded_prompt_hash_differs_from_full_reconstruction')
            if row['question'].removesuffix(old.SCIENCE_SUFFIX) not in text:
                issues.append('prompt_builder_omits_question_or_options')
            if node.get('context_length', 4096) != 4096 or node.get('output_cap', cap) != cap:
                issues.append('unexpected_recorded_context_or_generation_cap')
            if node['input_tokens'] <= 0 or node['input_tokens'] + cap > 4096:
                possible_window_risks.append({'id': row['id'], 'path': path, 'input_tokens': node['input_tokens'], 'reserved_output_cap': cap})
            if issues:
                errors.append({'id': row['id'], 'path': path, 'issues': issues})
            stats[row['subject']].append({'input_tokens': node['input_tokens'], 'prompt_utf8_bytes': len(text.encode()),
                                        'context_headroom_after_output_reservation': 4096 - node['input_tokens'] - cap})
            checked.append({'id': row['id'], 'path': path, 'prompt_sha256': node['prompt_sha256'],
                            'input_tokens': node['input_tokens'], 'reserved_output_cap': cap})
    aggregate = {subject: {'nodes': len(values), 'max_input_tokens': max(v['input_tokens'] for v in values),
                          'max_prompt_utf8_bytes': max(v['prompt_utf8_bytes'] for v in values),
                          'min_context_headroom_after_output_reservation': min(v['context_headroom_after_output_reservation'] for v in values)}
                 for subject, values in stats.items()}
    return {'nodes': len(checked), 'sealed_prefix_sha256': complete['base_bank_sha256'],
            'prompt_alignment_sha256': digest(checked), 'errors': errors, 'possible_window_risks': possible_window_risks,
            'by_subject': aggregate,
            'limit': 'Reconstruction and returned token counts check context-window risk; no tokenizer replay or server-side execution trace was available.'}


def run(args):
    if args.output.exists():
        raise ValueError('Use a new private audit output directory')
    manifest_path = args.data / 'splits/selection.json'
    manifest = json.loads(manifest_path.read_bytes())
    training = args.data / 'splits/train-questions.jsonl'
    if sha(training) != manifest['splits']['train']['sha256']:
        raise ValueError('Registered training question file changed')
    rows = [json.loads(line) for line in training.read_bytes().splitlines()]
    if len(rows) != 512 or len({r['id'] for r in rows}) != 512 or Counter(r['subject'] for r in rows) != {'math': 256, 'chemistry': 128, 'physics': 128}:
        raise ValueError('Training IDs or prescribed50/25/25subject counts changed')
    source = args.science / 'SuperGPQA-all.jsonl'
    pin = json.loads((args.science / 'pin.json').read_bytes())
    if pin['repo'] != manifest['science']['repo'] or pin['revision'] != manifest['science']['revision'] or sha(source) != manifest['science']['sha256']:
        raise ValueError('Pinned science source differs from the reservation')
    if sha(args.math) != manifest['math_sha256']:
        raise ValueError('Pinned math source differs from the reservation')
    official = science_rows(source, {r['id'] for r in rows if r['domain'] == 'science'})
    official.update(math_rows(args.math, {r['id'] for r in rows if r['domain'] == 'math'}))
    errors, warnings, alignment = [], [], []
    options = Counter(); letter_histogram = Counter(); parser_accepts = Counter()
    schema = defaultdict(Counter)
    visual = []; special = []; normalized_stems = defaultdict(list)
    for row in rows:
        raw = official[row['id']]
        schema[row['subject']].update(raw.keys())
        if row['domain'] == 'science':
            problems, cautions, count = check_science(row, raw, pin['revision'])
            options[str(count)] += 1; letter_histogram[row['answer']] += 1
        else:
            problems, cautions = check_math(row, raw), []
        parsed = old.parse_answer(row['domain'], 'Final: ' + row['answer'])
        accepts = correct({'answer': parsed}, row['answer'])
        parser_accepts[row['subject']] += int(accepts)
        if not accepts:
            problems.append('solver_parser_cannot_represent_reference')
        for word in ('<|im_start|>', '<|im_end|>', '</think>'):
            if word in row['question']:
                special.append({'id': row['id'], 'marker': word})
        stem = raw['question'].strip()
        norm = normalize(stem)
        normalized_stems[digest(norm)].append(row['id'])
        image_fields = [key for key, value in raw.items() if re.search(r'image|diagram|picture', key, re.I) and value]
        explicit_visual = bool(re.search(r'(?:shown|illustrated|depicted|displayed).{0,65}(?:figure|diagram|sketch|graph)|(?:figure|diagram|sketch).{0,45}(?:below|above|shown)', stem, re.I | re.S))
        if image_fields or explicit_visual:
            visual.append({'id': row['id'], 'nonempty_visual_source_fields': image_fields,
                           'explicit_external_visual_reference_pattern': explicit_visual,
                           'question_sha256': digest(stem), 'classification': 'Needs source-context review; no outcome-based exclusion applied'})
        if problems:
            errors.append({'id': row['id'], 'subject': row['subject'], 'issues': sorted(set(problems))})
        if cautions:
            warnings.append({'id': row['id'], 'subject': row['subject'], 'issues': cautions})
        alignment.append({'id': row['id'], 'subject': row['subject'], 'source_record_sha256': digest(raw),
                          'stored_row_sha256': digest(row), 'question_sha256': digest(row['question']),
                          'reference_sha256': digest(row['answer']), 'reference_parser_accepts': bool(accepts),
                          'source_alignment_passed': not problems})
    prompts = prompt_audit(args.data, rows)
    files = {str(path.resolve()): sha(path) for path in (manifest_path, training, source, args.science / 'pin.json', args.math)}
    code = {str(Path(path).resolve()): sha(path) for path in {__file__, inspect.getfile(previous.gsm8k), inspect.getfile(old.solver_prompt),
            inspect.getfile(old.parse_answer), inspect.getfile(correct), inspect.getfile(option_entry), inspect.getfile(text_pair)}}
    result = {'schema': 'round2-training-source-alignment-audit-v1', 'training_only': True,
              'no_development_test_question_response_values_read': True, 'inference_calls': 0,
              'solver_correctness_outcomes_scored': False, 'frozen_banks_modified': False,
              'questions': 512, 'subjects': dict(Counter(r['subject'] for r in rows)),
              'decoded_source_rows': {'gsm8k': 256, 'supergpqa': 256},
              'integrity_passed': not errors and not prompts['errors'], 'source_alignment_errors': errors,
              'schema_warnings': warnings, 'science_options_count_histogram': dict(options),
              'science_reference_letter_histogram': dict(letter_histogram), 'reference_parser_acceptance': dict(parser_accepts),
              'source_schema_fields': {subject: dict(fields) for subject, fields in schema.items()},
              'duplicate_normalized_training_stems': [ids for ids in normalized_stems.values() if len(ids) > 1],
              'model_control_token_markers_in_questions': special, 'external_visual_context_flags': visual,
              'prompt_checks': prompts, 'training_alignment_sha256': digest(alignment),
              'input_files': files, 'source_files': code,
              'scope': 'Reference transport/schema/parser checks; does not independently establish scientific correctness of official gold labels.'}
    args.output.mkdir(parents=True)
    (args.output / 'training-alignment.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in alignment))
    result['training_alignment_file_sha256'] = sha(args.output / 'training-alignment.jsonl')
    (args.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('integrity_passed', 'questions', 'source_alignment_errors',
                'schema_warnings', 'reference_parser_acceptance', 'external_visual_context_flags')}), flush=True)
    print(json.dumps({'output': str(args.output.resolve()), 'receipt_sha256': sha(args.output / 'receipt.json')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--science', type=Path, required=True)
    parser.add_argument('--math', type=Path, default=previous.MATH)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
