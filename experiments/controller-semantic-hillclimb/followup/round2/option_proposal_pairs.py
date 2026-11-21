"""Registered training-only pilot of discriminative option feedback to repair.

This adds an answer-proposal feedback action; it is not pure retry-gate tuning.
All options are ranked and charged, on every fixed-hash selected question.
Registration reads no checkpoint, calibration result, or model outcome.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary, committed
from semantic_data import read
from scoring import correct
from transfer_variants import actor
from transport import resilient
import extended_acquire as old
from science_auxiliary_data import LETTERS, PROFILE, SCIENCE_SUFFIX

SEED = 2026112161
APP = 'srr-followup-solver-20261120'
PROTOCOL = 'discriminative-option-proposal-repair-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def selection(rows):
    selected = []
    for subject in ('chemistry', 'physics'):
        candidates = sorted((r for r in rows if r['subject'] == subject),
                            key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).digest())
        if len(candidates) < 32:
            raise ValueError('Need32registered science training questions per subject')
        selected.extend({'id': r['id'], 'subject': subject} for r in candidates[:32])
    if len({r['id'] for r in selected}) != 64:
        raise ValueError('Duplicate selected training IDs')
    return sorted(selected, key=lambda r: hashlib.sha256(f'{SEED}:order:{r["id"]}'.encode()).digest())


def register(args):
    if args.output.exists():
        raise ValueError('Use a new option-proposal registration directory')
    qpath = args.data / 'splits/train-questions.jsonl'
    rows = read(qpath)
    split = json.loads((args.data / 'splits/selection.json').read_bytes())['splits']['train']
    if len(rows) != 512 or sha(qpath) != split['sha256']:
        raise ValueError('Use the full registered512training pool')
    chosen = selection(rows)
    files = [Path(__file__), Path(__file__).with_name('science_option_observer.py'),
             Path(__file__).with_name('science_auxiliary_data.py')]
    receipt = {'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
               'source_sha256': {p.name: sha(p) for p in files}, 'selection': chosen,
               'archived_protocol_sha256': {inspect.getfile(f): sha(inspect.getfile(f))
                                            for f in (actor, old.repair_prompt)},
               'subject_counts': dict(Counter(r['subject'] for r in chosen)),
               'question_split_sha256': sha(qpath),
               'selection_rule': 'Fixed hash64science training questions:32chemistry32physics; all used independent of outcomes.',
               'observer_output': str(args.verifier.resolve()), 'observer_profile': PROFILE,
               'observer_training_registration_sha256': sha(args.verifier / 'training-registration.json'),
               'observer_checkpoint_or_calibration_results_read': False,
               'feedback': 'Top option plus calibrated binary logit and top-versus-runner-up logit margin; explicit fallibility caveat; no generated rationale.',
               'ranking': 'All official options in original order, single-answer passes; descending binary margin, alphabetical ties.',
               'solver_digest': old.MODEL_DIGEST, 'solver_cap': 256, 'temperature': .65,
               'solver_seed': 'Same original logical repair seed for each question',
               'action_change': 'Existing repair_prompt optional feedback now receives a non-LLM discriminative option proposal.',
               'cost': 'Charge root solver + new repair solver + every option-encoder pass and actual unpadded slots; no free discarded candidates.',
               'comparators': ['Stop', 'Repair once', 'Repair twice'],
               'scope': 'Training-only action pilot; not fresh validation or a controller result',
               'app': APP, 'max_cloud_gpus': 2, 'reference_answers_supplied_at_inference': False}
    args.output.mkdir(parents=True)
    (args.output / 'registration.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'questions': 64, 'registration_sha256': sha(args.output / 'registration.json'),
                      'checkpoint_or_calibration_results_read': False, 'inference_started': False}), flush=True)


def options(row):
    if 'options' in row:
        result = list(row['options'])
    else:
        body = row['question'].removesuffix(SCIENCE_SUFFIX).partition('\n\nOptions:\n')[2]
        chunks = re.split(r'(?m)^([A-J])\. ', body)
        result = []
        for i in range(1, len(chunks), 2):
            if chunks[i] != LETTERS[len(result)]:
                raise ValueError('Noncanonical option order')
            result.append(chunks[i + 1].strip())
    if not 4 <= len(result) <= 10 or not all(result):
        raise ValueError('Preserve every official option')
    return result


def rank_options(observer, row):
    safe = {**row, 'answer': ''}
    choices = options(safe)
    scores = []
    for i, text in enumerate(choices):
        letter = LETTERS[i]
        path = 'candidate-' + letter
        value = observer.values(safe, path, {path: {'answer': letter}})
        raw = np.asarray(value[0], dtype=float)
        logit = float(np.logaddexp(raw[1], raw[3]) - np.logaddexp(raw[0], raw[2])) / observer.temperature
        scores.append({'letter': letter, 'calibrated_logit': logit, 'token_slots': int(value[3]),
                       'input_sha256': canonical_sha(observer.entry(safe, path, {path: {'answer': letter}}))})
    ordered = sorted(scores, key=lambda r: (-r['calibrated_logit'], r['letter']))
    return {'all_options': scores, 'proposed_letter': ordered[0]['letter'],
            'proposed_option': choices[LETTERS.index(ordered[0]['letter'])],
            'calibrated_logit': ordered[0]['calibrated_logit'],
            'calibrated_margin': ordered[0]['calibrated_logit'] - ordered[1]['calibrated_logit'],
            'encoder_calls': len(scores), 'encoder_token_slots': sum(r['token_slots'] for r in scores),
            'encoder_parameters_billions': observer._recipe['parameters'] / 1e9}


def feedback(proposal):
    return ('A discriminative option scorer suggests ' + proposal['proposed_letter'] + '. '
            + proposal['proposed_option'] + '\n'
            + f'Its calibrated correctness logit is {proposal["calibrated_logit"]:.3f}; '
            + f'the top-versus-next option logit margin is {proposal["calibrated_margin"]:.3f}.\n'
            + 'This fallible statistical suggestion may be wrong and supplies no worked solution. '
            + 'Independently check the science and calculations before accepting or rejecting it.')


def repair_seed(qid):
    return int(hashlib.sha256(f'{old.SEED}:{qid}:repair'.encode()).hexdigest()[:8], 16) % 2**31


def charges(root, repair, proposal):
    solver_tokens = sum(n['input_tokens'] + n['output_tokens'] for n in (root, repair))
    neural = proposal['encoder_parameters_billions'] * proposal['encoder_token_slots'] / 3.09
    return {'solver_tokens': solver_tokens, 'solver_calls': 2,
            'encoder_calls': proposal['encoder_calls'], 'encoder_token_slots': proposal['encoder_token_slots'],
            'compute': solver_tokens * 4. / 3.09 + neural}


def run(args):
    registration = json.loads((args.output / 'registration.json').read_bytes())
    for filename, digest in registration['source_sha256'].items():
        if sha(Path(__file__).with_name(filename)) != digest:
            raise ValueError('Registered proposal pilot source changed')
    for filename, digest in registration['archived_protocol_sha256'].items():
        if sha(filename) != digest:
            raise ValueError('Archived actor or repair prompt changed')
    if str(args.verifier.resolve()) != registration['observer_output']:
        raise ValueError('Wrong registered auxiliary observer')
    if sha(args.verifier / 'training-registration.json') != registration['observer_training_registration_sha256']:
        raise ValueError('Auxiliary training plan changed')
    if sha(args.data / 'splits/train-questions.jsonl') != registration['question_split_sha256']:
        raise ValueError('Registered training pool changed')
    while not (args.verifier / 'training-summary.json').exists():
        time.sleep(30)
    summary = json.loads((args.verifier / 'training-summary.json').read_bytes())
    if not summary.get('calibration_used_reloaded_archive'):
        raise ValueError('Use the calibrated deployed checkpoint')
    profile_sha = sha(args.verifier / 'full/recipe.json')
    if profile_sha != summary['recipe_sha256']:
        raise ValueError('Auxiliary recipe changed after calibration')
    rows, trees, _, _ = ordinary(args.data, 'train')
    by_id = {r['id']: r for r in rows}
    if selection(rows) != registration['selection']:
        raise ValueError('Changed registered cohort')
    controls = {item['id']: {p: trees[item['id']][p] for p in ('', 'repair', 'repair/repair')}
                for item in registration['selection']}
    runtime = {'registration_sha256': sha(args.output / 'registration.json'),
               'recipe_sha256': profile_sha, 'control_bank_sha256': canonical_sha(controls),
               'training_summary_sha256': sha(args.verifier / 'training-summary.json')}
    runtime_path = args.output / 'runtime.json'
    if runtime_path.exists() and json.loads(runtime_path.read_bytes()) != runtime:
        raise ValueError('Changed runtime model or matched controls')
    runtime_path.write_text(json.dumps(runtime, indent=2) + '\n')
    from science_option_observer import from_checkpoint
    import modal
    observer = from_checkpoint(args.verifier / 'full', args.device)
    observer.load()
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    actual = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    if actual[old.MODEL] != old.MODEL_DIGEST:
        raise ValueError('Compact solver weights changed')
    destination = args.output / 'train.jsonl'
    raw = destination.read_bytes() if destination.exists() else b''
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Incomplete last receipt requires explicit recovery')
    saved = committed(destination)
    known = {r['id']: r for r in saved}
    if len(known) != len(saved) or set(known) - set(controls):
        raise ValueError('Duplicate or unregistered proposal receipt')
    with destination.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, item in enumerate(registration['selection']):
            qid = item['id']; row = {**by_id[qid], 'answer': ''}
            if qid not in known:
                proposal = rank_options(observer, row)
                prompt = old.repair_prompt(row, controls[qid][''], feedback(proposal))
                response = actor(request, row, prompt, repair_seed(qid), 256)
                response['protocol'] = PROTOCOL
                record = {'id': qid, 'profile_sha256': profile_sha, 'proposal': proposal, 'response': response,
                          'cost': charges(controls[qid][''], response, proposal)}
                stream.write(json.dumps(record) + '\n'); stream.flush()
                known[qid] = record
            record = known[qid]
            proposal = record['proposal']
            if ([r['letter'] for r in proposal['all_options']] != list(LETTERS[:len(options(row))])
                    or proposal['encoder_calls'] != len(options(row))
                    or proposal['encoder_token_slots'] != sum(r['token_slots'] for r in proposal['all_options'])
                    or proposal['encoder_parameters_billions'] != observer._recipe['parameters'] / 1e9
                    or record['cost'] != charges(controls[qid][''], record['response'], proposal)):
                raise ValueError('Saved proposal omits a candidate or has inconsistent cost')
            prompt = old.repair_prompt(row, controls[qid][''], feedback(record['proposal']))
            if (record['profile_sha256'], record['response']['model_digest'], record['response']['prompt_sha256'],
                record['response']['seed'], record['response']['output_cap'], record['response']['temperature']) != (
                    profile_sha, old.MODEL_DIGEST, hashlib.sha256(prompt.encode()).hexdigest(), repair_seed(qid), 256, .65):
                raise ValueError('Saved proposal response no longer matches the registered call')
            if (index + 1) % 8 == 0:
                print(f'Option-proposal training pairs {index + 1}/64', flush=True)
    methods = {}
    for name, paths in (('Stop', ['']), ('Repair once', ['', 'repair']), ('Repair twice', ['', 'repair', 'repair/repair'])):
        tokens = [sum(controls[q][p]['input_tokens'] + controls[q][p]['output_tokens'] for p in paths) for q in controls]
        methods[name] = {'n': 64, 'correct': sum(correct(controls[q][paths[-1]], by_id[q]['answer']) for q in controls),
                         'mean_solver_tokens': float(np.mean(tokens)), 'mean_compute': float(np.mean(tokens)) * 4 / 3.09,
                         'mean_encoder_calls': 0., 'mean_encoder_token_slots': 0.}
    methods['Option-proposal repair'] = {'n': 64,
        'correct': sum(correct(known[q]['response'], by_id[q]['answer']) for q in controls),
        **{'mean_' + k: float(np.mean([known[q]['cost'][k] for q in controls]))
           for k in ('solver_tokens', 'compute', 'encoder_calls', 'encoder_token_slots')}}
    for method in methods.values():
        method['accuracy'] = method['correct'] / method['n']
    result = {'training_questions': 64, 'methods': methods, 'runtime_sha256': sha(runtime_path),
              'bank_sha256': sha(destination), 'fresh_data_used': False,
              'scope': registration['scope'], 'action_change': registration['action_change']}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'collect'))
    for name in ('data', 'verifier', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = p.parse_args()
    (register if args.phase == 'register' else run)(args)
