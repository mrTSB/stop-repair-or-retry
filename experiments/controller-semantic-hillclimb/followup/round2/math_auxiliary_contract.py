"""Public-only contract for all seven ordinary paths on512 extra math questions."""
from pathlib import Path

import expanded_mentor_io as io
import reserve_math_auxiliary as reserved
from auxiliary_second_paired_contract import fingerprint
import science_complete_actions as complete

SCHEMA = 'auxiliary-math512-plain7-TRAIN-v1'
PLAIN = complete.pilot.protocol.PLAIN
PATHS = PLAIN


def public_inputs(args):
    prior = io.read(args.reservation/'registration.json'); done = io.read(args.reservation/'complete.json')
    if (prior['schema'] != reserved.SCHEMA or done['schema'] != reserved.SCHEMA
        or done['registration_sha256'] != io.sha(args.reservation/'registration.json')
        or done['questions'] != 512 or done['reference_values_decoded'] or done['model_calls']
        or prior['legacy_encoder_overlap'] or prior['historical_or_reserved_overlap']
        or prior['protected_reference_values_decoded'] or not prior['source_question_column_only']):
        raise ValueError('Require the sealed public-only disjoint512 math reservation')
    io.verify(prior['source_files']); io.verify({str(args.reservation/k):v for k,v in done['digests'].items()})
    data = Path(prior['paths']['data']); selection = io.read(data/'splits/selection.json')
    if prior['input_evidence'][str((data/'splits/selection.json').resolve())] != io.sha(data/'splits/selection.json'):
        raise ValueError('Protected membership reservation changed')
    for split in ('train','dev','test'):
        path = data/'splits'/f'{split}-questions.jsonl'
        if prior['input_evidence'][str(path.resolve())] != selection['splits'][split]['sha256']:
            raise ValueError('Reservation does not exclude the current protected splits')
    source = reserved.reserve.previous.MATH
    if prior['math_source_sha256'] != io.sha(source): raise ValueError('Pinned math source changed')
    rows = complete.pilot.protocol.strict_records(args.reservation/'train-questions.jsonl')
    if (len(rows) != 512 or len({r['id'] for r in rows}) != 512 or fingerprint(rows) != prior['public_rows_sha256']
        or any(r['answer'] != '' or r['domain'] != 'math' or r['subject'] != 'math' for r in rows)):
        raise ValueError('Changed complete reference-blank math cohort')
    files = [args.reservation/'registration.json', args.reservation/'complete.json', args.reservation/'train-questions.jsonl',
             data/'splits/selection.json', source]
    evidence = {str(p.resolve()):io.sha(p) for p in files}
    return rows, {'files':evidence, 'reserved_questions':512, 'reference_values_decoded':False,
                  'protected_question_records_read':False, 'frozen_observer_ancestry_bound_by_reservation':True}


def priority_receipt(path):
    expected = io.PROJECT/'runs/controller-semantic-followup/round2/evidence-first-textbook-root-review-pilot-v3/complete.json'
    if path.resolve() != expected: raise ValueError('Finish the active evidence-first experiment before more cloud work')
    done = io.read(path); record = io.read(path.parent/'registration.json')
    if (done['registration_sha256'] != io.sha(path.parent/'registration.json') or done['coverage']['physical_calls'] != 320
        or not done['training_only'] or done['test_used'] or done['scored']):
        raise ValueError('Prior collector is incomplete')
    io.verify(record['source_files']); io.verify({str(path.parent/k):v for k,v in done['digests'].items()})
    capacity = complete.pilot.protocol.strict_records(path.parent/'capacity.jsonl')
    if capacity[-1] != {'min_containers':0, 'max_containers':2}: raise ValueError('Prior collector must release capacity')
    return {'complete':str(path.resolve()), 'sha256':io.sha(path), 'registration_sha256':done['registration_sha256']}
