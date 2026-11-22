"""Verify and audit the full TRAIN science Complete/strong-mentor input join.

Preparation only: no transfer/model fitting, development search, neural call,
cloud call or heldout values. The selected parent C and matched-B extension
must still be explicitly registered before a later fit/development experiment.
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import completion_mentor_transfer as transfer
import expanded_mentor_inputs as expanded
import expanded_mentor_io as io
import fit_expanded_mentor as fitting
import science_complete_actions as complete
import training_sparse_text as guarded

MATCHED_B_PLAN = {
    'status': 'Required before development claim; not implemented or evaluated by this preparation',
    'retain_existing_best_B': True,
    'math': 'Exact retained best B math route/settings',
    'review': 'Same already-paid strong root review, original prompt/model/physical phases',
    'inputs': 'Only paid review action/confidence/agreement and current-answer validity; no NN or labels',
    'new_rule_family': {
        'root_stop': ['LLM action', 'proposal agreement', .8, .95, .995, 1.],
        'first_repair_style': ['ordinary', 'retained guided'],
        'repair_or_retry': 'Follow LLM action; retry is ordinary, repair uses declared style',
        'last_call': ['Complete if current answer invalid', 'Complete unconditionally'],
        'guided_child': 'Stop on proposal agreement, otherwise retained reuse; never Complete',
        'root_stop_invalid': 'Complete consumes first continuation; never submit teacher answer',
        'candidate_count':24},
    'same_horizon': 'At most two additional compact calls; Complete is terminal',
    'stronger_B_retained': True,
    'additional_controls': 'All original five plus new fixed/validity Complete controls on same panel',
    'not_a_silent_change_to_old_B': True}


def load_inputs(expanded_fit, complete_directory):
    registered = io.read(expanded_fit/'registration.json')
    fitting.verify_complete(expanded_fit, io.sha(expanded_fit/'registration.json'))
    args = SimpleNamespace(**{k:Path(v) for k,v in registered['paths'].items()})
    # This existing TRAIN-only reconstruction validates every source response,
    # root-review phase, exact pure-retry alias and original observer input.
    bundle = expanded.load_inputs(args)
    bundle = fitting.bind_registered_input_identity(bundle, io.read(args.neural/'registration.json'))
    rows, trees, complete_receipt = complete.training_data(args.new_data, complete_directory)
    records, joined, receipt = transfer.join(rows, trees, bundle['rows'], bundle['trees'], bundle['judges'])
    evidence = {**registered['input_evidence'], **bundle['identity']['input_evidence'],
        str((expanded_fit/'registration.json').resolve()):io.sha(expanded_fit/'registration.json'),
        str((expanded_fit/'complete.json').resolve()):io.sha(expanded_fit/'complete.json'),
        str((complete_directory/'complete.json').resolve()):complete_receipt['complete_sha256']}
    evidence.update({str((complete_directory/name).resolve()):digest for name,digest in complete_receipt['digests'].items()})
    evidence.update(complete_receipt['public_input_evidence']['files'])
    io.verify(evidence)
    return records, {**receipt, 'input_evidence':evidence,
        'expanded_fit_directory':str(expanded_fit.resolve()), 'complete_directory':str(complete_directory.resolve()),
        'complete_grid_receipt':complete_receipt, 'expanded_input_identity':bundle['identity'],
        'hindsight_availability':transfer.availability(rows,joined),
        'matched_B_plan':MATCHED_B_PLAN, 'parent_C_selection':'Pending explicit strongest-C registration',
        'actual_transfer_fit_performed':False, 'model_or_controller_accuracy_reported':False}


def run(args):
    with guarded.no_model_calls():
        if args.output.exists(): raise ValueError('Preserve existing preparation; use a new output directory')
        records, receipt = load_inputs(args.expanded_fit,args.complete)
        receipt['source_files'] = io.sources(sys.modules[__name__])
        args.output.mkdir(parents=True)
        io.jsonl(args.output/'private-transfer-records.jsonl', records)
        receipt['records_file_sha256'] = io.sha(args.output/'private-transfer-records.jsonl')
        io.write(args.output/'preparation.json',receipt)
        io.write(args.output/'hindsight-availability.json',receipt['hindsight_availability'])
        print(receipt['hindsight_availability'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('expanded_fit','complete','output'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    run(parser.parse_args())
