"""Synthetic-only exact alignment, blinding, full-grid and seal tests."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import expanded_mentor_inputs as run
from test_science_mentor_expansion import row, cohort, plain, Remote


class PublicObserver:
    def entry(self, safe, path, paid, judge=None):
        if safe['answer'] != '':
            raise AssertionError('Reference entered observation')
        allowed = {''} | ({path} if path else set())
        if '/' in path:
            allowed.add(path.split('/')[0])
        if set(paid) != allowed:
            raise AssertionError('Unpaid branch entered observation')
        return {'question': safe['question'], 'path': path,
                'paid': {key: value['text'] for key, value in paid.items()},
                'review': None if judge is None else judge['text']}


def grid():
    public, contract = cohort()
    rows = [{**r, 'answer': '2' if r['domain'] == 'math' else 'B'} for r in public]
    old_ids = {r['id'] for r in contract['selection'] if r['bank'] == 'new'}
    new_ids = {r['id'] for r in run.expansion.selection(public, contract)}
    def trees(ids):
        result, judges = {}, {}
        for i, r in enumerate(rows):
            tree = {p: {'text': 'paid-' + p, 'answer': 'A', 'prompt_sha256': r['id'] + ':' + p}
                    for p in run.prepare.PLAIN}
            if r['id'] in ids:
                tree.update({p: {'text': 'paid-' + p, 'answer': 'B', 'prompt_sha256': r['id'] + ':' + p}
                             for p in ('guided', 'guided/reuse', 'guided/retry', 'retry/reuse')})
                tree['guided/retry'] = tree['repair/retry']
                judges[r['id'], ''] = {'text': 'Paid review', 'answer': 'B', 'prompt_sha256': 'review-' + r['id']}
            result[r['id']] = tree
        return result, judges
    before, old_reviews = trees(old_ids)
    after, reviews = trees(old_ids | new_ids)
    old_metadata, _ = run.prepare.entries(rows, before, old_reviews, set(), PublicObserver())
    metadata, inputs = run.prepare.entries(rows, after, reviews, set(), PublicObserver())
    outputs = {name: np.arange(len(old_metadata) * width, dtype=np.float32).reshape(-1, width)
               for name, width in run.WIDTHS.items()}
    return rows, after, reviews, metadata, inputs, old_metadata, outputs, new_ids


class ExpandedInputTests(unittest.TestCase):
    def test_exact_full_5184_partition_and_all_parent_values_retained(self):
        rows, trees, judges, metadata, inputs, prior, outputs, new_ids = grid()
        reused, missing = run.alignment(metadata, inputs, prior, outputs, new_ids)
        self.assertEqual((len(reused), len(missing)), (4224, 960))
        self.assertEqual(len(judges), 320)
        self.assertEqual(set(reused.values()), set(range(4224)))
        plan = run.extraction_plan(metadata, inputs, reused, missing, 'recipe')
        self.assertEqual(plan['missing_kind_path_counts'], {'solver:guided': 192, 'solver:guided/reuse': 192,
            'solver:guided/retry': 192, 'solver:retry/reuse': 192, 'original-review:': 192})
        self.assertEqual(plan['missing_inputs_sha256'], run.fingerprint([inputs[i] for i in missing]))
        for i, j in reused.items():
            self.assertEqual(metadata[i], prior[j])
        self.assertTrue(all(r['fold'] == 'full' for r in metadata))

    def test_every_metadata_alignment_field_and_duplicate_fail_closed(self):
        *_, metadata, inputs, prior, outputs, new_ids = grid()
        changes = {'id': 'foreign', 'kind': 'original-review', 'path': 'changed', 'fold': 0,
                   'train': False, 'valid': False, 'label': 0, 'prompt_sha256': 'changed', 'input_sha256': 'changed'}
        # First synthetic label is hard+incorrect=2, so label0 changes it.
        for field, value in changes.items():
            with self.subTest(field=field):
                modified = copy.deepcopy(prior); modified[0][field] = value
                with self.assertRaises(ValueError):
                    run.alignment(metadata, inputs, modified, outputs, new_ids)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            run.alignment(metadata, inputs, prior + prior[:1], outputs, new_ids)
        with self.assertRaises(ValueError):
            run.alignment(metadata, inputs, prior, outputs, set(list(new_ids)[:-1]))

    def test_nonfinite_output_changed_input_and_missing_head_rejected(self):
        *_, metadata, inputs, prior, outputs, new_ids = grid()
        altered = {k: v.copy() for k, v in outputs.items()}; altered['state'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            run.alignment(metadata, inputs, prior, altered, new_ids)
        changed_inputs = copy.deepcopy(inputs); changed_inputs[0]['question'] = 'Changed public question'
        with self.assertRaisesRegex(ValueError, 'input/fold'):
            run.alignment(metadata, changed_inputs, prior, outputs, new_ids)
        with self.assertRaisesRegex(ValueError, 'heads'):
            run.alignment(metadata, inputs, prior, {'state': outputs['state']}, new_ids)

    def test_inputs_ignore_reference_and_future_answers_but_metadata_labels_change(self):
        rows, trees, judges, metadata, inputs, *_ = grid()
        altered = [{**r, 'answer': 'PRIVATE_REFERENCE'} for r in rows]
        new_meta, new_inputs = run.prepare.entries(altered, trees, judges, set(), PublicObserver())
        self.assertEqual(inputs, new_inputs)
        self.assertNotEqual([r['label'] for r in metadata], [r['label'] for r in new_meta])
        qid = rows[0]['id']; changed = copy.deepcopy(trees); changed[qid]['repair/repair']['text'] = 'UNPAID_SECRET'
        _, after = run.prepare.entries(rows, changed, judges, set(), PublicObserver())
        for i, item in enumerate(metadata):
            if item['id'] == qid and item['path'] in ('', 'repair', 'retry'):
                self.assertEqual(inputs[i], after[i])
        self.assertTrue(all('PRIVATE_REFERENCE' not in json.dumps(v) for v in inputs))

    def test_terminal_fitting_metadata_not_marked_as_runtime_training_nodes(self):
        *_, metadata, inputs, prior, outputs, new_ids = grid()
        trained = {r['path'] for r in metadata if r['train']}
        self.assertEqual(trained, {'', 'repair', 'retry', 'guided'})
        self.assertTrue(all(not r['train'] for r in metadata if r['kind'] == 'original-review' or '/' in r['path']))
        with run.parent_fit.no_neural_inference():
            with self.assertRaisesRegex(AssertionError, 'CPU-only'):
                run.parent_fit.mentor.perception.Observer('unused', 1.).load()

    def test_training_headroom_requires_all256_science_and_all11_paths(self):
        rows, trees, judges, *_ = grid()
        result = run.training_headroom(rows, trees, judges)
        all_science = result['panels']['all_science']
        self.assertEqual(all_science['questions'], 256)
        self.assertEqual(all_science['original_seven_path_oracle_correct'], 0)
        self.assertEqual(all_science['expanded_eleven_path_oracle_correct'], 256)
        self.assertEqual(all_science['new_correct_beyond_plain_oracle'], 256)
        self.assertEqual(len(all_science['paths']), 11)
        self.assertEqual(all_science['distinct_solver_responses_per_question'], 10)
        bad = copy.deepcopy(trees); del bad[next(r['id'] for r in rows if r['subject'] == 'chemistry')]['guided/retry']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            run.training_headroom(rows, bad, judges)
        with self.assertRaisesRegex(ValueError, 'all256'):
            run.training_headroom(rows[:-1], trees, judges)
        bad = copy.deepcopy(trees)
        qid = next(r['id'] for r in rows if r['subject'] == 'chemistry')
        bad[qid]['guided/retry'] = {**bad[qid]['guided/retry'], 'text': 'Changed alias'}
        with self.assertRaisesRegex(ValueError, 'alias'):
            run.training_headroom(rows, bad, judges)

    def test_complete_gate_checks_physical_grid_hashes_and_never_reads_dev_prerequisites(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder); directory = root / 'expansion'; directory.mkdir()
            data, mentor, proposal = root / 'data', root / 'mentor', root / 'proposal'; proposal.mkdir()
            public = [row(i, subject) for subject in ('chemistry', 'physics') for i in range(96)]
            controls = {r['id']: plain(r) for r in public}
            trees, reviews, known, _ = run.expansion.restore(public, controls, [], [])
            with (directory / 'train.jsonl').open('a') as bank, (directory / 'raw-generations.jsonl').open('a') as raw:
                remote = Remote(two_phase=True)
                journal = run.expansion.RequestJournal(remote, raw)
                collector = run.expansion.PairCollector(public, trees, reviews, known, bank, journal)
                for safe in public:
                    collector.one(safe)
            records = run.expansion.protocol.strict_records(directory / 'train.jsonl')
            raw = run.expansion.protocol.strict_records(directory / 'raw-generations.jsonl')
            _, _, _, coverage = run.expansion.restore(public, controls, records, raw)
            chosen = [{'id': r['id'], 'subject': r['subject'], 'bank': 'new'} for r in public]
            registration = {'paths': {'data': str(data), 'parent_bank': str(mentor), 'proposal': str(proposal),
                'output': str(directory), 'b_reference': str(root / 'DO_NOT_OPEN_DEV')},
                'proposal_sha256': 'proposal', 'priority_completions': [str(root / 'DO_NOT_OPEN_DEV')],
                'fitting_contract': {'fixed': True}, 'source_files': {}}
            for name, value in [('registration.json', registration), ('prerequisites.json', {'opaque_hash': 'only'})]:
                run.expansion.write(directory / name, value)
            for name in ('proposal.json', 'selection.json'):
                run.expansion.write(proposal / name, {})
            complete = {'schema': run.expansion.SCHEMA, 'questions': 192, 'records': 960,
                'source_matched_total_pairs': 320, 'references_supplied': False, 'training_labels_scored': False,
                'development_or_test_values_decoded': False, 'neural_calls': 0, 'outcome_filtering': False,
                'digests': {name: run.sha(directory / name) for name in
                    ('train.jsonl', 'raw-generations.jsonl', 'registration.json', 'prerequisites.json')},
                'registration_sha256': run.sha(directory / 'registration.json'),
                'bank_sha256': run.sha(directory / 'train.jsonl'), 'fitting_contract': registration['fitting_contract'],
                'unique_physical_generations': 960, 'raw_coverage': coverage}
            run.expansion.write(directory / 'complete.json', complete)
            stack.enter_context(patch.object(run.expansion, 'registration', return_value=registration))
            stack.enter_context(patch.object(run.expansion, 'inputs', return_value=(public, controls, chosen, {'selection_sha256': 'selection'})))
            stack.enter_context(patch.object(run.expansion, 'prerequisite_receipts', side_effect=AssertionError('Do not read DEV')))
            actual, selection, receipt = run.complete_expansion(directory, data, mentor)
            self.assertEqual((len(actual), len(selection), receipt['physical_generations']), (960, 192, 960))
            # Even an unsealed new transport log invalidates complete accounting.
            (directory / 'transport-errors.jsonl').write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'sealed'):
                run.complete_expansion(directory, data, mentor)
            (directory / 'transport-errors.jsonl').unlink()
            with (directory / 'raw-generations.jsonl').open('a') as f:
                f.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'Changed registered'):
                run.complete_expansion(directory, data, mentor)


if __name__ == '__main__':
    unittest.main()
