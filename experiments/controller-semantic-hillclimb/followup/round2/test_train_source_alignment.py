"""Synthetic checks for source mapping, parser coverage and selective decoding."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import audit_train_source_alignment as audit


class SourceAlignmentTests(unittest.TestCase):
    def science(self):
        source = {'uuid': 'q', 'question': 'Question?', 'options': ['first', 'second', 'third', 'fourth'],
                  'answer_letter': 'B', 'answer': 'second', 'field': 'Chemistry', 'discipline': 'Science'}
        row = {'id': 'supergpqa-q', 'domain': 'science', 'subject': 'chemistry', 'answer': 'B', 'source': 'm-a-p/SuperGPQA@revision',
               'question': 'Question?\n\nOptions:\nA. first\nB. second\nC. third\nD. fourth' + audit.old.SCIENCE_SUFFIX}
        return row, source

    def test_exact_source_mapping_and_all_candidate_parser_inputs(self):
        row, source = self.science()
        self.assertEqual(audit.check_science(row, source, 'revision'), ([], [], 4))
        for changes, expected in [({'answer': 'A'}, 'science_reference_letter_differs'),
                                   ({'question': row['question'].replace('B. second', 'B. changed')}, 'science_question_or_option_order'),
                                   ({'domain': 'math'}, 'science_domain_subject')]:
            errors, _, _ = audit.check_science({**row, **changes}, source, 'revision')
            self.assertTrue(any(expected in error for error in errors))
        changed = {**source, 'answer': 'third'}
        self.assertIn('official_answer_text_and_letter_disagree', audit.check_science(row, changed, 'revision')[0])

    def test_normalized_duplicate_warning_and_nested_option_parser_issue(self):
        row, source = self.science()
        source['options'][3] = ' FIRST '
        row['question'] = row['question'].replace('D. fourth', 'D. FIRST')
        self.assertIn('choices_duplicate_after_unicode_case_whitespace_normalization', audit.check_science(row, source, 'revision')[1])
        row, source = self.science()
        source['options'][0] = 'first\nB. nested label'
        row['question'] = row['question'].replace('A. first\n', 'A. first\nB. nested label\n')
        errors, _, _ = audit.check_science(row, source, 'revision')
        self.assertTrue(any('candidate' in error or 'observer' in error for error in errors))

    def test_math_reference_preserves_official_final_value(self):
        row = {'domain': 'math', 'subject': 'math', 'question': 'Question?', 'answer': '3.5'}
        source = {'question': 'Question?', 'answer': 'Worked solution\n#### 3.5'}
        self.assertEqual(audit.check_math(row, source), [])
        parsed = audit.old.parse_answer('math', 'Final: 3.5')
        self.assertTrue(audit.correct({'answer': parsed}, row['answer']))
        self.assertIn('math_reference_differs_from_official_final_value', audit.check_math({**row, 'answer': '35'}, source))

    def test_source_reader_does_not_decode_unselected_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.jsonl'
            path.write_text('{"uuid":"protected", INVALID_JSON\n' + json.dumps({'uuid': 'wanted', 'question': 'selected'}) + '\n')
            self.assertEqual(audit.science_rows(path, {'supergpqa-wanted'}), {'supergpqa-wanted': {'uuid': 'wanted', 'question': 'selected'}})
            with self.assertRaises(ValueError):
                audit.science_rows(path, {'supergpqa-missing'})


if __name__ == '__main__':
    unittest.main()
