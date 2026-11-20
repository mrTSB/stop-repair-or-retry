import unittest

from review_fields import fields,parse


class CompletedFields(unittest.TestCase):
    def test_retains_completed_answer_before_truncated_solution(self):
        obj,valid,complete=parse('{"answer":"B","p_correct":0.1,"action":"repair","solution":"unfinished')
        self.assertEqual(obj,{'answer':'B','p_correct':.1,'action':'repair'})
        self.assertTrue(valid)
        self.assertFalse(complete)

    def test_does_not_invent_missing_decision_fields(self):
        obj,valid,complete=parse('{"answer":"42","solution":"unfinished')
        self.assertEqual(obj,{'answer':'42'})
        self.assertFalse(valid)
        self.assertFalse(complete)

    def test_does_not_read_keys_inside_a_truncated_quoted_solution(self):
        obj,complete=fields('{"solution":"example \\"answer\\": \\"A\\"')
        self.assertEqual(obj,{})
        self.assertFalse(complete)

    def test_rejects_invalid_confidence_and_action(self):
        for p,a in ((True,'stop'),(1.1,'repair'),(.7,'unknown')):
            import json
            self.assertFalse(parse(json.dumps({'answer':'C','p_correct':p,'action':a}))[1])

    def test_complete_json_with_latex_escaping(self):
        obj,valid,complete=parse('{"answer":"A","p_correct":0.9,"action":"stop","solution":"\\lambda = 2"}')
        self.assertEqual(obj['solution'],'\\lambda = 2')
        self.assertTrue(valid)
        self.assertTrue(complete)


if __name__=='__main__':
    unittest.main()
