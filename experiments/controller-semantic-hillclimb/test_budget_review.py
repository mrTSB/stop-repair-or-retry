import unittest

from budget_review import review
from long_review_pilot import review as long_review


class BoundedReview(unittest.TestCase):
    review = staticmethod(review)
    row={'id':'sample','domain':'science','question':'Choose the correct option.','answer':'B'}
    current={'text':'Final: A'}

    def response(self,text,cap):
        return {'model':'qwen3.5:35b','done':True,'response':text,'prompt_eval_count':100,
                'eval_count':cap,'done_reason':'stop'}

    def test_completed_wrong_answer_is_not_regenerated(self):
        payloads=[]
        def request(path,payload):
            payloads.append(payload)
            return self.response('Checked it. </think>{"answer":"A","p_correct":0.8,"action":"stop","check":"checked"}',30)
        result=self.review(request,self.row,self.current)
        self.assertEqual(len(payloads),1)
        self.assertEqual(result['answer'],'A')
        self.assertEqual(result['physical_api_calls'],1)
        self.assertEqual(result['input_tokens'],100)

    def test_both_scheduled_phases_are_charged(self):
        payloads=[]
        def request(path,payload):
            payloads.append(payload)
            return self.response('Checking the assumptions.' if len(payloads)==1 else
                                 'B", "p_correct":0.1,"action":"repair","check":"corrected"}',
                                 256 if len(payloads)==1 else 40)
        result=self.review(request,self.row,self.current)
        self.assertEqual(result['physical_api_calls'],2)
        self.assertEqual((result['input_tokens'],result['output_tokens']),(200,296))
        self.assertEqual(result['answer'],'B')
        self.assertTrue(result['json_complete'])
        self.assertTrue(payloads[1]['prompt'].endswith('{"answer": "'))

    def test_reference_answer_never_enters_either_prompt(self):
        all_payloads=[]
        for gold in ('B','C'):
            payloads=[]
            def request(path,payload):
                payloads.append(payload)
                return self.response('Still checking.' if len(payloads)==1 else
                                     'B", "p_correct":0.1,"action":"repair","check":"checked"}',20)
            self.review(request,{**self.row,'answer':gold},self.current)
            all_payloads.append(payloads)
        self.assertEqual(all_payloads[0],all_payloads[1])

class LongReview(BoundedReview):
    review = staticmethod(long_review)


if __name__=='__main__':unittest.main()
