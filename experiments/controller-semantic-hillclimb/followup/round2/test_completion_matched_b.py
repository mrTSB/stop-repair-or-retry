import unittest

import completion_matched_b as b


def node(answer,tokens=10):
    return {'answer':answer,'text':answer,'input_tokens':tokens,'output_tokens':1}


class PaidTree(dict):
    def __init__(self,*args,allowed,**kwargs): super().__init__(*args,**kwargs);self.allowed=allowed;self.seen=[]
    def __getitem__(self,key):
        if key not in self.allowed: raise AssertionError('Read an unavailable/future response: '+key)
        self.seen.append(key);return super().__getitem__(key)


class CompletionBTests(unittest.TestCase):
    def exercise(self,action,root='B',child='B',style='ordinary',when='always',root_stop='action',answer='A'):
        row={'id':'q','domain':'science','subject':'physics','answer':answer}
        tree={p:node('A' if p.endswith('complete') or p.endswith('reuse') else child,20+i)
              for i,p in enumerate(['','repair','retry','guided','guided/reuse','complete','repair/complete','retry/complete'])}
        tree['']['answer']=root
        calls=[{'input_tokens':30,'output_tokens':4},{'input_tokens':40,'output_tokens':5}]
        judge={**node('A'),'input_tokens':70,'output_tokens':9,'action':action,'p_correct':.9,
               'calls':calls,'physical_api_calls':2}
        settings={'root_stop':root_stop,'first_repair':style,'ordinary_completion':when}
        return b.policy(row,tree,{('q',''):judge},settings),tree

    def test_fixed24_and_paid_review_on_valid_stop_and_invalid_complete(self):
        self.assertEqual(len(b.SETTINGS),24)
        result,_=self.exercise('stop',root='A')
        self.assertEqual(result['path'],'');self.assertEqual(result['judge_tokens'],79)
        result,tree=self.exercise('stop',root='')
        self.assertEqual(result['path'],'complete');self.assertEqual(result['solver_calls'],2)
        self.assertEqual(result['judge_physical_calls'],2);self.assertEqual(result['verifier_calls'],0)
        self.assertAlmostEqual(result['compute'],(4*sum(tree[p]['input_tokens']+1 for p in ('','complete'))+3*79)/3.09)

    def test_ordinary_repair_and_retry_complete_consume_last_slot(self):
        for action in ('repair','retry'):
            result,_=self.exercise(action)
            self.assertEqual(result['path'],action+'/complete');self.assertEqual(result['solver_calls'],3)
            stopped,_=self.exercise(action,when='invalid')
            self.assertEqual(stopped['path'],action)
            repaired,_=self.exercise(action,child='',when='invalid')
            self.assertEqual(repaired['path'],action+'/complete')

    def test_guided_never_complete_or_neural_and_reference_cannot_choose(self):
        first,_=self.exercise('repair',style='guided')
        second,_=self.exercise('repair',style='guided',answer='D')
        self.assertEqual(first['path'],'guided/reuse');self.assertEqual(first['completion_calls'],0)
        self.assertEqual(first['decisions'],second['decisions']);self.assertEqual(first['compute'],second['compute'])
        self.assertNotEqual(first['correct'],second['correct'])
        stopped,_=self.exercise('repair',style='guided',child='A')
        self.assertEqual(stopped['path'],'guided')

    def test_existing_six_root_rules_preserved(self):
        for rule in b.original.ROOT_STOPS:
            result,_=self.exercise('retry',root='A',root_stop=rule)
            expect='retry/complete' if rule=='action' or isinstance(rule,float) and rule>.9 else ''
            self.assertEqual(result['path'],expect)

    def test_missing_selected_completion_does_not_read_alternative_or_reference(self):
        row={'id':'q','domain':'science','subject':'physics','answer':'private'}
        tree=PaidTree({'':node('B'),'repair':node('B'),'retry':node('A')},allowed={'','repair'})
        judge={**node('A'),'action':'repair','p_correct':.5,'physical_api_calls':1,
               'calls':[{'input_tokens':10,'output_tokens':1}]}
        with self.assertRaises(b.original.MissingObservation) as error:
            b.policy(row,tree,{('q',''):judge},b.SETTINGS[0])
        self.assertEqual(error.exception.path,'repair/complete')
        self.assertNotIn('retry',tree.seen)

    def test_original_stronger_B_retained_and_equal_accuracy_uses_lower_compute(self):
        candidates=[{'settings':s,'dev':{'n':256,'accuracy':.6,'mean_compute':3000}} for s in b.SETTINGS]
        old={'dev':{'n':256,'accuracy':.7,'mean_compute':4000}}
        result=b.compare(candidates,[old]);self.assertIs(result['B'],old)
        candidates[0]['dev'].update(accuracy=.7,mean_compute=3999)
        self.assertIs(b.compare(candidates,[old])['B'],candidates[0])
        for candidate in candidates: candidate['settings']=dict(sorted(candidate['settings'].items()))
        self.assertIs(b.compare(candidates,[old])['B'],candidates[0])
        with self.assertRaises(ValueError):b.compare(candidates[:-1],[old])


if __name__=='__main__':unittest.main()
