"""Exact small-tree checks for latent-state training and held-out policy inputs."""
import copy
import itertools
import math
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/pomdp-controller'))
import latent_states as l


def random_model():
    rng=random.Random(238)
    def distribution(n):
        a=[rng.random()+.1 for _ in range(n)];return [v/sum(a) for v in a]
    base={'states':list('abcd'),'observations':l.p.OBSERVATIONS,'terminal_reward':[1,0,0,0],
          'prior':distribution(4),'initial_observation':[distribution(8) for _ in range(4)]}
    contexts={}
    for context in l.c.CONTEXTS:
        contexts[context]=dict(base,transition={a:[distribution(4) for _ in range(4)] for a in l.p.ACTIONS},
            observation={a:[distribution(8) for _ in range(4)] for a in l.p.ACTIONS},cost_tokens={a:[20]*4 for a in l.p.ACTIONS})
    return {'pooled_prior_model':base,'contexts':contexts}


def row(index=0):
    paths=('','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry')
    return {'id':str(index),'split':'train','domain':'math','tree':{path:{'correct':(j+index)%3==0,
        'confidence':['low','medium','high'][j%3],'answer_parse_valid':j%2==0,'input_tokens':10+j,'output_tokens':3} for j,path in enumerate(paths)}}


class LatentTests(unittest.TestCase):
    def test_three_node_four_state_exact_enumeration(self):
        m=random_model();r=row();r['tree']={k:v for k,v in r['tree'].items() if k in ('','repair','retry')}
        r['tree']['']['correct']=False;r['tree']['repair']['correct']=False;r['tree']['retry']['correct']=True
        paths=list(r['tree']);weights=[]
        for states in itertools.product(range(4),repeat=3):
            assign=dict(zip(paths,states))
            if any((assign[q]==0)!=r['tree'][q]['correct'] for q in paths):continue
            s=assign[''];prob=m['pooled_prior_model']['prior'][s]*m['pooled_prior_model']['initial_observation'][s][l.p.observation(r['tree'][''])]
            for action in l.p.ACTIONS:
                sp=assign[action];ctx=m['contexts']['initial']
                prob*=ctx['transition'][action][s][sp]*ctx['observation'][action][sp][l.p.observation(r['tree'][action])]
            weights.append((assign,prob))
        total=sum(w for _,w in weights);post=l.posterior(m,r)
        self.assertAlmostEqual(post['log_likelihood'],math.log(total),places=12)
        for path in paths:
            for s in range(4):self.assertAlmostEqual(post['gamma'][path][s],sum(w for a,w in weights if a[path]==s)/total,places=12)
        for action in l.p.ACTIONS:
            for s,sp in itertools.product(range(4),repeat=2):self.assertAlmostEqual(post['xi'][action][s][sp],sum(w for a,w in weights if a['']==s and a[action]==sp)/total,places=12)

    def test_heldout_correctness_labels_do_not_change_decisions(self):
        m=random_model();r=row();other=copy.deepcopy(r)
        for node in other['tree'].values():node['correct']=not node['correct']
        a=l.c.replay(m,[r],.248)[0];b=l.c.replay(m,[other],.248)[0]
        for key in ('steps','path','final_belief','tokens','controller_tokens'):self.assertEqual(a[key],b[key])

    def test_two_state_matches_calibrated_control_with_reordered_states(self):
        rows=[row(i) for i in range(12)]
        fitted=l.fit(rows,2);reference=l.cv.fit(rows,'context_calibrated')
        for ctx in ['pooled_prior_model',*l.c.CONTEXTS]:
            a=fitted[ctx] if ctx=='pooled_prior_model' else fitted['contexts'][ctx]
            b=reference[ctx] if ctx=='pooled_prior_model' else reference['contexts'][ctx]
            for s in range(2):
                self.assertAlmostEqual(a['prior'][s],b['prior'][1-s],places=12)
                for z in range(8):self.assertAlmostEqual(a['initial_observation'][s][z],b['initial_observation'][1-s][z],places=12)
                for action in l.p.ACTIONS:
                    self.assertAlmostEqual(a['cost_tokens'][action][s],b['cost_tokens'][action][1-s],places=10)
                    for sp in range(2):self.assertAlmostEqual(a['transition'][action][s][sp],b['transition'][action][1-s][1-sp],places=12)
                    for z in range(8):self.assertAlmostEqual(a['observation'][action][s][z],b['observation'][action][1-s][z],places=12)

if __name__=='__main__':unittest.main()
