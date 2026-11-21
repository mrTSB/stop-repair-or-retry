"""Two-state correctness POMDP with exact two-call optional-review Bellman backups."""

from dataclasses import dataclass,field
import hashlib
import numpy as np
from difficulty_pomdp import normalize,visible,MissingObservation
from context_pomdp import encode,mentor_comparison,outcome,REVIEW_WIDTH
from scoring import correct

N_OBS=6
REWARD=np.array([0.,1.])


def observation(probability,valid):
    if not valid:return 0
    return 1+int(np.searchsorted([.2,.5,.8,.97],probability[1]))


@dataclass
class Dynamics:
    prior: np.ndarray
    initial: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    costs: np.ndarray
    review: np.ndarray
    reviewed_transition: np.ndarray
    reviewed_emission: np.ndarray
    reviewed_costs: np.ndarray
    review_costs: np.ndarray
    mentor: np.ndarray
    _values: dict = field(default_factory=dict, repr=False)
    _information: dict = field(default_factory=dict, repr=False)
    _alphas: dict = field(default_factory=dict, repr=False)

    def update(self, belief, likelihood):
        return normalize(belief * likelihood)

    def next_belief(self, belief, transition, action, observed, stage, reviewed, comparison=None):
        likelihood = (self.reviewed_emission if reviewed else self.emission)[stage, action, :, observed]
        if comparison is not None:
            likelihood = likelihood * self.mentor[observed, :, comparison]
        return self.update(belief @ transition[action], likelihood)

    def likelihood(self, stage, condition):
        return self.review[stage, condition] if self.review.ndim == 4 else self.review[condition]

    def continuation(self, masses, conditions, penalty, future_reviews, threshold):
        """One-call values, including the option to buy a review at the next checkpoint."""
        if penalty not in self._alphas:
            regular=np.array([REWARD, *[self.transition[1,a]@REWARD-
                             penalty*self.costs[1,a]/1000 for a in (0,1)]])
            reviewed=np.zeros((REVIEW_WIDTH,3,2));reviewed[:,0]=REWARD
            for signal in range(REVIEW_WIDTH):
                for a in (0,1):
                    reviewed[signal,a+1]=(self.reviewed_transition[1,signal,a]@REWARD-
                                           penalty*self.reviewed_costs[1,a]/1000)
            self._alphas[penalty]=regular,reviewed
        regular,reviewed=self._alphas[penalty]
        value=(masses@regular.T).max(axis=1)
        if not future_reviews:
            return value
        likelihood=self.likelihood(1,np.asarray(conditions))
        joint=masses[:,None,:]*likelihood.transpose(0,2,1)
        after=np.einsum('nzs,zas->nza',joint,reviewed).max(axis=2).sum(axis=1)
        after-=penalty*(masses@self.review_costs[1])/1000
        after-=threshold*masses.sum(axis=1)
        return np.maximum(value,after)

    def values(self, belief, remaining, penalty, transition=None, reviewed=False, reuse=False,
               future_reviews=False, threshold=0.):
        """Exact Bellman backup over the finite observation alphabet and two-call horizon."""
        stage = 2-remaining
        t = self.transition[stage] if transition is None else transition
        key = belief.tobytes(), remaining, penalty, t.tobytes(), reviewed, reuse, future_reviews, threshold
        if key in self._values:
            return self._values[key]
        costs = (self.reviewed_costs if reviewed else self.costs)[stage]
        values = np.array([belief @ REWARD, 0., 0.])
        for action in (0, 1):
            predicted = belief @ t[action]
            if remaining == 1:
                future = predicted @ REWARD
            else:
                emission = (self.reviewed_emission if reviewed else self.emission)[stage, action]
                joint = predicted[:, None] * emission
                if reviewed and reuse:
                    expanded = joint.T[:, :, None] * self.mentor
                    masses=expanded.transpose(0,2,1).reshape(-1,2)
                    conditions=np.repeat(np.arange(N_OBS),3)
                else:
                    masses=joint.T;conditions=np.arange(N_OBS)
                future=self.continuation(masses,conditions,penalty,future_reviews,threshold).sum()
            values[action+1] = future - penalty*(belief @ costs[action])/1000
        self._values[key] = values
        return values

    def information_value(self, belief, remaining, penalty, condition, reuse,
                          future_reviews=False, threshold=0.):
        key = belief.tobytes(), remaining, penalty, condition, reuse, future_reviews, threshold
        if key in self._information:
            return self._information[key]
        total = 0.
        for signal in range(REVIEW_WIDTH):
            # Homogeneity permits an unnormalised posterior and includes its probability.
            mass = belief * self.likelihood(2-remaining,condition)[:,signal]
            total += self.values(mass, remaining, penalty,
                                 self.reviewed_transition[2-remaining, signal], True, reuse,
                                 future_reviews,threshold).max()
        result = float(total - penalty*(belief @ self.review_costs[2-remaining])/1000)
        self._information[key] = result
        return result


@dataclass
class Planner:
    classifier: object
    dynamics: dict
    fit_receipt: dict
    _observed: dict = field(default_factory=dict, repr=False)
    review_observer: object = None

    def review_signal(self,row,path,tree,judge):
        return encode(judge,tree[path]) if self.review_observer is None else self.review_observer.signal(row,path,tree,judge)

    def observed(self, row, path, tree):
        if path not in tree:
            raise MissingObservation('solver', path)
        key = row['id'], path, tree[path]['prompt_sha256'], tree[path]['text']
        if key not in self._observed:
            if hasattr(self.classifier,'row_probabilities'):
                p=self.classifier.row_probabilities(row,path,tree)
            else:
                raise ValueError('A semantic observation model is required')
            self._observed[key] = observation(p, bool(tree[path]['answer']))
        return self._observed[key]


def fit(rows, trees, judges, leaf=12, smoothing=32., late_feedback=False, stage_review=False,
        classifier_override=None,scores_override=None,extra_costs=None,
        review_observer=None,review_signals=None):
    hard = {r['id']:0 for r in rows}
    examples = [(r, p) for r in rows for p in trees[r['id']]]
    x = np.array([visible(r, p, trees[r['id']]) for r, p in examples])
    y = np.array([2*hard[r['id']] + int(correct(trees[r['id']][p], r['answer'])) for r, p in examples])
    folds = np.array([int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16)%5 for r,p in examples])
    if classifier_override is not None:
        scores=np.asarray(scores_override)
        if scores.shape!=(len(y),2) or not np.isfinite(scores).all():raise ValueError('Invalid out-of-question neural observations')
        classifier=classifier_override
    else:
        raise ValueError('A supervised binary observation model is required')
    extra_costs={} if extra_costs is None else extra_costs
    observed = {(r['id'],p):observation(v,bool(trees[r['id']][p]['answer'])) for (r,p),v in zip(examples,scores)}
    dynamics = {}
    for subject in sorted({r['subject'] for r in rows}):
        group = [r for r in rows if r['subject']==subject]
        prior = np.ones(2)
        initial = np.full((2,N_OBS),.5)
        trans = np.ones((2,2,2,2))
        reviewed = np.ones((2,REVIEW_WIDTH,2,2,2))
        emission=np.full((2,2,2,N_OBS),.5)
        re_emission=emission.copy()
        price_sum=np.zeros((2,2,2));price_n=np.zeros_like(price_sum)
        re_price_sum=np.zeros_like(price_sum);re_price_n=np.zeros_like(price_sum)
        review=np.zeros((2,N_OBS,2,REVIEW_WIDTH)) if stage_review else np.zeros((N_OBS,2,REVIEW_WIDTH))
        pool_review=np.ones((2,2,REVIEW_WIDTH)) if stage_review else np.ones((2,REVIEW_WIDTH))
        judge_sum=np.zeros((2,2));judge_n=np.zeros_like(judge_sum)
        mentor=np.zeros((N_OBS,2,3));pool_mentor=np.ones((2,3))
        for row in group:
            qid=row['id'];tree=trees[qid]
            def state(path): return 2*hard[qid]+int(correct(tree[path],row['answer']))
            prior[state('')]+=1;initial[state(''),observed[qid,'']]+=1
            for path in ('','repair','retry','guided'):
                if path not in tree:continue
                stage=int(bool(path));s=state(path)
                for a,action in enumerate(('repair','retry')):
                    child=f'{path}/{action}' if path else action
                    if child in tree:
                        nxt=state(child);o=observed[qid,child];node=tree[child]
                        trans[stage,a,s,nxt]+=1;emission[stage,a,nxt,o]+=1
                        price_sum[stage,a,s]+=(node['input_tokens']+node['output_tokens'])*4/3.09+extra_costs.get((qid,child),0.)
                        price_n[stage,a,s]+=1
                if (qid,path) not in judges:continue
                judge=judges[qid,path]
                signal=encode(judge,tree[path]) if review_signals is None else review_signals[qid,path]
                condition=observed[qid,path]
                if stage_review:
                    review[stage,condition,s,signal]+=1;pool_review[stage,s,signal]+=1
                else:
                    review[condition,s,signal]+=1;pool_review[s,signal]+=1
                judge_sum[stage,s]+=(judge['input_tokens']+judge['output_tokens'])*3/3.09;judge_n[stage,s]+=1
                for a,action in enumerate(('repair','retry')):
                    child=('guided' if a==0 else 'retry') if not path else f'{path}/{action}'
                    if late_feedback and path and a==0:
                        child=path+'/reviewed'
                    if child not in tree:continue
                    nxt=state(child);o=observed[qid,child];node=tree[child]
                    reviewed[stage,signal,a,s,nxt]+=1;re_emission[stage,a,nxt,o]+=1
                    re_price_sum[stage,a,s]+=(node['input_tokens']+node['output_tokens'])*4/3.09+extra_costs.get((qid,child),0.)
                    re_price_n[stage,a,s]+=1
                    if stage==0:
                        comparison=mentor_comparison(judge,node)
                        mentor[o,nxt,comparison]+=1;pool_mentor[nxt,comparison]+=1
        trans/=trans.sum(axis=-1,keepdims=True)
        reviewed+=smoothing*trans[:,None]
        reviewed/=reviewed.sum(axis=-1,keepdims=True)
        for o in (initial,emission,re_emission):
            o[...,1::2,0] = 0
            o/=o.sum(axis=-1,keepdims=True)
        pool_review/=pool_review.sum(axis=-1,keepdims=True)
        review+=smoothing*(pool_review[:,None] if stage_review else pool_review[None])
        review/=review.sum(axis=-1,keepdims=True)
        pool_mentor/=pool_mentor.sum(axis=-1,keepdims=True)
        mentor+=smoothing*pool_mentor[None];mentor/=mentor.sum(axis=-1,keepdims=True)
        cost=np.divide(price_sum,price_n,out=np.full_like(price_sum,550.),where=price_n>0)
        rcost=np.divide(re_price_sum,re_price_n,out=np.full_like(re_price_sum,650.),where=re_price_n>0)
        jcost=np.divide(judge_sum,judge_n,out=np.full_like(judge_sum,600.),where=judge_n>0)
        dynamics[subject]=Dynamics(normalize(prior),initial,trans,emission,cost,review,reviewed,re_emission,rcost,jcost,mentor)
    return Planner(classifier,dynamics,{'training_questions':len(rows),'observations':len(examples),
                                      'leaf':leaf,'smoothing':smoothing,'grouped_folds':3,
                                      'late_feedback':late_feedback,'stage_review':stage_review,
                                      'hidden_states':['wrong','correct'],'mentor_estimation':'root reviewed continuations only'},
                   review_observer=review_observer)
