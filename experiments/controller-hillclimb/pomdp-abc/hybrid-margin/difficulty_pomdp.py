"""Four-state POMDP: hidden persistent difficulty and current-answer correctness."""

import ast
from dataclasses import dataclass, field
import hashlib
import re

import numpy as np

from margin_data import SEED
from context_pomdp import encode, equal, features as previous_features, mentor_comparison, outcome, REVIEW_WIDTH
from scoring import correct

N_OBS = 11
REWARD = np.array([0., 1., 0., 1.])
PLAIN = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
FIXED = ('', 'repair', 'retry', 'repair/repair', 'retry/retry')


def arithmetic(expression):
    """Evaluate only a small numeric expression, with bounded exponentiation."""
    if len(expression) > 80:
        raise ValueError('Long expression')
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            elif isinstance(node.op, ast.Mult): value = left * right
            elif isinstance(node.op, ast.Div): value = left / right
            elif isinstance(node.op, ast.Pow) and abs(right) <= 8: value = left ** right
            else: raise ValueError('Unsupported operation')
        else:
            raise ValueError('Non-numeric expression')
        if not np.isfinite(value) or abs(value) > 1e15:
            raise ValueError('Unbounded expression')
        return value
    return visit(ast.parse(expression, mode='eval'))


def visible(row, path, tree):
    # previous_features reads the current node and its ancestors only.
    values = previous_features(row, path, tree)
    if path.endswith('/reviewed'):
        values[-6] = True
        values[-2] = True
    text = tree[path]['text'].replace('×', '*').replace('÷', '/').replace('−', '-')
    equations, errors, matches = 0, 0, 0
    for match in re.finditer(r'(?<![\w.])([\d.()+*/ ^-]{3,70})\s*=\s*(-?\d+(?:\.\d+)?)', text):
        expression = match[1].strip().replace('^', '**')
        if not any(op in expression for op in ('+', '-', '*', '/')):
            continue
        try:
            value, stated = arithmetic(expression), float(match[2])
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
            continue
        equations += 1
        errors += not np.isclose(value, stated, rtol=1e-5, atol=1e-5)
        try:
            matches += np.isclose(stated, float(tree[path]['answer']), rtol=1e-5, atol=1e-5)
        except (ValueError, TypeError):
            pass
    question = row['question'].lower()
    topics = ('electric', 'magnetic', 'quantum', 'energy', 'force', 'wave', 'thermal',
              'acid', 'equilibrium', 'organic', 'molecule', 'reaction', 'electron',
              'which of the following', 'incorrect', 'except', 'not true', 'all of')
    return values + [min(equations, 5)/5, min(errors, 3)/3, min(matches, 3)/3] + [int(t in question) for t in topics]


def normalize(values):
    values = np.asarray(values, dtype=float)
    mass = values.sum()
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError('Impossible POMDP observation')
    return values / mass


def observation(probability, valid):
    if not valid:
        return 0
    pc = probability[1] + probability[3]
    hard = probability[2] + probability[3] >= .5
    return 1 + int(np.searchsorted([.2, .5, .8, .97], pc)) * 2 + int(hard)


def forest(leaf, seed):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=160, min_samples_leaf=leaf, max_features=.7,
                                max_depth=14, criterion='log_loss', n_jobs=-1, random_state=seed)


def probabilities(model, x):
    out = np.zeros((len(x), 4))
    out[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    # Stabilise finite observation bins against thread-order roundoff in forest sums.
    return np.round(out,12)


class MissingObservation(Exception):
    def __init__(self, kind, path):
        self.kind, self.path = kind, path
        super().__init__(kind + ':' + path)


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
            reviewed=np.zeros((REVIEW_WIDTH,3,4));reviewed[:,0]=REWARD
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
                    masses=expanded.transpose(0,2,1).reshape(-1,4)
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
                p = probabilities(self.classifier, np.array([visible(row, path, tree)]))[0]
            self._observed[key] = observation(p, bool(tree[path]['answer']))
        return self._observed[key]


def fit(rows, trees, judges, leaf=12, smoothing=32., late_feedback=False, stage_review=False,
        classifier_override=None,scores_override=None,extra_costs=None,
        review_observer=None,review_signals=None):
    hard = {r['id']: int(np.mean([correct(trees[r['id']][p], r['answer']) for p in FIXED]) < .6) for r in rows}
    examples = [(r, p) for r in rows for p in trees[r['id']]]
    x = np.array([visible(r, p, trees[r['id']]) for r, p in examples])
    y = np.array([2*hard[r['id']] + int(correct(trees[r['id']][p], r['answer'])) for r, p in examples])
    folds = np.array([int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16)%5 for r,p in examples])
    if classifier_override is not None:
        scores=np.asarray(scores_override)
        if scores.shape!=(len(y),4) or not np.isfinite(scores).all():raise ValueError('Invalid out-of-question neural observations')
        classifier=classifier_override
    else:
        scores = np.zeros((len(y), 4))
        for fold in range(5):
            current = forest(leaf, SEED+fold).fit(x[folds!=fold], y[folds!=fold])
            scores[folds==fold] = probabilities(current, x[folds==fold])
        classifier = forest(leaf, SEED+10).fit(x, y)
    extra_costs={} if extra_costs is None else extra_costs
    observed = {(r['id'],p):observation(v,bool(trees[r['id']][p]['answer'])) for (r,p),v in zip(examples,scores)}
    dynamics = {}
    for subject in sorted({r['subject'] for r in rows}):
        group = [r for r in rows if r['subject']==subject]
        prior = np.ones(4)
        initial = np.full((4,N_OBS),.5)
        trans = np.zeros((2,2,4,4))
        reviewed = np.zeros((2,REVIEW_WIDTH,2,4,4))
        for h in (0,1):
            trans[:,:,2*h:2*h+2,2*h:2*h+2]=1
            reviewed[:,:,:,2*h:2*h+2,2*h:2*h+2]=1
        emission=np.full((2,2,4,N_OBS),.5)
        re_emission=emission.copy()
        price_sum=np.zeros((2,2,4));price_n=np.zeros_like(price_sum)
        re_price_sum=np.zeros_like(price_sum);re_price_n=np.zeros_like(price_sum)
        review=np.zeros((2,N_OBS,4,REVIEW_WIDTH)) if stage_review else np.zeros((N_OBS,4,REVIEW_WIDTH))
        pool_review=np.ones((2,4,REVIEW_WIDTH)) if stage_review else np.ones((4,REVIEW_WIDTH))
        judge_sum=np.zeros((2,4));judge_n=np.zeros_like(judge_sum)
        mentor=np.zeros((N_OBS,4,3));pool_mentor=np.ones((4,3))
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
                                      'leaf':leaf,'smoothing':smoothing,'grouped_folds':5,
                                      'late_feedback':late_feedback,'stage_review':stage_review,
                                      'hidden_states':['easy/wrong','easy/correct','hard/wrong','hard/correct']},
                   review_observer=review_observer)


def evaluate(row,tree,judges,model,settings,method):
    if method not in ('A','C','always','never'):
        raise ValueError('This planner implements A/C only; B remains an independent LLM policy')
    d=model.dynamics[row['subject']]
    penalty=settings['penalty']*(settings.get('science_weight',1.) if row['domain']=='science' else 1.)
    reuse=settings.get('reuse_review',True)
    future_reviews=method=='C' and settings.get('plan_reviews',False)
    threshold=settings.get('threshold',0.)
    observed=model.observed(row,'',tree)
    belief=d.update(d.prior,d.initial[:,observed])
    path='';visited=[''];acquired=[];decisions=[]
    for remaining in (2,1):
        stage=2-remaining;condition=model.observed(row,path,tree)
        values=d.values(belief,remaining,penalty,future_reviews=future_reviews,threshold=threshold)
        information=None;query=method=='always'
        if method=='C':
            information=d.information_value(belief,remaining,penalty,condition,reuse,
                                            future_reviews,threshold)-values.max()
            query=bool(information>threshold)
        transition=d.transition[stage]
        judge=None
        if query:
            if (row['id'],path) not in judges:raise MissingObservation('judge',path)
            judge=judges[row['id'],path];acquired.append(path)
            signal=model.review_signal(row,path,tree,judge) if hasattr(model,'review_signal') else encode(judge,tree[path])
            belief=d.update(belief,d.likelihood(stage,condition)[:,signal])
            transition=d.reviewed_transition[stage,signal]
            values=d.values(belief,remaining,penalty,transition,True,reuse,future_reviews,threshold)
        action=('stop','repair','retry')[int(values.argmax())]
        decisions.append({'path':path,'belief':belief.tolist(),'action':action,'queried':query,
                          'information_value':information})
        if action=='stop':break
        edge='guided' if query and not path and action=='repair' else action
        if query and path and action=='repair' and settings.get('late_feedback',False):
            edge='reviewed'
        path=f'{path}/{edge}' if path else edge
        if remaining==1 and hasattr(getattr(model,'classifier',None),'row_probabilities'):
            if path not in tree:raise MissingObservation('solver',path)
            visited.append(path)
            break  # Submission is forced; no observation-model pass can change the decision.
        observed=model.observed(row,path,tree);visited.append(path)
        comparison=mentor_comparison(judge,tree[path]) if query and reuse else None
        belief=d.next_belief(belief,transition,('repair','retry').index(action),observed,stage,query,comparison)
    return outcome(row,tree,judges,path,visited,acquired,decisions)
