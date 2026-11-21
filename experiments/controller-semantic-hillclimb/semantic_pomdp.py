"""Bayesian POMDP with paid semantic observations and conditional transition forecasts."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import semantic_data
from examples import text_pair
from network import encode, load, predict
import difficulty_pomdp as base
from contextual_pomdp import evaluate as contextual_evaluate
from context_pomdp import equal, REVIEW_WIDTH, encode as ordinary_review_signal, mentor_comparison, outcome


def softmax(logits, temperature):
    logits = np.asarray(logits, dtype=float) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=-1, keepdims=True)


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -35, 35)))


def review_node(judge):
    return {**judge, 'text':str(judge['verdict'].get('solution','')) + '\nFinal: ' + judge['answer'],
            'confidence':None}


def review_signal(probability, judge, current):
    if not judge['parse_valid'] or not judge['answer']:
        return REVIEW_WIDTH - 1
    pc = probability[1] + probability[3]
    bucket = int(np.searchsorted([.2,.5,.8,.97],pc))
    return (('stop','repair','retry').index(judge['action']) * 5 + bucket) * 2 + int(equal(judge['answer'],current['answer']))


@dataclass
class Observer:
    directory: str
    temperature: float
    calibration: object = None
    device: str = 'cpu'
    _model: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)
    _recipe: dict = field(default_factory=dict, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)
    classes_: object = field(default_factory=lambda:np.arange(4))

    def __getstate__(self):
        return {**vars(self), '_model':None, '_tokenizer':None, '_recipe':{}, '_cache':{}}

    def set_params(self, **kw):
        if set(kw) - {'n_jobs'}:
            raise ValueError('Unknown runtime setting')
        return self

    def load(self):
        if self._model is None:
            torch.set_num_threads(4)
            self._model,self._tokenizer,self._recipe = load(self.directory)
            self._model=self._model.to(self.device)

    def key(self, row, path, node, kind):
        return kind,row['id'],path,node['prompt_sha256'],node['text']

    def entry(self, row, path, tree, judge=None):
        node = tree[path] if judge is None else review_node(judge)
        left,right = text_pair(row,node)
        current = tree if judge is None else {**tree,path:node}
        return {'left':left,'right':right,'numeric':base.visible(row,path,current)}

    def values(self, row, path, tree, judge=None):
        self.load()
        node = tree[path] if judge is None else judge
        key = self.key(row,path,node,'solver' if judge is None else 'review')
        if key not in self._cache:
            tensors = encode(self._tokenizer,[self.entry(row,path,tree,judge)],self._recipe['max_length'])
            outputs = predict(self._model,tensors,device=self.device,batch=1)
            self._cache[key] = (*[v[0] for v in outputs],int(tensors['attention_mask'].sum()))
        return self._cache[key]

    def row_probabilities(self, row, path, tree):
        logits=self.values(row,path,tree)[0]
        if self.calibration is None:
            return softmax(logits,self.temperature)
        features=self.calibration_features(row,path,logits)
        probabilities=np.zeros(4)
        probabilities[self.calibration.classes_.astype(int)]=self.calibration.predict_proba([features])[0]
        return probabilities

    def calibration_features(self,row,path,logits):
        values=np.asarray(logits,dtype=float)/self.temperature
        values-=values.mean()
        return [*values,float(bool(path)),*[float(row['subject']==s) for s in ('math','chemistry','physics','biology')]]

    def transitions(self, row, path, tree, reviewed):
        return sigmoid(self.values(row,path,tree)[2 if reviewed else 1]).reshape(2,4)

    def signal(self, row, path, tree, judge):
        # Every acquired review is encoded, including an invalid review. The
        # planning cost and actual charge both include this pass.
        p = softmax(self.values(row,path,tree,judge)[0],self.temperature)
        return review_signal(p,judge,tree[path])

    def warm(self, rows, trees, judges):
        self.load()
        examples,keys = [],[]
        for full_row in rows:
            row = {**full_row,'answer':''}
            tree = trees[row['id']]
            for path in ('','repair','retry','guided'):
                if path not in tree:
                    continue
                examples.append(self.entry(row,path,tree))
                keys.append(self.key(row,path,tree[path],'solver'))
                judge = judges.get((row['id'],path))
                if judge is not None:
                    examples.append(self.entry(row,path,tree,judge))
                    keys.append(self.key(row,path,judge,'review'))
        if examples:
            tensors = encode(self._tokenizer,examples,self._recipe['max_length'])
            outputs = predict(self._model,tensors,device=self.device,batch=16)
            for i,key in enumerate(keys):
                self._cache[key] = (*[values[i] for values in outputs],int(tensors['attention_mask'][i].sum()))

    def charges(self, row, tree, judges, decisions, include_reviews=True):
        self.load()
        tokens,calls = 0,0
        for decision in decisions:
            path = decision['path']
            tokens += self.values(row,path,tree)[3]; calls += 1
            if decision['queried'] and include_reviews:
                tokens += self.values(row,path,tree,judges[row['id'],path])[3]; calls += 1
        return tokens,calls,calls*self._recipe['max_length'],self._recipe['parameters']/1e9


@dataclass
class Planner(base.Planner):
    context_weight: float = .5
    _current_context: dict = field(default_factory=dict, repr=False)
    _review_context: dict = field(default_factory=dict, repr=False)
    observation_cohorts: dict = field(default_factory=dict)

    def likelihood(self,row,path,tree):
        prior,validity=self.observation_cohorts[row['subject'],int(bool(path))]
        if not tree[path]['answer']:
            return 1-validity
        return self.classifier.row_probabilities(row,path,tree)/prior*validity

    def context(self, row, path, tree, stage, reviewed, weight):
        key = row['id'],path,tree[path]['prompt_sha256'],tree[path]['text'],weight
        cache = self._review_context if reviewed else self._current_context
        if key not in cache:
            d = self.dynamics[row['subject']]
            transition = d.reviewed_transition[stage].copy() if reviewed else d.transition[stage].copy()[None]
            forecast = self.classifier.transitions(row,path,tree,reviewed)
            for action in (0,1):
                for state in range(4):
                    h = state // 2
                    pooled = transition[:,action,state,2*h+1]
                    p = np.clip(weight*forecast[action,state]+(1-weight)*pooled,1e-6,1-1e-6)
                    transition[:,action,state,2*h] = 1-p
                    transition[:,action,state,2*h+1] = p
            cache[key] = transition if reviewed else transition[0]
        return cache[key]


def fit(rows, trees, judges, directory, metadata, outputs, smoothing=32., late=False,
        semantic_review=True, context_weight=.5):
    summary = json.loads((Path(directory).parent/'training-summary.json').read_bytes())
    observer = Observer(str(Path(directory).resolve()),summary['temperature_selected_on_training_oof'])
    recipe = json.loads((Path(directory)/'recipe.json').read_bytes())
    price = recipe['parameters']/1e9*recipe['max_length']/3.09
    lookup = {(r['id'],r['kind'],r['path']):v for r,v in zip(metadata,outputs['state'])}
    by_id={r['id']:r for r in rows}
    calibration_rows=[r for r in metadata if r['kind']=='solver' and r['train'] and r['valid']]
    from sklearn.linear_model import LogisticRegression
    x=[observer.calibration_features(by_id[r['id']],r['path'],lookup[r['id'],'solver',r['path']]) for r in calibration_rows]
    observer.calibration=LogisticRegression(C=1.,max_iter=1000).fit(x,[r['label'] for r in calibration_rows])
    scores=[]
    for row in rows:
        for path in trees[row['id']]:
            p=np.zeros(4)
            features=observer.calibration_features(row,path,lookup[row['id'],'solver',path])
            p[observer.calibration.classes_.astype(int)]=observer.calibration.predict_proba([features])[0]
            scores.append(p)
    scores=np.asarray(scores)
    cohorts={}
    for subject in {r['subject'] for r in rows}:
        for stage in (0,1):
            count=np.ones(4);valid_count=np.ones(4)
            for r in metadata:
                if (r['kind']=='solver' and r['train'] and int(bool(r['path']))==stage
                        and by_id[r['id']]['subject']==subject):
                    count[r['label']]+=1
                    valid_count[r['label']]+=int(r['valid'])
            validity=np.clip(valid_count/count,1e-5,1-1e-5)
            validity[[1,3]]=1.
            cohorts[subject,stage]=(valid_count/valid_count.sum(),validity)
    extra = {(r['id'],path):price if path in ('repair','retry','guided') else 0.
             for r in rows for path in trees[r['id']]}
    signals = None
    if semantic_review:
        signals = {}
        for row in rows:
            for path in ('','repair','retry','guided'):
                judge = judges.get((row['id'],path))
                if judge is None:
                    continue
                kind = 'late-review' if path and late else 'original-review'
                probability = softmax(lookup[row['id'],kind,path],observer.temperature)
                signals[row['id'],path] = review_signal(probability,judge,trees[row['id']][path])
    initial = base.fit(rows,trees,judges,leaf=0,smoothing=smoothing,late_feedback=late,stage_review=True,
                       classifier_override=observer,scores_override=scores,extra_costs=extra,
                       review_observer=observer if semantic_review else None,review_signals=signals)
    if semantic_review:
        for dynamics in initial.dynamics.values():
            dynamics.review_costs += price
    receipt = {**initial.fit_receipt,'semantic_observer':True,'semantic_review':semantic_review,
               'parameters':recipe['parameters'],'max_length':recipe['max_length'],
               'context_weight':context_weight,'weights':recipe['weights'],
               'grouped_folds':3,'current_transitions':'semantic conditional fix/damage forecasts',
               'future_transitions':'pooled learned dynamics when future response context is unknown'}
    return Planner(initial.classifier,initial.dynamics,receipt,
                   review_observer=initial.review_observer,context_weight=context_weight,
                   observation_cohorts=cohorts)


def evaluate(row, tree, judges, model, settings, family):
    if family not in ('A','C'):
        raise ValueError('B remains a separate LLM controller')
    if settings.get('continuous_filter',False):
        result=continuous_evaluate(row,tree,judges,model,settings,family)
    else:
        result = contextual_evaluate(row,tree,judges,model,settings,family)
    tokens,calls,slots,parameters = model.classifier.charges(row,tree,judges,result['decisions'],
                                                          include_reviews=model.review_observer is not None)
    extra = parameters*slots/3.09
    result.update(verifier_tokens=tokens,verifier_calls=calls,verifier_token_slots=slots,
                  verifier_parameters_billions=parameters)
    result['compute'] += extra
    result['total_parameter_compute'] += extra
    return result


def continuous_evaluate(row,tree,judges,model,settings,family):
    d=model.dynamics[row['subject']]
    penalty=settings['penalty']*(settings.get('science_weight',1.) if row['domain']=='science' else 1.)
    threshold=settings.get('threshold',0.);weight=settings.get('context_weight',model.context_weight)
    reuse=settings.get('reuse_review',True)
    belief=d.update(d.prior,model.likelihood(row,'',tree))
    path='';visited=[''];acquired=[];decisions=[]
    for remaining in (2,1):
        stage=2-remaining;condition=model.observed(row,path,tree)
        transition=model.context(row,path,tree,stage,False,weight)
        values=d.values(belief,remaining,penalty,transition)
        information=None;query=False;reviewed_transitions=None
        if family=='C':
            reviewed_transitions=model.context(row,path,tree,stage,True,weight)
            expectation=0.
            for signal in range(REVIEW_WIDTH):
                mass=belief*d.likelihood(stage,condition)[:,signal]
                expectation+=d.values(mass,remaining,penalty,reviewed_transitions[signal],True,reuse).max()
            expectation-=penalty*(belief@d.review_costs[stage])/1000
            information=float(expectation-values.max());query=bool(information>threshold)
        judge=None
        if query:
            if (row['id'],path) not in judges:raise base.MissingObservation('judge',path)
            judge=judges[row['id'],path];acquired.append(path)
            signal=model.review_signal(row,path,tree,judge)
            belief=d.update(belief,d.likelihood(stage,condition)[:,signal])
            transition=reviewed_transitions[signal]
            values=d.values(belief,remaining,penalty,transition,True,reuse)
        action=('stop','repair','retry')[int(values.argmax())]
        decisions.append({'path':path,'belief':belief.tolist(),'action':action,'queried':query,
                          'information_value':information,'continuous_filter':True})
        if action=='stop':break
        edge='guided' if query and not path and action=='repair' else action
        if query and path and action=='repair' and settings.get('late_feedback',False):edge='reviewed'
        path=f'{path}/{edge}' if path else edge
        if path not in tree:raise base.MissingObservation('solver',path)
        visited.append(path)
        if remaining==1:break
        likelihood=model.likelihood(row,path,tree)
        observed=model.observed(row,path,tree)
        if query and reuse:
            likelihood=likelihood*d.mentor[observed,:,mentor_comparison(judge,tree[path])]
        belief=d.update(belief@transition[('repair','retry').index(action)],likelihood)
    return outcome(row,tree,judges,path,visited,acquired,decisions)
