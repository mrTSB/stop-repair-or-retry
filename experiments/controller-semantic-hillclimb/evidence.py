"""Reference-free textbook retrieval for richer correctness observations."""

import hashlib
import json
from pathlib import Path
import pickle

import semantic_data
from extended_acquire import SCIENCE_SUFFIX

DEFAULT=semantic_data.ROOT/'.cache/semantic-textbooks'


class Retriever:
    def __init__(self,directory=DEFAULT):
        self.directory=Path(directory)
        self.passages=[json.loads(s) for s in (self.directory/'passages.jsonl').read_text().splitlines()]
        self.sha256=hashlib.sha256((self.directory/'passages.jsonl').read_bytes()).hexdigest()
        target=self.directory/'tfidf.pkl'
        if target.exists():
            cached=pickle.loads(target.read_bytes())
            if cached['sha256']==self.sha256:
                self.vectorizer,self.matrix=cached['vectorizer'],cached['matrix'];return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=160000,
            sublinear_tf=True,strip_accents='unicode',dtype='float32')
        self.matrix=self.vectorizer.fit_transform([r['title']+' '+r['text'] for r in self.passages])
        target.write_bytes(pickle.dumps({'sha256':self.sha256,'vectorizer':self.vectorizer,'matrix':self.matrix},protocol=5))

    def retrieve(self,row):
        if row['domain']!='science':return []
        # The query excludes proposed answers and references. All candidates for
        # a question therefore receive exactly the same factual background.
        query=row['question'].removesuffix(SCIENCE_SUFFIX).partition('\n\nOptions:\n')[0]
        scores=(self.matrix@self.vectorizer.transform([query]).T).toarray().ravel()
        order=scores.argsort()[::-1]
        chosen=[];modules=set()
        for i in order:
            if scores[i]<.06:break
            source=self.passages[int(i)]
            if source['subject']!=row['subject']:continue
            module=source['source']
            if module in modules:continue
            modules.add(module)
            words=source['text'].split()
            chosen.append({**source,'text':' '.join(words[:85]),'retrieval_score':float(scores[i]),
                           'excerpt_modified':len(words)>85})
            if len(chosen)==2:break
        return chosen

    def context(self,row):
        entries=self.retrieve(row)
        return '\n\n'.join(r['title']+': '+r['text'] for r in entries),entries


def left_text(row,background):
    text='QUESTION:\n'+row['question'].removesuffix(SCIENCE_SUFFIX)
    if background:text+='\n\nBACKGROUND EXPOSITION (may be incomplete):\n'+background
    return text
