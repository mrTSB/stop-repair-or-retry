"""Append question-independent OCR evidence to the sealed BGE textbook index.

Old passage bytes and vectors remain identical. Only new OCR passages receive
local CPU encoder forwards. The combined lexical index is rebuilt without
reading any benchmark records; query inference retains the original bill.
"""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import dense_textbook_index as old
import dense_textbook_retrieval as retrieval
import ocr_textbook_corpus as corpus

SCHEMA = 'question-independent-incremental-OCR-BGE-index-v1'
ROOT = Path(__file__).resolve().parents[4]
INDEX = ROOT/'runs/controller-semantic-followup/round2/ocr-dense-textbook-index-v1'
read, write, sha = old.read, old.write, old.sha


def sealed(folder):
    done = read(folder/'complete.json')
    if done.get('benchmark_records_read', False): raise ValueError('Question-independent assets required')
    if done['registration_sha256'] != sha(folder/'registration.json'):
        raise ValueError('Changed asset registration')
    for name, digest in done['digests'].items():
        if sha(folder/name) != digest: raise ValueError('Changed sealed asset: '+name)
    return done


def append_vectors(previous, added, count):
    if (previous.shape != (count,384) or added.ndim != 2 or added.shape[1] != 384
            or not np.isfinite(previous).all() or not np.isfinite(added).all()):
        raise ValueError('Aligned finite384-dimensional vectors required')
    result = np.concatenate((previous,added))
    if (not np.array_equal(result[:count],previous)
            or not np.allclose(np.linalg.norm(result,axis=1),1.,atol=1e-5)):
        raise ValueError('Changed old vectors or malformed new vectors')
    return result


def contract(args):
    done, parent_done = sealed(args.corpus), sealed(args.parent)
    manifest = read(args.corpus/'manifest.json')
    parent = read(args.parent/'registration.json')
    parent_corpus = Path(parent['corpus'])
    original = (parent_corpus/'passages.jsonl').read_bytes()
    combined = (args.corpus/'passages.jsonl').read_bytes()
    if (done['schema'] != corpus.SCHEMA or parent_done['schema'] != old.SCHEMA
            or parent['source_sha256'] != sha(old.__file__)
            or sha(args.corpus/'passages.jsonl') != manifest['sha256']
            or sha(parent_corpus/'passages.jsonl') != parent['corpus_sha256']
            or manifest['old_passages_sha256'] != parent['corpus_sha256']
            or not combined.startswith(original) or parent['passages'] != manifest['old_passages']):
        raise ValueError('Require exact old passage prefix and frozen parent encoder')
    rows = [json.loads(line) for line in combined.splitlines()]
    if len(rows) != manifest['passages'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Repeated or missing passage identities')
    if any(not r['id'].startswith('ocr:') for r in rows[parent['passages']:]):
        raise ValueError('Only attributed OCR rows may be appended')
    return {'schema':SCHEMA,'source_sha256':sha(__file__),
        'dependencies':{str(Path(m.__file__).resolve()):sha(m.__file__) for m in (old,retrieval,corpus)},
        'parent':str(args.parent.resolve()),'parent_complete_sha256':sha(args.parent/'complete.json'),
        'corpus':str(args.corpus.resolve()),'corpus_complete_sha256':sha(args.corpus/'complete.json'),
        'corpus_sha256':manifest['sha256'],'old_passages':parent['passages'],'passages':len(rows),
        'OCR_passages':manifest['OCR_passages'],'model':old.MODEL,'revision':old.REVISION,
        'assets':str((args.parent/'assets').resolve()),'batch_size':16,'max_length':256,
        'pooling':'CLS followed by L2 normalization','threads':2,'device':'cpu',
        'lexical':'Frozen parent TF-IDF settings, refitted on original plus OCR passages; explicit effective numeric dtype',
        'old_vectors_unchanged':True,'cloud_calls':0,'training':False,'benchmark_records_read':False,
        'query_billing':'Original frozen BGE parameter count times actual query slots, including cache hits',
        'output':str(args.output.resolve())}


def build(args,record):
    old.torch.set_num_threads(2)
    encoder = old.Encoder(record['assets'])
    if encoder.parameters != 33360000: raise ValueError('Changed frozen retrieval encoder')
    rows = [json.loads(line) for line in (args.corpus/'passages.jsonl').open()]
    count = record['old_passages']
    values,receipts = [],[]
    for start in range(count,len(rows),16):
        batch = rows[start:start+16]
        vector,receipt = encoder.encode([r['title']+'\n'+r['text'] for r in batch])
        values.append(vector)
        receipts.append({'start':start,'rows':len(batch),**receipt})
    previous = np.load(args.parent/'vectors.npy',allow_pickle=False)
    result = append_vectors(previous,np.concatenate(values),count)
    np.save(args.output/'vectors.npy',result,allow_pickle=False)
    parent_lexical = pickle.loads((Path(read(args.parent/'registration.json')['corpus'])/'tfidf.pkl').read_bytes())
    settings = parent_lexical['vectorizer'].get_params()
    settings['dtype'] = parent_lexical['matrix'].dtype.type
    lexical = TfidfVectorizer(**settings)
    matrix = lexical.fit_transform([r['title']+' '+r['text'] for r in rows])
    (args.output/'tfidf.pkl').write_bytes(pickle.dumps({'sha256':record['corpus_sha256'],
        'vectorizer':lexical,'matrix':matrix},protocol=5))
    write(args.output/'passage-identities.json',[{'id':r['id'],'subject':r['subject']} for r in rows])
    write(args.output/'offline-forward-receipts.json',receipts)
    write(args.output/'summary.json',{'schema':SCHEMA,'old_vectors_reused':count,'new_vectors':len(rows)-count,
        'forward_calls':len(receipts),'offline_forward_token_slots':sum(r['actual_forward_token_slots'] for r in receipts),
        'model_parameters':encoder.parameters,'cloud_calls':0,'training':False,'benchmark_records_read':False})
    if contract(args) != record: raise ValueError('Changed source or evidence during indexing')
    names = ['registration.json','vectors.npy','tfidf.pkl','passage-identities.json','offline-forward-receipts.json','summary.json']
    write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'digests':{name:sha(args.output/name) for name in names},'cloud_calls':0,'training':False,'benchmark_records_read':False})
    print(read(args.output/'summary.json'),flush=True)


class Retriever(retrieval.Retriever):
    def __init__(self,directory=INDEX):
        self.directory = Path(directory)
        done = sealed(self.directory)
        record = read(self.directory/'registration.json')
        args = argparse.Namespace(corpus=Path(record['corpus']),parent=Path(record['parent']),output=self.directory)
        if done['schema'] != SCHEMA or contract(args) != record: raise ValueError('Changed OCR index contract')
        self.passages = [json.loads(line) for line in (args.corpus/'passages.jsonl').open()]
        if read(self.directory/'passage-identities.json') != [{'id':r['id'],'subject':r['subject']} for r in self.passages]:
            raise ValueError('Changed passage order')
        self.vectors = np.load(self.directory/'vectors.npy',allow_pickle=False)
        count = record['old_passages']
        append_vectors(self.vectors[:count],self.vectors[count:],count)
        if not np.array_equal(self.vectors[:count],np.load(args.parent/'vectors.npy',allow_pickle=False)):
            raise ValueError('Changed parent vector prefix')
        cached = pickle.loads((self.directory/'tfidf.pkl').read_bytes())
        self.vectorizer,self.matrix = cached['vectorizer'],cached['matrix']
        if cached['sha256'] != record['corpus_sha256'] or self.matrix.shape[0] != len(self.passages):
            raise ValueError('Changed lexical coverage')
        old.torch.set_num_threads(2)
        self.encoder = old.Encoder(record['assets'])
        if self.encoder.parameters != 33360000: raise ValueError('Changed query encoder')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','build'))
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    record = contract(args)
    if args.phase == 'register':
        args.output.mkdir(parents=True,exist_ok=False)
        write(args.output/'registration.json',record)
        print({'registration_sha256':sha(args.output/'registration.json'),'new_passages':record['OCR_passages']},flush=True)
    else:
        if (not args.registration_sha256 or sha(args.output/'registration.json') != args.registration_sha256
                or read(args.output/'registration.json') != record): raise ValueError('Changed incremental-index registration')
        if {p.name for p in args.output.iterdir()} != {'registration.json'}: raise ValueError('Preserve previous build')
        build(args,record)


if __name__ == '__main__': main()
