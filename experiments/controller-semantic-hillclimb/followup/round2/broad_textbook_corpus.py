"""Question-independent broader MIT OCW lecture exposition in a new corpus.

Only lecture-page PDF links are followed. Explicit answer keys, exams, benchmark records,
and search-engine snippets never enter the corpus. Every fetched byte is sealed.
This prepares evidence, not a controller result or permission to change a pilot.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import pickle
import re
import sys
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evidence import Retriever

COURSES = (
    ('chemistry', '5-12-organic-chemistry-i-spring-2003', 'Sarah Tabacco; Barbara Imperiali', 'lecture-handouts'),
    ('chemistry', '5-13-organic-chemistry-ii-fall-2006', 'Kimberly Berkowski; Timothy Jamison', 'lecture-notes'),
    ('physics', '8-06-quantum-physics-iii-spring-2018', 'Barton Zwiebach', 'lecture-notes'),
    ('physics', '8-07-electromagnetism-ii-fall-2012', 'Alan Guth; Min Chen', 'lecture-notes'),
    ('physics', '22-02-introduction-to-applied-nuclear-physics-spring-2012', 'Paola Cappellaro', 'lecture-notes'),
)
# The electromagnetic course includes two journal-article resources; omit them
# rather than assigning the general course's CC license to third-party papers.
THIRD_PARTY_RESOURCES = {'mit8_07f12_ln12', 'mit8_07f12_ln13'}


def allowed_resource(url, title):
    value = (url+' '+title).casefold()
    if any(token in value for token in THIRD_PARTY_RESOURCES): return False
    return not re.search(r'(?:^|[/_\W])(?:answers?|ans|solutions?|exam\d*|pset\d*|prob_sol|prob_solving)(?:[_\W]|$)', value)

LICENSE = 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
SCHEMA = 'question-independent-broader-OCW-lecture-corpus-v2'


def sha(data): return hashlib.sha256(data).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]; self.current=None
    def handle_starttag(self, tag, attrs):
        if tag=='a': self.current=[dict(attrs).get('href',''),'']
    def handle_data(self, text):
        if self.current is not None: self.current[1]+=text
    def handle_endtag(self, tag):
        if tag=='a' and self.current is not None:
            self.links.append(tuple(self.current)); self.current=None


def links(raw):
    parser=Links(); parser.feed(raw.decode('utf-8')); return parser.links


def checked_url(url):
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.netloc!='ocw.mit.edu' or not parsed.path.startswith('/courses/'):
        raise ValueError('Only official public OCW course assets are allowed')
    return url


def fetch(folder, url):
    checked_url(url); key=sha(url.encode()); path=folder/key; receipt=folder/(key+'.json')
    if path.exists() or receipt.exists():
        if not(path.exists() and receipt.exists()): raise ValueError('Unsealed download requires inspection')
        info=json.loads(receipt.read_text()); raw=path.read_bytes()
        if info['url']!=url or info['sha256']!=sha(raw): raise ValueError('Changed archived source')
        return raw
    with urlopen(Request(url,headers={'User-Agent':'CS238 academic retrieval experiment'}),timeout=45) as response:
        checked_url(response.url); raw=response.read(40*1024*1024+1)
        if len(raw)>40*1024*1024: raise ValueError('Unexpectedly large lecture asset')
    temporary=path.with_suffix('.tmp'); temporary.write_bytes(raw); temporary.replace(path)
    write(receipt,{'url':url,'sha256':sha(raw),'bytes':len(raw),'fetched_utc':datetime.now(timezone.utc).isoformat()})
    return raw


def documents(folder):
    result={}
    for subject,course,author,section in COURSES:
        page=f'https://ocw.mit.edu/courses/{course}/pages/{section}/'
        lecture=fetch(folder,page)
        if LICENSE.encode() not in lecture: raise ValueError('Course license link changed')
        resources=sorted({(urljoin(page,u),' '.join(t.split())) for u,t in links(lecture)
                          if 'PDF' in t.upper() and f'/courses/{course}/' in urljoin(page,u) and allowed_resource(urljoin(page,u),t)})
        if not 5<=len(resources)<=60: raise ValueError('Unexpected lecture PDF inventory')
        for resource,title in resources:
            if resource.lower().endswith('.pdf'): targets=[resource]
            else:
                targets=sorted({urljoin(resource,u) for u,_ in links(fetch(folder,resource))
                    if urlparse(urljoin(resource,u)).path.lower().endswith('.pdf')
                    and f'/courses/{course}/' in urljoin(resource,u)})
            if len(targets)!=1: raise ValueError('Lecture resource must identify exactly one PDF')
            url=checked_url(targets[0])
            if not allowed_resource(url,title): continue
            result[url]={'subject':subject,'course':course,'title':title.removesuffix(' (PDF)'),
                'author':author,'source':url,'resource_page':resource,'course_page':page,'license':LICENSE}
        print({'course':course,'lecture_resources':len(resources)},flush=True)
    return [result[k] for k in sorted(result)]


def build(args):
    from pypdf import PdfReader, __version__ as pdf_version
    args.output.mkdir(parents=True,exist_ok=True); raw_dir=args.output/'sources';raw_dir.mkdir(exist_ok=True)
    contract={'schema':SCHEMA,'courses':COURSES,'source_sha256':sha(Path(__file__).read_bytes()),
        'retriever_source_sha256':sha(Path(inspect.getfile(Retriever)).read_bytes()),
        'base_passages_sha256':sha((args.base/'passages.jsonl').read_bytes()),'pypdf':pdf_version,
        'chunk_words':140,'stride_words':110,'minimum_page_words':35,'benchmark_records_read':False,
        'source_selection':'Fixed five-course expansion, lecture-page PDF links excluding answer keys, assignments and two third-party papers; no question or outcome selection',
        'zero_text_documents':'Retain coverage receipts and report exclusion from text index; no OCR or replacement documents',
        'extraction':'Page-local text; normalize whitespace; no OCR or equation reconstruction'}
    registration=args.output/'registration.json'
    canonical=json.loads(json.dumps(contract))
    if registration.exists():
        if json.loads(registration.read_text())!=canonical: raise ValueError('Changed corpus build contract')
    else: write(registration,contract)
    if (args.output/'complete.json').exists(): raise ValueError('Preserve completed corpus')
    inventory=documents(raw_dir); write(args.output/'document-inventory.json',inventory)
    passages=[json.loads(s) for s in (args.base/'passages.jsonl').read_text().splitlines()]
    original=len(passages); coverage=[]
    for i,entry in enumerate(inventory):
        raw=fetch(raw_dir,entry['source'])
        if not raw.startswith(b'%PDF'): raise ValueError('Expected PDF bytes')
        reader=PdfReader(BytesIO(raw)); included=0;count=0;words_total=0
        for page_index,page in enumerate(reader.pages):
            words=re.sub(r'\s+',' ',page.extract_text() or '').strip().split();words_total+=len(words)
            if len(words)<35: continue
            included+=1
            for start in range(0,len(words),110):
                chunk=words[start:start+140]
                if len(chunk)<35: break
                passages.append({**entry,'text':' '.join(chunk),'page':page_index+1,'word_offset':start,
                    'modified':'Extracted page text, normalized whitespace, overlapping140-word windows',
                    'source_sha256':sha(raw),'id':sha((entry['source']+f':{page_index}:{start}').encode())})
                count+=1
        coverage.append({'source':entry['source'],'sha256':sha(raw),'pages':len(reader.pages),
            'included_pages':included,'words_extracted':words_total,'passages':count})
        if (i+1)%10==0: print({'documents_processed':i+1,'total':len(inventory),'new_passages':len(passages)-original},flush=True)
    if not any(v['passages']>0 for v in coverage): raise ValueError('No extractable lecture exposition')
    out=args.output/'passages.jsonl';out.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in passages))
    index=Retriever(args.output)
    manifest={**contract,'sha256':sha(out.read_bytes()),'base_passages':original,'new_passages':len(passages)-original,
        'passages':len(passages),'documents':len(inventory),'new_by_subject':dict(Counter(v['subject'] for v in passages[original:])),
        'coverage':coverage,'zero_text_documents':[v['source'] for v in coverage if v['passages']==0],'quality_limitations':'PDF equations and reading order may be degraded; no OCR; not evaluated on benchmark questions',
        'license_notice':'MIT OCW lecture adaptations: CC BY-NC-SA4.0 with attribution per passage; original base passage licenses retained'}
    if index.sha256!=manifest['sha256']: raise ValueError('Index/text mismatch')
    write(args.output/'manifest.json',manifest)
    write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':sha(registration.read_bytes()),
        'benchmark_records_read':False,'model_calls':0,'digests':{str(p.relative_to(args.output)):sha(p.read_bytes())
            for p in args.output.rglob('*') if p.is_file() and p.name!='complete.json'}})
    print({k:manifest[k] for k in ('documents','new_passages','new_by_subject','sha256')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','output'):parser.add_argument('--'+name,type=Path,required=True)
    build(parser.parse_args())
