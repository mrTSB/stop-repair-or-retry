"""Collect pinned OpenStax exposition only, excluding exercises and worked answer material."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

BOOKS={'chemistry':'osbooks-chemistry-bundle','physics':'osbooks-university-physics-bundle'}
BLOCKED={'example','exercise','problem','solution','answer','equation-number','footnote'}


def get(url):
    with urlopen(Request(url,headers={'User-Agent':'cs238-controller-research'}),timeout=40) as response:
        return response.read()


def tag(node):
    return node.tag.rsplit('}',1)[-1]


def exposition(raw):
    document=ET.fromstring(raw)
    titles=[re.sub(r'\s+',' ',' '.join(n.itertext())).strip() for n in document.iter() if tag(n)=='title']
    title=titles[0] if titles else ''
    passages=[]
    def math_text(node):
        children=list(node);kind=tag(node)
        if kind=='msup' and len(children)==2:return '('+math_text(children[0])+')^('+math_text(children[1])+')'
        if kind=='msub' and len(children)==2:return math_text(children[0])+'_('+math_text(children[1])+')'
        if kind=='mfrac' and len(children)==2:return '('+math_text(children[0])+')/('+math_text(children[1])+')'
        if kind=='msqrt':return 'sqrt('+''.join(math_text(c) for c in children)+')'
        if kind=='annotation':return ''
        return (node.text or '')+''.join(math_text(c)+(c.tail or '') for c in children)
    def plain(node):
        if tag(node)=='math':return ' '+math_text(node)+' '
        return (node.text or '')+''.join(plain(c)+(c.tail or '') for c in node)
    def walk(node,excluded=False):
        kind=tag(node)
        classes=node.get('class','').lower()
        excluded=excluded or kind in BLOCKED or any(word in classes for word in
            ('exercise','problem','solution','answer','review-question','check-understanding','conceptual-question'))
        if excluded:return
        if kind=='para':
            text=re.sub(r'\s+',' ',plain(node)).strip()
            if (len(text)>=90 and len(text)<=2400 and '?' not in text and not
                    re.search(r'\b(?:correct answer|answer is|choice [A-J])\b',text,re.I)):
                passages.append(text)
            return
        if kind=='equation':
            formula=re.sub(r'\s+',' ',plain(node)).strip()
            if formula and passages:passages[-1]+=' Formula: '+formula
            return
        for child in node:walk(child,excluded)
    walk(document)
    return title,passages


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    manifest={};all_passages=[];seen=set()
    for subject,repo in BOOKS.items():
        api='https://api.github.com/repos/openstax/'+repo
        pin=output/(subject+'-pin.json')
        if pin.exists():revision=json.loads(pin.read_bytes())['revision']
        else:
            obj=json.loads(get(api+'/commits/main'));revision=obj['sha']
            pin.write_text(json.dumps({'repository':'https://github.com/openstax/'+repo,'revision':revision},indent=2)+'\n')
        commit=json.loads(get(api+'/commits/'+revision))
        tree=json.loads(get(api+'/git/trees/'+commit['commit']['tree']['sha']+'?recursive=1'))
        paths=[r['path'] for r in tree['tree'] if r['path'].startswith('modules/') and r['path'].endswith('/index.cnxml')]
        folder=output/subject;folder.mkdir(exist_ok=True)
        license_url='https://raw.githubusercontent.com/openstax/'+repo+'/'+revision+'/LICENSE'
        (folder/'LICENSE').write_bytes(get(license_url))
        def one(path):
            target=folder/(path.split('/')[1]+'.cnxml')
            url='https://raw.githubusercontent.com/openstax/'+repo+'/'+revision+'/'+path
            if not target.exists():target.write_bytes(get(url))
            raw=target.read_bytes();title,paragraphs=exposition(raw)
            return path,url,hashlib.sha256(raw).hexdigest(),title,paragraphs
        with ThreadPoolExecutor(max_workers=4) as pool:
            for path,url,fingerprint,title,paragraphs in pool.map(one,paths):
                for index,text in enumerate(paragraphs):
                    digest=hashlib.sha256(text.encode()).hexdigest()
                    if digest in seen:continue
                    seen.add(digest)
                    all_passages.append({'id':subject+':'+path.split('/')[1]+':'+str(index),
                        'subject':subject,'title':title,'text':text,'source':url,'revision':revision,
                        'source_sha256':fingerprint,'license':'CC BY-NC-SA 4.0','publisher':'OpenStax, Rice University'})
        manifest[subject]={'repository':'https://github.com/openstax/'+repo,'revision':revision,
                           'modules':len(paths),'license_url':license_url}
        print(subject,'modules',len(paths),'total exposition passages',len(all_passages),flush=True)
    p=output/'passages.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in all_passages))
    manifest.update({'passages':len(all_passages),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                     'filter':'Exposition paragraphs only. Exercises, problems, solutions, answers, worked examples, question paragraphs excluded.'})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
