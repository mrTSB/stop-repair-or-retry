"""Recover all seven frozen zero-text PDFs using offline CPU-requested OCR.

The document subset is fixed by the pre-existing text-extraction coverage,
not by any benchmark question or model outcome. Preserve the complete old
passage prefix byte-for-byte; append attributed machine transcripts only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

SCHEMA = 'question-independent-zero-text-lecture-OCR-corpus-v1'
TOOL = Path(__file__).with_name('textbook_ocr.swift')


def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def chunks(text,words=140,stride=110):
    values = text.split()
    if len(values) < 20: return []
    return [(start,' '.join(values[start:start+words])) for start in range(0,len(values),stride)
            if len(values[start:start+words]) >= 20]


def inputs(args):
    manifest,done = read(args.base/'manifest.json'),read(args.base/'complete.json')
    if manifest['benchmark_records_read'] or done['benchmark_records_read']:
        raise ValueError('Require the original question-independent corpus')
    for name,value in done['digests'].items():
        if sha(args.base/name) != value: raise ValueError('Changed frozen corpus file')
    coverage = {v['source']:v for v in manifest['coverage']}
    inventory = read(args.base/'document-inventory.json')
    if isinstance(inventory,dict): inventory = inventory.get('documents',inventory.get('inventory',[]))
    by_source = {v['source']:v for v in inventory}
    documents = []
    evidence = {str((args.base/k).resolve()):sha(args.base/k) for k in ('manifest.json','complete.json','passages.jsonl','document-inventory.json')}
    selected = manifest['zero_text_documents']
    if len(selected) != 7 or len(set(selected)) != 7: raise ValueError('Require all seven original zero-text documents')
    for url in sorted(selected):
        current = coverage[url]
        if current['passages'] != 0: raise ValueError('Source subset must depend only on the sealed extraction failure')
        path = args.base/'sources'/hashlib.sha256(url.encode()).hexdigest()
        if sha(path) != current['sha256']: raise ValueError('Changed frozen PDF bytes')
        info = by_source[url]
        documents.append({**info,'source_sha256':current['sha256'],'pages':current['pages'],'local_file':str(path.resolve())})
        evidence[str(path.resolve())] = current['sha256']
    compiler = subprocess.check_output(['swiftc','--version'],text=True).strip()
    renderer = subprocess.run(['pdftoppm','-v'],capture_output=True,text=True,check=True).stderr.strip()
    record = {'schema':SCHEMA,'source_sha256':sha(__file__),'ocr_source_sha256':sha(TOOL),
        'input_evidence':evidence,'paths':{'base':str(args.base.resolve()),'output':str(args.output.resolve())},
        'documents':documents,'old_passages':manifest['passages'],'new_document_selection':'All seven PDFs with zero passages in the frozen coverage manifest',
        'render_long_edge':1400,'render_format':'PNG','renderer':renderer,'swift_compiler':compiler,'macOS':platform.mac_ver()[0],
        'OCR_revision':3,'OCR_language':'en-US','CPU_only_requested':True,'language_correction':False,
        'all_OCR_candidates_preserved':True,'confidence_filter':False,'minimum_page_words':20,'chunk_words':140,'stride_words':110,
        'old_passage_prefix_unchanged':True,'new_neural_training':False,'cloud_calls':0,'benchmark_records_read':False,
        'local_OCR_compute':'Offline wall time/CPU-requested recognition; OS-owned model parameter count unknown; excluded from per-question inference proxy and disclosed separately',
        'license_notice':'New transcripts adapt frozen MIT OCW lectures under CC BY-NC-SA4.0; retain each source author, course, page and URL; original passage licenses unchanged',
        'quality_limitations':'Machine OCR may misread mathematical notation, chemical structures and reading order; original PDF/page evidence remains available'}
    return documents,record


def build(args,documents,record):
    env = {**os.environ,'CLANG_MODULE_CACHE_PATH':'/private/tmp/srr-swift-cache','SWIFT_MODULECACHE_PATH':'/private/tmp/srr-swift-cache'}
    tool = args.output/'ocr-tool'
    with (args.output/'compiler.log').open('w') as log:
        subprocess.run(['swiftc',str(TOOL),'-o',str(tool)],env=env,stdout=log,stderr=log,check=True)
    pages = args.output/'pages'
    pages.mkdir()
    rendered,render_time = [],0.
    for document in documents:
        prefix = pages/hashlib.sha256(document['source'].encode()).hexdigest()
        started = time.monotonic()
        with (args.output/'render.log').open('a') as log:
            subprocess.run(['pdftoppm','-png','-scale-to','1400',document['local_file'],str(prefix)],stdout=log,stderr=log,check=True)
        render_time += time.monotonic()-started
        found = list(pages.glob(prefix.name+'-*.png'))
        numbers = {int(re.search(r'-(\d+)\.png$',p.name).group(1)):p for p in found}
        if set(numbers) != set(range(1,document['pages']+1)): raise ValueError('Incomplete rendered PDF; preserve partial outputs')
        for number in sorted(numbers):
            rendered.append({'path':str(numbers[number].resolve()),'page':number,'document':document,'image_sha256':sha(numbers[number])})
        print({'rendered_documents':len({v['document']['source'] for v in rendered}),'rendered_pages':len(rendered)},flush=True)
    by_path = {v['path']:v for v in rendered}
    raw = args.output/'ocr-pages.jsonl'
    seen = set()
    added,empty,ocr_time = [],[],0.
    with raw.open('x') as stream,(args.output/'ocr-errors.log').open('w') as errors:
        process = subprocess.Popen([str(tool.resolve()),*by_path],stdout=subprocess.PIPE,stderr=errors,text=True,env=env)
        for line in process.stdout:
            result = json.loads(line)
            path = result['file']
            if path not in by_path or path in seen or result['revision'] != 3 or not result['cpu_only_requested']:
                raise ValueError('Unexpected OCR page/revision or repeated recognition')
            seen.add(path)
            stream.write(json.dumps(result,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
            meta = by_path[path]
            document = meta['document']
            ocr_time += result['wall_seconds']
            text = ' '.join(v['text'] for v in result['lines'])
            windows = chunks(text)
            if not windows: empty.append({'source':document['source'],'page':meta['page'],'words':len(text.split())})
            for start,body in windows:
                identity = hashlib.sha256(f'{document["source"]}:{meta["page"]}:{start}'.encode()).hexdigest()
                added.append({k:v for k,v in {**document,'id':'ocr:'+identity,'page':meta['page'],'word_offset':start,
                    'text':body,'title':'OCR lecture page '+str(meta['page'])+' '+document.get('course',''),
                    'modified':'Local machine OCR; equations/structures/reading order may be imperfect; normalized whitespace and overlapping140-word windows',
                    'license':'https://creativecommons.org/licenses/by-nc-sa/4.0/',
                    'image_sha256':meta['image_sha256'],'OCR_revision':3}.items() if k not in ('local_file','pages')})
            if len(seen)%8 == 0 or len(seen) == len(rendered):
                print({'OCR_pages':len(seen),'total':len(rendered),'added_passages':len(added)},flush=True)
        if process.wait() != 0 or seen != set(by_path): raise ValueError('OCR process failed or skipped a source page')
    old = (args.base/'passages.jsonl').read_bytes()
    if old and not old.endswith(b'\n'): raise ValueError('Broken original passage prefix')
    payload = old+''.join(json.dumps(v,sort_keys=True)+'\n' for v in added).encode()
    (args.output/'passages.jsonl').write_bytes(payload)
    if not payload.startswith(old): raise ValueError('Original passage prefix changed')
    write(args.output/'page-provenance.json',rendered)
    write(args.output/'manifest.json',{'schema':SCHEMA,'old_passages':record['old_passages'],'OCR_passages':len(added),
        'passages':record['old_passages']+len(added),'sha256':sha(args.output/'passages.jsonl'),
        'old_passages_sha256':record['input_evidence'][str((args.base/'passages.jsonl').resolve())],
        'documents':len(documents),'rendered_pages':len(rendered),'local_OCR_calls':len(seen),
        'pages_without_qualifying_OCR':empty,'render_wall_seconds':render_time,'OCR_wall_seconds':ocr_time,
        'CPU_only_requested':True,'cloud_calls':0,'training':False,'benchmark_records_read':False,
        'license_notice':record['license_notice'],'quality_limitations':record['quality_limitations']})
    if inputs(args)[1] != record: raise ValueError('Registered source/runtime evidence changed')
    names = [str(p.relative_to(args.output)) for p in args.output.rglob('*') if p.is_file()]
    write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'digests':{name:sha(args.output/name) for name in names},'cloud_calls':0,'benchmark_records_read':False,'training':False})
    print(read(args.output/'manifest.json'),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','build'))
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    documents,record = inputs(args)
    if args.phase == 'register':
        args.output.mkdir(parents=True,exist_ok=False)
        write(args.output/'registration.json',record)
        print({'registration_sha256':sha(args.output/'registration.json'),'documents':len(documents),'pages':sum(v['pages'] for v in documents),'cloud_calls':0},flush=True)
        return
    if (not args.registration_sha256 or sha(args.output/'registration.json') != args.registration_sha256
            or read(args.output/'registration.json') != record): raise ValueError('Changed OCR corpus contract')
    if {p.name for p in args.output.iterdir()} != {'registration.json'}: raise ValueError('Preserve previous corpus-build outputs')
    build(args,documents,record)


if __name__ == '__main__': main()
