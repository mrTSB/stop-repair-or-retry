"""Reserve independent GPQA-main science and unused GSM8K questions before grading.

Raw GPQA examples stay in ignored local storage. Do not publish their text,
answer choices, reference answers, or model rationales containing examples.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys
from urllib.request import Request, urlopen
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from semantic_data import ROOT, read, write, previous
from extended_acquire import SCIENCE_SUFFIX

SEED = 2026112041
REPOSITORY = 'https://github.com/idavidrein/gpqa'
CANARY_PASSWORD = b'deserted-untie-orchid'  # Public download password in the author's README.


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'CS238-controller-research'}), timeout=120) as response:
        return response.read()


def stem_hash(row):
    return sha(previous.stem(row['question']).encode())


def reserve(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cache = ROOT / '.cache/controller-semantic-followup/gpqa'
    cache.mkdir(parents=True, exist_ok=True)
    pin = cache / 'pin.json'
    if pin.exists():
        revision = json.loads(pin.read_bytes())['revision']
    else:
        revision = json.loads(fetch('https://api.github.com/repos/idavidrein/gpqa/commits/main'))['sha']
        pin.write_text(json.dumps({'repository': REPOSITORY, 'revision': revision}, indent=2) + '\n')
    for name in ('dataset.zip', 'LICENSE', 'README.md'):
        target = cache / name
        if not target.exists():
            target.write_bytes(fetch(f'https://raw.githubusercontent.com/idavidrein/gpqa/{revision}/{name}'))
    archive = (cache / 'dataset.zip').read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = [n for n in zipped.namelist() if Path(n).name == 'gpqa_main.csv']
        if len(names) != 1:
            raise ValueError('Unexpected official archive layout')
        raw = zipped.read(names[0], pwd=CANARY_PASSWORD)
    table = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    expected = {'Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2',
                'Incorrect Answer 3', 'High-level domain', 'Record ID'}
    if not table or not expected <= set(table[0]):
        raise ValueError('Unexpected official GPQA-main schema')
    history = json.loads((args.previous / 'provenance/historical-exclusions.json').read_bytes())
    excluded_ids = set(history['ids'])
    excluded_stems = set(history['normalized_stem_sha256'])
    for split in ('train', 'dev', 'test'):
        for row in read(args.previous / 'splits' / f'{split}-questions.jsonl'):
            excluded_ids.add(row['id'])
            excluded_stems.add(stem_hash(row))
    science = []
    quality_exclusions = []
    for item in table:
        subject = item['High-level domain'].strip().lower()
        if subject not in ('chemistry', 'physics'):
            continue
        qid = 'gpqa-main-' + sha(item['Record ID'].encode())[:20]
        options = [(item['Correct Answer'].strip(), True)] + [
            (item[f'Incorrect Answer {i}'].strip(), False) for i in (1, 2, 3)]
        if len({t for t, _ in options}) != 4 or any(not t for t, _ in options):
            quality_exclusions.append({'id': qid, 'subject': subject,
                                       'reason': 'Empty or duplicated answer choices; checked before inference'})
            continue
        random.Random(f'{SEED}:options:{qid}').shuffle(options)
        letters = 'ABCD'
        question = item['Question'].strip() + '\n\nOptions:\n' + '\n'.join(
            f'{letters[i]}. {text}' for i, (text, _) in enumerate(options)) + SCIENCE_SUFFIX
        row = {'id': qid, 'domain': 'science', 'subject': subject, 'question': question,
               'answer': next(letters[i] for i, (_, correct) in enumerate(options) if correct)}
        if qid not in excluded_ids and stem_hash(row) not in excluded_stems:
            science.append(row)
            excluded_ids.add(qid)
            excluded_stems.add(stem_hash(row))
    counts = Counter(r['subject'] for r in science)
    quarter = min(counts['chemistry'], counts['physics'], 200)
    if quarter < 150:
        raise ValueError('Too few independent, non-overlapping validated science questions')

    def choose(rows, n, label):
        return sorted(rows, key=lambda r: hashlib.sha256(f'{SEED}:{label}:{r["id"]}'.encode()).digest())[:n]

    selected = [r for subject in ('chemistry', 'physics')
                for r in choose([r for r in science if r['subject'] == subject], quarter, subject)]
    math = []
    for row in choose(previous.gsm8k(previous.MATH, 'train'), 100000, 'math-pool'):
        fingerprint = stem_hash(row)
        if row['id'] not in excluded_ids and fingerprint not in excluded_stems:
            math.append(row)
            excluded_ids.add(row['id'])
            excluded_stems.add(fingerprint)
    if len(math) < 2 * quarter:
        raise ValueError('Too few unused math questions')
    selected += choose(math, 2 * quarter, 'math')
    selected = choose(selected, len(selected), 'shuffle')
    assert len({r['id'] for r in selected}) == len(selected)
    assert len({stem_hash(r) for r in selected}) == len(selected)
    (args.output / 'splits').mkdir()
    (args.output / 'provenance').mkdir()
    for split in ('train', 'dev'):
        shutil.copy2(args.previous / 'splits' / f'{split}-questions.jsonl',
                     args.output / 'splits' / f'{split}-questions.jsonl')
        for suffix in ('observations', 'late-observations'):
            shutil.copy2(args.previous / f'{split}-{suffix}.jsonl', args.output / f'{split}-{suffix}.jsonl')
    write(args.output / 'splits/test-questions.jsonl', selected)
    history['ids'] = sorted(set(history['ids']) | {r['id'] for split in ('train', 'dev', 'test')
                                                 for r in read(args.previous / 'splits' / f'{split}-questions.jsonl')})
    history['normalized_stem_sha256'] = sorted(set(history['normalized_stem_sha256']) | {
        stem_hash(r) for split in ('train', 'dev', 'test')
        for r in read(args.previous / 'splits' / f'{split}-questions.jsonl')})
    (args.output / 'provenance/historical-exclusions.json').write_text(json.dumps(history, indent=2) + '\n')
    manifest = {'reserved_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
                'questions': len(selected), 'subjects': dict(Counter(r['subject'] for r in selected)),
                'science_source': {'repository': REPOSITORY, 'revision': revision,
                                   'archive_sha256': sha(archive), 'main_csv_sha256': sha(raw),
                                   'subset': 'main; author-validated subset, chemistry and physics only'},
                'math_source_sha256': sha(previous.MATH.read_bytes()),
                'test_sha256': sha((args.output / 'splits/test-questions.jsonl').read_bytes()),
                'previous_confirmation_sha256': sha((args.previous / 'splits/test-questions.jsonl').read_bytes()),
                'selection': 'Fixed hash order and same 50/25/25 proportions; no model outcomes used. Reserve before reading earlier confirmation scores.',
                'fresh_inference_performed': False, 'raw_examples_publishable': False,
                'source_quality_exclusions': quality_exclusions,
                'authors_request': 'Keep GPQA example text, choices, answers and revealing rationales out of online artifacts.',
                'comparison': 'Separate transfer benchmark; report separately from the original same-source panel.'}
    manifest['splits'] = {split: {'n': len(read(args.output / 'splits' / f'{split}-questions.jsonl')),
                                  'sha256': sha((args.output / 'splits' / f'{split}-questions.jsonl').read_bytes())}
                          for split in ('train', 'dev', 'test')}
    (args.output / 'splits/selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('reserved_at_utc', 'questions', 'subjects', 'test_sha256', 'selection')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    reserve(parser.parse_args())
