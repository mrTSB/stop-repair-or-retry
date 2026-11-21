"""Reserve disjoint human-labeled science knowledge data for a cheap verifier.

The model host receives public UUIDs only, and downloads the pinned official
source itself. No local question text, answers, or solver responses are uploaded.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from reserve import DuplicateIndex, normalized
from semantic_data import ROOT, read

SEED = 2026112141


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Use a new science auxiliary reservation')
    pin = json.loads((args.cache / 'pin.json').read_bytes())
    source = args.cache / 'SuperGPQA-all.jsonl'
    old_ids = set()
    index = DuplicateIndex()
    files = {}
    for base in (ROOT / 'runs', ROOT / 'results'):
        for path in sorted(base.rglob('*-questions.jsonl')):
            files[str(path.relative_to(ROOT))] = sha(path)
            for row in read(path):
                old_ids.add(row['id'])
                text = normalized(row['question'])
                if text not in index.exact:
                    index.add(text)
    available = defaultdict(list)
    excluded = Counter()
    items = sorted(read(source), key=lambda r: hashlib.sha256(f'{SEED}:{r["uuid"]}'.encode()).digest())
    for item in items:
        if item['discipline'] != 'Science' or item['field'] not in ('Chemistry', 'Physics'):
            continue
        qid = 'supergpqa-' + item['uuid']
        options = item['options']
        letter = item['answer_letter'].strip().upper()
        valid = (4 <= len(options) <= 10 and len(set(options)) == len(options) and all(t.strip() for t in options)
                 and len(letter) == 1 and letter in 'ABCDEFGHIJ'[:len(options)])
        if not valid or ' '.join(options[ord(letter) - 65].split()).casefold() != ' '.join(item['answer'].split()).casefold():
            excluded['invalid_choice_structure'] += 1
            continue
        norm = normalized(item['question'])
        if qid in old_ids or index.duplicate(norm):
            excluded['historical_or_near_duplicate'] += 1
            continue
        index.add(norm)
        available[item['field']].append(item['uuid'])
    train, calibration = [], []
    for subject in ('Chemistry', 'Physics'):
        if len(available[subject]) < 800:
            raise ValueError(f'Need800disjoint {subject} questions, found {len(available[subject])}')
        train += available[subject][:720]
        calibration += available[subject][720:800]
    protected = {r['id'].removeprefix('supergpqa-') for split in ('train', 'dev', 'test')
                 for r in read(args.protected / 'splits' / (split + '-questions.jsonl')) if r['domain'] == 'science'}
    assert not protected & set(train + calibration)
    metadata = {'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
                'repository': pin['repo'], 'revision': pin['revision'], 'source_sha256': sha(source),
                'train_ids': train, 'calibration_ids': calibration,
                'protected_split_sha256': {split: sha(args.protected / 'splits' / (split + '-questions.jsonl'))
                                           for split in ('train', 'dev', 'test')},
                'history_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                'selection': 'Fixed hashes and equal subject counts; no outcomes or difficulty labels select data.',
                'license': 'ODC-BY and respective referenced-source licenses',
                'public_source_url': 'https://huggingface.co/datasets/m-a-p/SuperGPQA',
                'raw_question_or_answer_payload': False, 'solver_response_payload': False,
                'source_code_sha256': sha(__file__), 'exclusions': dict(excluded)}
    args.output.mkdir(parents=True)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'train_questions': len(train), 'calibration_questions': len(calibration),
                      'metadata_sha256': sha(args.output / 'metadata.json'),
                      'remaining_pool': {k: len(v) - 800 for k, v in available.items()},
                      'protected_split_overlap': 0, 'text_or_answers_uploaded': False}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=ROOT / '.cache/controller-semantic-followup/supergpqa')
    p.add_argument('--protected', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
