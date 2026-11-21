"""Local, pinned public-science supervision and durable exclusion provenance."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks  # Establish archived imports without changing them.
from semantic_data import read, previous
from extended_acquire import SCIENCE_SUFFIX

PROFILE = 'supergpqa-option-correctness-v1'
LETTERS = 'ABCDEFGHIJ'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def question(item):
    return item['question'].strip() + '\n\nOptions:\n' + '\n'.join(
        f'{LETTERS[i]}. {v.strip()}' for i, v in enumerate(item['options'])) + SCIENCE_SUFFIX


def checked_groups(metadata_path, cache, protected):
    """Read only registered auxiliary labels; protected rows supply IDs/hashes."""
    metadata = json.loads(metadata_path.read_bytes())
    pin = json.loads((cache / 'pin.json').read_bytes())
    source = cache / 'SuperGPQA-all.jsonl'
    if (pin['repo'], pin['revision'], sha(source)) != (
            metadata['repository'], metadata['revision'], metadata['source_sha256']):
        raise ValueError('Pinned auxiliary source changed')
    protected_ids = set()
    for split, expected in metadata['protected_split_sha256'].items():
        path = protected / 'splits' / (split + '-questions.jsonl')
        if sha(path) != expected:
            raise ValueError('Protected question split changed')
        protected_ids.update(r['id'] for r in read(path))
    chosen = metadata['train_ids'] + metadata['calibration_ids']
    if len(chosen) != len(set(chosen)) or len(chosen) != 1600:
        raise ValueError('Expected 1600 unique registered auxiliary questions')
    if protected_ids & {'supergpqa-' + q for q in chosen}:
        raise ValueError('Auxiliary data overlaps a protected question split')
    by_id = {r['uuid']: r for r in read(source) if r['uuid'] in set(chosen)}
    if set(by_id) != set(chosen):
        raise ValueError('Registered UUID missing from pinned source')
    groups = {}
    for split, count in (('train', 1440), ('calibration', 160)):
        rows = []
        for qid in metadata[split + '_ids']:
            r = by_id[qid]
            opts = [o.strip() for o in r['options']]
            letter = r['answer_letter'].strip().upper()
            if not (4 <= len(opts) <= 10 and len(set(opts)) == len(opts) and all(opts)
                    and letter in LETTERS[:len(opts)] and len(letter) == 1):
                raise ValueError('Malformed registered option structure')
            if ' '.join(opts[LETTERS.index(letter)].split()).casefold() != ' '.join(r['answer'].split()).casefold():
                raise ValueError('Official letter and answer text disagree')
            rows.append({'id': 'supergpqa-' + qid, 'subject': r['field'].casefold(),
                         'question': question(r), 'options': opts, 'answer': letter})
        if len(rows) != count or Counter(r['subject'] for r in rows) != {'chemistry': count // 2, 'physics': count // 2}:
            raise ValueError('Changed registered subject quotas')
        groups[split] = rows
    return metadata, groups


def seal(metadata_path, cache, protected):
    metadata, groups = checked_groups(metadata_path, cache, protected)
    rows = groups['train'] + groups['calibration']
    folder = metadata_path.parent / 'provenance'
    folder.mkdir(exist_ok=True)
    exclusion = {'ids': sorted(r['id'] for r in rows),
                 'normalized_stem_sha256': sorted({hashlib.sha256(previous.stem(r['question']).encode()).hexdigest() for r in rows}),
                 'metadata_sha256': sha(metadata_path), 'source_sha256': metadata['source_sha256'],
                 'note': 'Reserved auxiliary supervised training and calibration; exclude from all future fresh panels.'}
    path = folder / 'historical-exclusions.json'
    if path.exists() and json.loads(path.read_bytes()) != exclusion:
        raise ValueError('Changed sealed auxiliary exclusions')
    path.write_text(json.dumps(exclusion, indent=2) + '\n')
    receipt = {'metadata_sha256': sha(metadata_path), 'source_sha256': metadata['source_sha256'],
               'historical_exclusions_sha256': sha(path), 'sealer_sha256': sha(__file__),
               'train_questions': 1440, 'calibration_questions': 160, 'protected_overlap': 0,
               'raw_text_or_answers_in_receipt': False, 'training_started': False,
               'source_revision': metadata['revision']}
    (folder / 'seal-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def option_entry(row, proposed, numeric_width):
    """Inference input uses only the question/options and proposed answer."""
    raw = str(proposed).strip()
    # Do not reinterpret arbitrary text as a correct choice. Unknown outputs
    # still cost an encoder pass and receive an explicit invalid-answer input.
    letter = raw.upper() if len(raw) == 1 else ''
    options = row.get('options')
    if options is None:
        body = row['question'].removesuffix(SCIENCE_SUFFIX).partition('\n\nOptions:\n')[2]
        chunks = re.split(r'(?m)^([A-J])\. ', body)
        options = []
        for pos in range(1, len(chunks), 2):
            if chunks[pos] != LETTERS[len(options)]:
                raise ValueError('Noncanonical public option order')
            options.append(chunks[pos + 1].strip())
    if not options or not 4 <= len(options) <= 10:
        raise ValueError('The science option observer requires all official options')
    valid = letter in LETTERS[:len(options)] and bool(letter)
    right = ('Proposed answer: ' + letter + '. ' + options[LETTERS.index(letter)]
             if valid else 'Proposed answer: invalid or unparsed answer.')
    return {'left': row['question'], 'right': right, 'numeric': [0.] * numeric_width}


def examples(rows, numeric_width):
    result = []
    for row in rows:
        count = len(row['options'])
        for i in range(count):
            letter = LETTERS[i]
            correct = letter == row['answer']
            result.append({**option_entry(row, letter, numeric_width), 'id': row['id'],
                           'subject': row['subject'], 'choice': letter, 'label': int(correct),
                           'weight': .5 if correct else .5 / (count - 1)})
    # Equal total weight per question and equal positive/negative mass. The
    # training loss uses a fixed batch denominator, never a batch weight sum.
    scale = len(result) / len(rows)
    for row in result:
        row['weight'] *= scale
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'cache', 'protected'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(seal(args.metadata, args.cache, args.protected)))
