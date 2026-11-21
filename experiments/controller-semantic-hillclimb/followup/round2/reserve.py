"""Reserve new paired fitting data and an untouched prospective transfer panel.

All selection happens before inference. Original study files are read-only.
Raw questions and references belong only in ignored runtime storage.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import banks  # Establish the archived imports without changing archived code.
from semantic_data import ROOT, previous, read, write
from extended_acquire import SCIENCE_SUFFIX

SEED = 2026112103


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(question):
    text = question.removesuffix(SCIENCE_SUFFIX).partition('\n\nOptions:\n')[0]
    return ' '.join(re.findall(r'\w+', text.casefold()))


def shingles(text):
    words = text.split()
    return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}


class DuplicateIndex:
    def __init__(self):
        self.exact = set()
        self.index = defaultdict(set)
        self.sizes = []

    def add(self, text):
        grams = shingles(text)
        idx = len(self.sizes)
        self.exact.add(text)
        self.sizes.append(len(grams))
        for gram in grams:
            self.index[gram].add(idx)

    def duplicate(self, text):
        if text in self.exact:
            return True
        grams = shingles(text)
        if len(grams) < 5:
            return False
        hits = Counter(i for gram in grams for i in self.index.get(gram, ()))
        return any(count >= .85 * min(len(grams), self.sizes[i])
                   and max(len(grams), self.sizes[i]) <= 1.5 * min(len(grams), self.sizes[i])
                   for i, count in hits.items())


def run(args):
    if args.output.exists():
        raise ValueError('Use a new reservation path')
    source = args.cache / 'SuperGPQA-all.jsonl'
    pin = json.loads((args.cache / 'pin.json').read_bytes())
    history_ids, history_stems = set(), set()
    index = DuplicateIndex()
    historical = {}
    for base in (ROOT / 'results', ROOT / 'runs'):
        for path in sorted(base.rglob('*-questions.jsonl')):
            historical[str(path.relative_to(ROOT))] = sha(path)
            for row in read(path):
                history_ids.add(row['id'])
                history_stems.add(hashlib.sha256(previous.stem(row['question']).encode()).hexdigest())
                norm = normalized(row['question'])
                if norm not in index.exact:
                    index.add(norm)
    # Historical exclusions also include earlier records no longer in runtime.
    for path in sorted((ROOT / 'results').rglob('historical-exclusions.json')):
        receipt = json.loads(path.read_bytes())
        history_ids.update(receipt.get('ids', ()))
        history_stems.update(receipt.get('normalized_stem_sha256', ()))
    for path in sorted((ROOT / 'runs').rglob('historical-exclusions.json')):
        receipt = json.loads(path.read_bytes())
        history_ids.update(receipt.get('ids', ()))
        history_stems.update(receipt.get('normalized_stem_sha256', ()))
    extra_ids, extra_text = previous.historical()
    history_ids.update(extra_ids)
    history_stems.update(hashlib.sha256(s.encode()).hexdigest() for s in extra_text)
    for text in extra_text:
        norm = normalized(text)
        if norm not in index.exact:
            index.add(norm)
    candidates = previous.gsm8k(previous.MATH, 'train')
    exclusions = Counter()
    for item in read(source):
        subject = item['field'].casefold()
        if item['discipline'] != 'Science' or subject not in ('chemistry', 'physics'):
            continue
        options = [v.strip() for v in item['options']]
        letter = item['answer_letter'].strip().upper()
        valid = (4 <= len(options) <= 10 and len(set(options)) == len(options)
                 and all(options) and len(letter) == 1 and letter in 'ABCDEFGHIJ'[:len(options)])
        if not valid:
            exclusions['invalid_choice_structure'] += 1
            continue
        if ' '.join(options[ord(letter) - 65].split()).casefold() != ' '.join(item['answer'].split()).casefold():
            exclusions['reference_text_does_not_match_labeled_option'] += 1
            continue
        question = item['question'].strip() + '\n\nOptions:\n' + '\n'.join(
            f'{chr(65 + i)}. {text}' for i, text in enumerate(options)) + SCIENCE_SUFFIX
        candidates.append({'id': 'supergpqa-' + item['uuid'], 'domain': 'science',
                           'subject': subject, 'question': question, 'answer': letter,
                           'source': 'm-a-p/SuperGPQA@' + pin['revision']})
    available = defaultdict(list)
    for row in sorted(candidates, key=lambda r: hashlib.sha256(f'{SEED}:pool:{r["id"]}'.encode()).digest()):
        norm = normalized(row['question'])
        old_stem = hashlib.sha256(previous.stem(row['question']).encode()).hexdigest()
        if row['id'] in history_ids or old_stem in history_stems or index.duplicate(norm):
            exclusions['historical_or_within_pool_duplicate'] += 1
            continue
        index.add(norm)
        available[row['subject']].append(row)
    groups = {}
    cursor = Counter()
    for split, n in (('test', 1024), ('dev', 256), ('train', 512)):
        chosen = []
        for subject, count in (('math', n // 2), ('chemistry', n // 4), ('physics', n // 4)):
            start = cursor[subject]
            block = available[subject][start:start + count]
            if len(block) != count:
                raise ValueError(f'Insufficient unused {subject}: {len(available[subject])}')
            chosen.extend(block)
            cursor[subject] += count
        groups[split] = sorted(chosen, key=lambda r: hashlib.sha256(f'{SEED}:{split}:{r["id"]}'.encode()).digest())
    args.output.mkdir(parents=True)
    (args.output / 'splits').mkdir()
    (args.output / 'provenance').mkdir()
    manifest = {'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
                'source_extension': 'New SuperGPQA chemistry/physics, unused GSM8K math; separate from the retired912 study.',
                'selection': 'Fixed hashes before inference; 50/25/25 mix; test reserved first. No model results used.',
                'science': {**pin, 'sha256': sha(source), 'license': 'ODC-BY; original referenced-source licenses also apply',
                            'url': 'https://huggingface.co/datasets/m-a-p/SuperGPQA',
                            'option_order': 'Preserved official order and labeled option; no distractors removed.'},
                'math_sha256': sha(previous.MATH), 'history_files': historical,
                'deduplication': 'Recorded IDs, archived normalized stems, and >=85% word-fivegram overlap with comparable length.',
                'exclusions': dict(exclusions), 'unused_pool': {k: len(v) for k, v in available.items()},
                'raw_examples_publishable': False, 'splits': {}}
    for split, rows in groups.items():
        path = args.output / 'splits' / f'{split}-questions.jsonl'
        write(path, rows)
        manifest['splits'][split] = {'n': len(rows), 'sha256': sha(path),
                                     'subjects': dict(Counter(r['subject'] for r in rows))}
        (args.output / (split + '-observations.jsonl')).touch()
        (args.output / (split + '-late-observations.jsonl')).touch()
    (args.output / 'splits/selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'provenance/historical-exclusions.json').write_text(json.dumps(
        {'ids': sorted(history_ids), 'normalized_stem_sha256': sorted(history_stems)}, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('registered_at_utc', 'unused_pool', 'exclusions', 'splits')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / '.cache/controller-semantic-followup/supergpqa')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
