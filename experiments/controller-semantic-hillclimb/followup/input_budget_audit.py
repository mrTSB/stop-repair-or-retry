"""Audit training-only input truncation with aggregate, text-free receipts."""

import argparse
import hashlib
import json
from pathlib import Path


def run(args):
    from transformers import AutoTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error()
    metadata = [json.loads(s) for s in args.examples.read_text().splitlines()]
    roots = [r for r in metadata if r['kind'] == 'solver' and not r['path']]
    selected = sorted(roots, key=lambda r: hashlib.sha256(r['id'].encode()).digest())[:256]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=False)
    marker = '\n\nBACKGROUND EXPOSITION (may be incomplete):\n'
    marker_length = len(tokenizer.encode(marker, add_special_tokens=False))
    records = []
    for r in selected:
        values = tokenizer(r['left'], r['right'], max_length=512, truncation='longest_first',
                           return_token_type_ids=True)
        left = sum(v == 0 for v in values['token_type_ids']) - 2
        right = sum(v == 1 for v in values['token_type_ids']) - 1
        question, present, context = r['left'].partition(marker)
        qtokens = len(tokenizer.encode(question, add_special_tokens=False))
        full_right = len(tokenizer.encode(r['right'], add_special_tokens=False))
        records.append({'science': r['id'].startswith('mmlu-pro-'), 'background_present': bool(present),
                        'background_kept_tokens': max(0, left - qtokens - marker_length) if present else 0,
                        'lost_question_tokens': max(0, qtokens - left),
                        'right_retained_fraction': right / max(1, full_right)})
    report = {'selection': 'First256training solver roots by fixed question hash; no outcome used.',
              'fresh_data_used': False, 'examples_sha256': hashlib.sha256(args.examples.read_bytes()).hexdigest(),
              'groups': {}}
    for label, group in [('all', records), ('science', [r for r in records if r['science']]),
                          ('math', [r for r in records if not r['science']])]:
        grounded = [r for r in group if r['background_present']]
        report['groups'][label] = {
            'n': len(group), 'retrieved_background_n': len(grounded),
            'question_truncated_fraction': sum(r['lost_question_tokens'] > 2 for r in group) / len(group),
            'mean_right_retained_fraction': sum(r['right_retained_fraction'] for r in group) / len(group),
            'retrieved_background_fully_lost_fraction': (sum(r['background_kept_tokens'] == 0 for r in grounded) / len(grounded)
                                                        if grounded else None),
            'mean_retrieved_background_tokens_retained': (sum(r['background_kept_tokens'] for r in grounded) / len(grounded)
                                                          if grounded else None)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('examples', 'model', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
