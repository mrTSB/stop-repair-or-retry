"""Regenerate paid neural observations across every source-by-subject stratum.

Run only after complete blind collection. Sampling uses fixed question hashes,
never correctness. This audit neither trains nor changes a controller.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from confirmation_common import (evidence, load_banks, models, run_policy, clear,
                                 read, sha, verify_fixed_prefix, verify_special_banks)


def run(args):
    if args.per_stratum <= 0:
        raise ValueError('Sample at least one question per stratum')
    frozen = json.loads((args.study / 'frozen.json').read_bytes())
    if frozen['evidence'] != evidence(args.data):
        raise ValueError('Frozen sources or fitting evidence changed')
    verify_fixed_prefix(frozen, args.data)
    verify_special_banks(frozen, args.medium)
    completed = json.loads((args.data / 'test-collection.json').read_bytes())
    if completed['completed'] != 912 or completed['freeze_sha256'] != sha(args.study / 'frozen.json'):
        raise ValueError('Complete every blind replay before this audit')
    if completed['neural_cache_sha256'] != sha(args.study / 'test-neural-cache.jsonl'):
        raise ValueError('Changed completed neural receipts')
    protocol = frozen['selected']['C']['protocol']
    reuse = Path(frozen['reuse_bank']['directory']) if 'reuse_bank' in frozen else None
    rows, trees, original, late, deeper, deep_reviews = load_banks(args.data, 'test', args.medium, protocol, reuse)
    selection = json.loads((args.data / 'splits/selection.json').read_bytes())
    panel_by_id = {qid: p for p, ids in selection['strata'].items() for qid in ids}
    strata = {}
    for row in rows:
        strata.setdefault((panel_by_id[row['id']], row['subject']), []).append(row)
    sample = []
    for key, group in sorted(strata.items()):
        chosen = sorted(group, key=lambda r: hashlib.sha256(('encoder-strata-2026112071:' + r['id']).encode()).hexdigest())[:args.per_stratum]
        sample.extend(chosen)
    planners = models(frozen, args.study)
    checks = []
    for family, model in planners.items():
        config = frozen['selected'][family]
        special = config['protocol'] in ('medium', 'root-medium', 'mentor-reuse', 'full-packet')
        bank = deeper if special else trees
        reviews = deep_reviews if special else late if config['protocol'] == 'late' else original
        for row in sample:
            safe = {k: row[k] for k in ('id', 'question', 'domain', 'subject')}
            safe['answer'] = ''
            clear(model)
            item = run_policy(family, safe, bank[row['id']], reviews, reviews, bank[row['id']], reviews, model, config)
            parts = item['path'].split('/') if item['path'] else []
            visited = [''] + ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
            paid_tree = {p: bank[row['id']][p] for p in visited}
            paid_reviews = {(row['id'], d['path']): reviews[row['id'], d['path']]
                            for d in item['decisions'] if d['queried']}
            descriptors = [(d['path'], None) for d in item['decisions']]
            if model.review_observer is not None:
                descriptors += [(p, n) for (_, p), n in paid_reviews.items()]
            for path, review in descriptors:
                node = paid_tree[path] if review is None else review
                key = model.classifier.key(safe, path, node, 'solver' if review is None else 'review')
                expected = model.classifier._cache.pop(key)
                actual = model.classifier.values(safe, path, paid_tree, review)
                assert actual[3] == expected[3]
                differences = [float(np.max(np.abs(x - y))) for x, y in zip(actual[:3], expected[:3])]
                assert all(np.allclose(x, y, atol=1e-4, rtol=1e-4) for x, y in zip(actual[:3], expected[:3]))
                checks.append({'family': family, 'id': row['id'], 'panel': panel_by_id[row['id']],
                               'subject': row['subject'], 'path': path, 'kind': 'solver' if review is None else 'review',
                               'tokens': actual[3], 'maximum_logit_difference': max(differences)})
            clear(model)
            replay = run_policy(family, safe, paid_tree, paid_reviews, paid_reviews, paid_tree, paid_reviews, model, config)
            # MPS may vary at the last floating-point bits; action, cost, and
            # visits must remain identical after a truly independent pass.
            for field in ('path', 'solver_tokens', 'judge_tokens', 'compute', 'verifier_tokens', 'verifier_calls'):
                assert replay[field] == item[field], (family, row['id'], field)
        print(f'Regenerated {family} across all six source-by-subject strata', flush=True)
    receipt = {'passed': True, 'freeze_sha256': sha(args.study / 'frozen.json'),
               'source_sha256': sha(__file__), 'neural_cache_sha256': completed['neural_cache_sha256'],
               'sampling': 'Fixed SHA256 ordering per source-by-subject stratum, no outcomes consulted',
               'per_stratum': args.per_stratum, 'sampled_questions': len(sample),
               'strata': {panel + '/' + subject: min(args.per_stratum, len(group)) for (panel, subject), group in strata.items()},
               'independent_encoder_regenerations': len(checks), 'references_provided_to_policy': False,
               'checks': checks}
    destination = args.study / 'encoder-panel-audit.json'
    if destination.exists():
        raise ValueError('Do not overwrite an independent audit receipt')
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'checks'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'data', 'medium'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--per-stratum', type=int, default=2)
    run(p.parse_args())
