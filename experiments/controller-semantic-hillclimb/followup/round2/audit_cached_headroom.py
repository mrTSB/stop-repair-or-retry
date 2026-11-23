"""Reproduce optimistic answer availability on already reused development data.

This is a diagnostic bound on the recorded output pool, ignoring compute and
requiring hindsight correctness. It is never controller accuracy, a guarantee
about uncollected paths, or a fresh-test result. No test file is opened.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

import concrete_checkpoint as files

ROOT = Path(__file__).resolve().parents[4]
SCORING = ROOT/'results/cached-tree-dataset/scoring.py'
SCORING_SHA = '357e5f659b6bf3011e32c190ea79e5793fbfd3b42919c8274a0b56c7d3f849fa'
PLAIN = {'','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry'}
BANKS = ('data/dev-observations.jsonl','data/dev-late-observations.jsonl',
    'development-retained-path-complete-v1/dev-observations.jsonl',
    'development-expanded-mentor-v1/dev-observations.jsonl',
    'development-complete-actions-v1/dev.jsonl')


def run(base):
    if files.sha(SCORING) != SCORING_SHA: raise ValueError('Changed original frozen scorer')
    module = importlib.util.spec_from_file_location('headroom_frozen_scoring',SCORING)
    scoring = importlib.util.module_from_spec(module); module.loader.exec_module(scoring)
    label_path = base/'data/splits/dev-questions.jsonl'
    rows = [json.loads(line) for line in label_path.open()]
    if (len(rows) != 256 or len({r['id'] for r in rows}) != 256
            or Counter(r['subject'] for r in rows) != {'math':128,'chemistry':64,'physics':64}):
        raise ValueError('Require the unchanged reused development cohort')
    labels = {r['id']:r for r in rows}
    groups = {name:{r['id']:[] for r in rows} for name in ('plain','strong_complete','ordinary','late_once','late_twice')}
    evidence = {str(label_path.relative_to(ROOT)):files.sha(label_path)}
    for bank in BANKS:
        path = base/bank
        evidence[str(path.relative_to(ROOT))] = files.sha(path)
        for line in path.open():
            item = json.loads(line)
            if item['id'] not in labels or item['kind'] != 'solver': continue
            action = item['path']
            if bank == BANKS[0]: group = 'plain' if action in PLAIN else 'ordinary'
            elif bank == BANKS[1]: group = 'late_twice' if action.startswith('guided/') else 'late_once'
            else: group = 'strong_complete'
            groups[group][item['id']].append(item['response'])
    stages = (('plain',('plain',)),('plain+strong+complete',('plain','strong_complete')),
        ('plain+strong+complete+ordinary+ordinary_late_once',('plain','strong_complete','ordinary','late_once')),
        ('plain+strong+complete+ordinary+ordinary_late_once+ordinary_late_twice',tuple(groups)))
    reports = {}
    for name,keys in stages:
        available = {r['id']:any(scoring.correct(n,r['answer']) for key in keys for n in groups[key][r['id']]) for r in rows}
        report = {}
        for subject in ('all','math','chemistry','physics'):
            panel = [r for r in rows if subject == 'all' or r['subject'] == subject]
            hits = sum(available[r['id']] for r in panel)
            report[subject] = {'n':len(panel),'correct_answer_available':hits,'fraction':hits/len(panel)}
        reports[name] = report
    return {'schema':'reused-development-recorded-answer-headroom-audit-v1',
        'source_sha256':files.sha(__file__),'scoring_source_sha256':SCORING_SHA,
        'scope':'Optimistic hindsight availability over recorded outputs, ignoring compute; reused DEV, not policy performance',
        'reports':reports,'evidence':evidence,'test_used':False,'controller_accuracy_claim':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'runs/controller-semantic-followup/round2')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.base)
    if args.output.exists():
        if json.loads(args.output.read_bytes()) != result: raise ValueError('Preserve previous bound audit')
    else: args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print({'reports':result['reports'],'controller_accuracy_claim':False,'test_used':False},flush=True)


if __name__ == '__main__': main()
