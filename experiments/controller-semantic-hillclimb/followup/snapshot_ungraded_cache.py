"""Register a separate pre-grading follow-up without changing its frozen parent.

Controller selection may use retired training/development data only. This tool
refuses a scored parent and preserves all912questions and every cached response,
without filtering on answers or predicted quality. Each follow-up controller is
frozen before its own replay. Both runs must be reported as sharing one panel.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from confirmation_common import read, sha, evidence, verify_fixed_prefix
from banks import committed


def prepare(args):
    frozen = json.loads((args.parent_study / 'frozen.json').read_bytes())
    if (args.parent_study / 'test').exists():
        raise ValueError('Cannot register this follow-up after parent grades are opened')
    if frozen['evidence'] != evidence(args.parent_data):
        raise ValueError('Parent frozen evidence changed')
    verify_fixed_prefix(frozen, args.parent_data)
    receipt = json.loads((args.parent_data / 'test-collection.json').read_bytes())
    if receipt['completed'] != 912 or receipt['freeze_sha256'] != sha(args.parent_study / 'frozen.json'):
        raise ValueError('Complete the entire ungraded parent replay before copying responses')
    for key, path in (('base_sha256', args.parent_data / 'test-observations.jsonl'),
                      ('late_sha256', args.parent_data / 'test-late-observations.jsonl'),
                      ('neural_cache_sha256', args.parent_study / 'test-neural-cache.jsonl')):
        if sha(path) != receipt[key]:
            raise ValueError('Parent completed cache changed')
    for split in ('train', 'dev'):
        if sha(args.fitting_data / 'splits' / (split + '-questions.jsonl')) != sha(args.parent_data / 'splits' / (split + '-questions.jsonl')):
            raise ValueError('Preserve exactly the retired fitting questions')
    rows = read(args.parent_data / 'splits/test-questions.jsonl')
    ids = {r['id'] for r in rows}
    required = {(qid, 'solver', path) for qid in ids for path in ('', 'repair', 'retry', 'repair/repair', 'retry/retry')}
    bank = committed(args.parent_data / 'test-observations.jsonl')
    if len(rows) != 912 or not required <= {(r['id'], r['kind'], r['path']) for r in bank}:
        raise ValueError('Preserve every fixed reference response for all912questions')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'splits').mkdir()
    for split in ('train', 'dev'):
        shutil.copy2(args.fitting_data / 'splits' / (split + '-questions.jsonl'),
                     args.output / 'splits' / (split + '-questions.jsonl'))
        for suffix in ('observations', 'late-observations', 'neural-observations'):
            p = args.fitting_data / (split + '-' + suffix + '.jsonl')
            if p.exists():
                shutil.copy2(p, args.output / p.name)
    shutil.copytree(args.parent_data / 'provenance', args.output / 'provenance')
    shutil.copy2(args.parent_data / 'splits/test-questions.jsonl', args.output / 'splits/test-questions.jsonl')
    for name in ('test-observations.jsonl', 'test-late-observations.jsonl'):
        shutil.copy2(args.parent_data / name, args.output / name)
    registration = {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                    'parent_freeze_sha256': sha(args.parent_study / 'frozen.json'),
                    'parent_study': str(args.parent_study), 'parent_data': str(args.parent_data),
                    'parent_collection_sha256': sha(args.parent_data / 'test-collection.json'),
                    'source_sha256': sha(__file__), 'parent_grades_opened': False,
                    'questions': 912, 'same_question_panel': True,
                    'selection_rule': 'Retired development data only; freeze new candidates before their own replay and before any parent/follow-up grading.',
                    'cache_rule': 'Copy all exact ungraded responses, never filter by correctness. Charge every policy use, including cache hits.',
                    'reporting_rule': 'Retain the original frozen study; these are the same912questions, not independent additional test examples.',
                    'reused': {name: sha(args.output / name) for name in ('test-observations.jsonl', 'test-late-observations.jsonl')}}
    manifest = json.loads((args.parent_data / 'splits/selection.json').read_bytes())
    manifest['secondary_registration'] = registration
    (args.output / 'splits/selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'provenance/secondary-ungraded-cache.json').write_text(json.dumps(registration, indent=2) + '\n')
    contract = {'questions': 912, 'paths': ['', 'repair', 'retry', 'repair/repair', 'retry/retry'],
                'source_sha256': sha(__file__), 'graded': False,
                'test_sha256': sha(args.output / 'splits/test-questions.jsonl'),
                'selection_sha256': sha(args.output / 'splits/selection.json'),
                'rule': registration['cache_rule'], 'parent_freeze_sha256': registration['parent_freeze_sha256']}
    (args.output / 'fixed-cache-contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    completion = {'questions': 912, 'solver_responses': 4560, 'graded': False,
                  'contract_sha256': sha(args.output / 'fixed-cache-contract.json'),
                  'response_bank_sha256': sha(args.output / 'test-observations.jsonl')}
    (args.output / 'fixed-cache-complete.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(registration), flush=True)


def neural(args):
    if (args.parent_study / 'test').exists() or (args.study / 'test').exists():
        raise ValueError('Import exact neural observations before either test is graded')
    target = json.loads((args.study / 'frozen.json').read_bytes())
    registration = json.loads((Path(target['data_directory']) / 'provenance/secondary-ungraded-cache.json').read_bytes())
    if registration['parent_freeze_sha256'] != sha(args.parent_study / 'frozen.json'):
        raise ValueError('Wrong parent study')
    source_receipt = json.loads((Path(registration['parent_data']) / 'test-collection.json').read_bytes())
    if sha(args.parent_study / 'test-neural-cache.jsonl') != source_receipt['neural_cache_sha256']:
        raise ValueError('Changed parent neural observations')
    expected = {family: sha(args.study / descriptor['observer_directory'] / 'recipe.json')
                for family, descriptor in target['models'].items()}
    selected = []
    for item in committed(args.parent_study / 'test-neural-cache.jsonl'):
        if item['recipe_sha256'] == expected.get(item['family']):
            selected.append(item)
    destination = args.study / 'test-neural-cache.jsonl'
    if destination.exists():
        raise ValueError('Do not overwrite a neural receipt bank')
    with destination.open('w') as stream:
        for item in selected:
            stream.write(json.dumps(item) + '\n')
    receipt = {'source_freeze_sha256': sha(args.parent_study / 'frozen.json'),
               'target_freeze_sha256': sha(args.study / 'frozen.json'),
               'source_cache_sha256': sha(args.parent_study / 'test-neural-cache.jsonl'),
               'target_cache_sha256': sha(destination), 'compatible_observations': len(selected),
               'target_cache_bytes': destination.stat().st_size,
               'rule': 'Exact full observer recipe hashes must match; raw predictions reused without outcome filtering. Every model pass is charged per policy.',
               'grades_opened': False}
    (args.study / 'neural-cache-import.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'neural'))
    p.add_argument('--parent-study', type=Path, required=True)
    for name in ('parent-data', 'fitting-data', 'output', 'study'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args()
    required = ('parent_data', 'fitting_data', 'output') if args.phase == 'prepare' else ('study',)
    if any(getattr(args, name) is None for name in required):
        p.error('Supply ' + ', '.join('--' + name.replace('_', '-') for name in required))
    (prepare if args.phase == 'prepare' else neural)(args)
