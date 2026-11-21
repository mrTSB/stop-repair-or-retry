"""Wait for completed A development summaries and run CPU-only training audits.

This helper only launches audit_a_refit.run. It cannot train a neural model,
evaluate a development/test policy, or acquire responses. The schedule binds
the audit sources before waiting and refuses subsequent source changes.
"""

import argparse
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import time

import audit_a_refit as audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed(path):
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError:
        if time.time() - path.stat().st_mtime < 10:
            return False
        raise ValueError('Completed summary is not valid JSON: ' + str(path))
    if 'selected_A' not in value:
        raise ValueError('Completed summary lacks a selected A: ' + str(path))
    if value.get('test_used') or value.get('fresh_data_used'):
        raise ValueError('This audit schedule permits development-selected A only')
    return True


def run(args):
    if args.output.exists():
        raise ValueError('Use a new training-audit schedule directory')
    summaries = [p.resolve() for p in args.summary]
    if len(set(summaries)) != len(summaries):
        raise ValueError('Duplicate scheduled A summary')
    # Walk the static imports of supported lazy audit branches, including older
    # feature/scoring code. Already-loaded unrelated drivers/tests are not roots.
    dependencies = set(audit.project_sources(declare_supported=True)) | {Path(__file__).resolve()}
    sources = {str(p): sha(p) for p in sorted(dependencies)}
    registration = {'schema': 'a-training-refit-schedule-v1', 'training_only': True,
                    'summaries': [str(p) for p in summaries], 'source_sha256': sources,
                    'declared_supported_runtime_modules': list(audit.SUPPORTED_RUNTIME_MODULES),
                    'train_questions_sha256': sha(args.data / 'splits/train-questions.jsonl'),
                    'no_neural_inference': True, 'no_development_or_test_records': True,
                    'atol': args.atol, 'rtol': args.rtol}
    args.output.mkdir(parents=True)
    (args.output / 'registration.json').write_text(json.dumps(registration, indent=2) + '\n')
    pending = list(enumerate(summaries)); results = []
    while pending:
        ready = [(i, path) for i, path in pending if completed(path)]
        for index, path in ready:
            if any(sha(Path(source)) != digest for source, digest in sources.items()):
                raise ValueError('Registered training-audit sources changed while waiting')
            if sha(args.data / 'splits/train-questions.jsonl') != registration['train_questions_sha256']:
                raise ValueError('Registered training questions changed while waiting')
            target = args.output / f'{index:02d}-{path.parent.name}.json'
            child = Namespace(data=args.data, summary=path, output=target, registry=args.registry,
                              base=args.base, science=args.science, refined=args.refined, fusion=args.fusion,
                              atol=args.atol, rtol=args.rtol)
            receipt = audit.run(child)
            unexpected = set(receipt['source_files']) - set(sources)
            changed = [source for source, digest in sources.items() if sha(Path(source)) != digest]
            if unexpected or changed:
                receipt['passed'] = False
                receipt['schedule_dependency_error'] = {'undeclared_sources': sorted(unexpected), 'changed_sources': changed}
                target.write_text(json.dumps(receipt, indent=2) + '\n')
                raise ValueError('Audit dependency closure changed; the emitted receipt was marked failed')
            results.append({'summary': str(path), 'audit': str(target.resolve()), 'audit_sha256': sha(target),
                            'model_sha256': receipt['selection']['model_sha256'], 'passed': receipt['passed']})
            pending.remove((index, path))
            status = {'registration_sha256': sha(args.output / 'registration.json'), 'completed': results,
                      'pending': [str(p) for _, p in pending], 'training_only': True}
            temporary = args.output / 'status.tmp'
            temporary.write_text(json.dumps(status, indent=2) + '\n')
            temporary.replace(args.output / 'status.json')
        if pending:
            if args.no_wait:
                raise ValueError('Requested summaries are incomplete; use waiting mode after they finish')
            print(json.dumps({'waiting_for': [str(p) for _, p in pending], 'completed_audits': len(results)}), flush=True)
            time.sleep(args.poll_seconds)
    print(json.dumps({'completed_audits': len(results), 'output': str(args.output.resolve())}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, action='append', required=True)
    for name in ('data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('registry', 'base', 'science', 'refined', 'fusion'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--no-wait', action='store_true')
    parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--rtol', type=float, default=1e-10)
    args = parser.parse_args()
    args.registry = args.registry or args.data.parent
    args.base = args.base or args.registry / 'development-repairability-v1'
    args.science = args.science or args.registry / 'development-science-option-v1'
    args.refined = args.refined or args.registry / 'development-refined-observer-v1'
    args.fusion = args.fusion or args.registry / 'development-science-fusion-v1'
    if not 1 <= args.poll_seconds <= 60 or min(args.atol, args.rtol) < 0 or max(args.atol, args.rtol) > 1e-8:
        parser.error('Poll in1–60seconds; tolerances in0–1e-8')
    run(args)
