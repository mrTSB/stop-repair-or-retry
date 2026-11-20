"""Archive validated margin results with deterministic gzip and a checksum manifest."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from hillclimb import new_run
from study_common import BASELINES


def pack(args):
    audit=json.loads((args.output/'audit.json').read_bytes())
    if not audit['passed'] or audit['questions']!=800:raise ValueError('Complete the independent audit before packing')
    summary=json.loads((args.output/'test/summary.json').read_bytes())
    frozen=json.loads((args.output/'frozen.json').read_bytes())
    with new_run(args.destination) as stage:
        (stage/'data').mkdir();shutil.copytree(args.data/'splits',stage/'data/splits')
        shutil.copytree(args.data/'provenance',stage/'data/provenance')
        for p in args.data.glob('*-observations.jsonl'):
            with (stage/'data'/(p.name+'.gz')).open('wb') as target:
                with gzip.GzipFile(fileobj=target,mode='wb',mtime=0) as compressed:
                    compressed.write(p.read_bytes())
        shutil.copyfile(args.data/'test-collection.json',stage/'data/test-collection.json')
        for name in ('transport-errors.jsonl','collection-errors.jsonl'):
            p=args.data/name
            if p.exists():shutil.copyfile(p,stage/'data'/name)
        for name in ('frozen.json','audit.json','development-reproduction.json'):
            shutil.copyfile(args.output/name,stage/name)
        development=stage/'development';development.mkdir()
        root=args.data.parent
        for probe in sorted(root.glob('*probe')):
            if not (probe/'summary.json').exists():continue
            destination=development/probe.name;destination.mkdir()
            for path in sorted(probe.glob('*.json')):shutil.copyfile(path,destination/path.name)
        for name in ('review-pilot-summary.json','routing-oracle-bounds.json','late-oracle-bounds.json',
                     'real-model-smoke.json','contextual-model-smoke.json','data-quality.json'):
            if (root/name).exists():shutil.copyfile(root/name,development/name)
        if (root/'review-pilot.jsonl').exists():
            (development/'review-pilot.jsonl.gz').write_bytes(gzip.compress((root/'review-pilot.jsonl').read_bytes(),mtime=0))
        verifier=root/'neural/verifier-v1'
        if verifier.exists():
            destination=development/'neural-training';destination.mkdir()
            for name in ('training-summary.json','contract.json','oof-probabilities.npz'):
                shutil.copyfile(verifier/name,destination/name)
            for fold in ('fold-0','fold-1','fold-2','full'):
                shutil.copyfile(verifier/fold/'recipe.json',destination/(fold+'-recipe.json'))
        if (root/'provenance').exists():shutil.copytree(root/'provenance',development/'source-snapshots')
        shutil.copytree(args.output/'models',stage/'models')
        if (args.output/'observers').exists():shutil.copytree(args.output/'observers',stage/'observers')
        shutil.copytree(args.output/'test',stage/'test')
        stats=summary['methods'];best=summary['highest_baseline']
        lines=['# Fresh accuracy-margin evaluation','',
               'All eight methods were frozen before evaluating the same 800 previously unused study questions. '
               'The compact solver checkpoint and the two additional solver-call budget are shared. '
               'A is a POMDP, B is an independent LLM controller, and C combines POMDP planning with paid LLM reviews.','',
               '| Method | Correct / 800 | Accuracy (%) | Mean raw tokens | Active-parameter compute proxy |',
               '|---|---:|---:|---:|---:|']
        for name in (*BASELINES,'A','B','C'):
            s=stats[name];tokens=s['mean_solver_tokens']+s['mean_judge_tokens']+s.get('mean_verifier_tokens',0.)
            lines.append(f'| {name} | {s["correct"]} | {100*s["accuracy"]:.2f} | {tokens:.1f} | {s["mean_compute"]:.1f} |')
        lines+=['',f'The highest observed fixed baseline is **{best}**. The confirmed targets use percentage points: '
                'A and B each need at least +10 over the highest baseline; C needs at least +10 over the higher of A/B, '
                'with +15 a stretch target. A clear C-versus-B win also requires paired 95% intervals favoring C for accuracy and compute.','',
                '| Criterion | Passed |','|---|---|']
        for name,passed in summary['criteria'].items():lines.append(f'| {name} | {"Yes" if passed else "No"} |')
        lines+=['',f'All required criteria passed: **{summary["all_required_passed"]}**.','',
                'Intervals are paired, subject-stratified bootstrap intervals with 5,000 resamples. '
                'The highest baseline and highest A/B are recomputed inside each resample. The full intervals are in `test/summary.json`.','',
                'The compute proxy is `(4 × solver input/output tokens + 3 × reviewer input/output tokens '
                '+ verifier parameters in billions × verifier padded token slots) / 3.09`. '
                'The frozen solver is Qwen3.5-4B and the shared reviewer is Qwen3.5-35B-A3B (35B total, 3B activated parameters). '
                'All acquired reviews are charged, including reviews recommending Stop. Reused review text in a solver prompt is also charged. '
                'This proxy does not measure wall-clock time, memory, energy, or offline collection/training cost. '
                'Total-parameter sensitivity is retained in the summary.','',
                'If selected, the small non-generative verifier supplies noisy observations to the POMDP; '
                'it does not generate answers or choose actions. Its input text and padded computation slots '
                'are reported separately, and every pass is charged. The compact solver remains frozen.','',
                'Training uses 672 retired questions and development uses 240 disjoint retired questions. '
                'The fresh study panel contains 400 GSM8K official training-source questions, 200 MMLU-Pro chemistry questions, '
                'and 200 physics questions. This is a study holdout, not the official GSM8K test split. '
                'The historical exclusion snapshot and exact split hashes are included. Prior test panels are explicitly retired '
                'into training/development and are not claimed as new validation.','',
                'The audit checks 6,400 outcomes against raw solver/review receipts and repeats 4,800 adaptive decisions '
                'with changed references and only paid observations. Feature caches are cleared before these replays. '
                'Completed malformed/truncated responses are retained. Transport-only retries are logged separately; '
                'a completed poor answer is never retried by the transport wrapper.','',
                'The four hidden POMDP states combine persistent question difficulty with current-answer correctness. '
                'A grouped, cross-fitted observation classifier supplies evidence; Bayesian filtering updates the state belief. '
                'C may plan for acquiring a review at a future checkpoint. Both B and C may use paid review feedback in a final repair '
                'when selected by development; the submitted answer always comes from the compact solver.','',
                'Frozen controller configurations:','', '```json',json.dumps(frozen['selected'],indent=2),'```','']
        (stage/'README.md').write_text('\n'.join(lines))
        manifest={str(p.relative_to(stage)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
                  for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Packed validated results: {args.destination}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    pack(parser.parse_args())
