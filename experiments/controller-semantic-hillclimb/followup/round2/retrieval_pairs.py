"""Training-only paired pilot of textbook feedback for compact-solver repair.

This is a new controller feedback channel, not a pure retry-gating change.
Question selection is fixed by ID hash before any new response is generated.
Retrieved exposition depends only on the original question, never its answer,
solver output, or correctness. All completed responses are retained.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary, committed
from evidence import Retriever, DEFAULT as CORPUS
from scoring import correct
from transfer_variants import actor
from transport import resilient
import extended_acquire as old

APP = 'srr-followup-solver-20261120'
SEED = 2026112147
PROTOCOL = 'reference-free-textbook-repair-v1'
COUNTS = {'math': 64, 'chemistry': 32, 'physics': 32}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select(rows):
    selected = []
    for subject, count in COUNTS.items():
        pool = sorted((r for r in rows if r['subject'] == subject),
                      key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).digest())
        if len(pool) < count:
            raise ValueError('Insufficient training questions for the fixed cohort')
        selected.extend(pool[:count])
    if len({r['id'] for r in selected}) != 128:
        raise ValueError('Repeated question IDs')
    return sorted(selected, key=lambda r: hashlib.sha256(f'{SEED}:order:{r["id"]}'.encode()).digest())


def seed_for(qid):
    return int(hashlib.sha256(f'{old.SEED}:{qid}:repair'.encode()).hexdigest()[:8], 16) % 2**31


def feedback(background):
    return ('REFERENCE TEXTBOOK EXPOSITION (may be incomplete or irrelevant):\n' + background
            if background else '')


def prepare(args):
    rows, trees, _, _ = ordinary(args.data, 'train')
    selected = select(rows)
    retriever = Retriever(args.corpus)
    source_manifest = json.loads((args.corpus / 'manifest.json').read_bytes())
    if source_manifest['sha256'] != retriever.sha256:
        raise ValueError('Textbook source manifest and passages differ')
    original_contract_path = args.data / 'train-plain-contract.json'
    original_contract = json.loads(original_contract_path.read_bytes())
    if (original_contract['continuation_cap'], original_contract['solver_digest']) != (256, old.MODEL_DIGEST):
        raise ValueError('Original plain controls used different compact settings')
    controls = {}
    for row in selected:
        qid = row['id']
        controls[qid] = {p: fingerprint(trees[qid][p]) for p in ('', 'repair', 'repair/repair')}
    contract = {
        'source_sha256': sha(__file__), 'retriever_source_sha256': sha(sys.modules['evidence'].__file__),
        'actor_source_sha256': sha(sys.modules['transfer_variants'].__file__),
        'repair_prompt_source_sha256': sha(sys.modules[old.repair_prompt.__module__].__file__),
        'selection_seed': SEED, 'selection': [{'id': r['id'], 'subject': r['subject']} for r in selected],
        'selection_rule': 'Fixed hash128new training questions,64math32chem32physics; no response-conditioned selection.',
        'training_questions_sha256': sha(args.data / 'splits/train-questions.jsonl'),
        'control_response_sha256': controls, 'source_manifest': source_manifest,
        'control_collection_contract_sha256': sha(original_contract_path),
        'corpus_sha256': retriever.sha256, 'source_manifest_sha256': sha(args.corpus / 'manifest.json'),
        'solver_digest': old.MODEL_DIGEST, 'solver_temperature': .65, 'solver_cap': 256,
        'context_length': 4096, 'seed_rule': 'Original logical repair seed for each question.',
        'feedback_protocol': PROTOCOL, 'llm_reviews': 0, 'app': APP, 'max_cloud_gpus': 2,
        'reference_answers_supplied': False, 'split': 'training only', 'test_used': False,
        'change': 'Existing compact repair template receives reference-free textbook exposition in its optional feedback field.',
        'compute_accounting': 'Solver input/output tokens include factual feedback. CPU retrieval latency is reported separately; no free LLM calls.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    cp = args.output / 'contract.json'
    if cp.exists() and json.loads(cp.read_bytes()) != contract:
        raise ValueError('Changed retrieval pilot contract')
    cp.write_text(json.dumps(contract, indent=2) + '\n')
    ep = args.output / 'evidence.jsonl'
    if ep.exists():
        evidence = {r['id']: r for r in committed(ep)}
        if set(evidence) != {r['id'] for r in selected}:
            raise ValueError('Incomplete registered retrieval evidence')
    else:
        evidence = {}
        for row in selected:
            safe = {**row, 'answer': ''}
            started = time.monotonic()
            background, passages = retriever.context(safe)
            elapsed = time.monotonic() - started
            if row['subject'] == 'math' and background:
                raise ValueError('Math controls must reuse their exact no-feedback repair')
            evidence[row['id']] = {'id': row['id'], 'background': background,
                'background_sha256': hashlib.sha256(background.encode()).hexdigest(),
                'passages': passages, 'retrieval_seconds': elapsed, 'corpus_sha256': retriever.sha256}
        ep.write_text(''.join(json.dumps(evidence[r['id']]) + '\n' for r in selected))
    # Check all registered feedback hashes before any cloud request. The evidence
    # file is private because it includes public textbook excerpts and IDs.
    for row in selected:
        e = evidence[row['id']]
        if e['corpus_sha256'] != retriever.sha256 or hashlib.sha256(e['background'].encode()).hexdigest() != e['background_sha256']:
            raise ValueError('Changed registered evidence')
    receipt = {'contract_sha256': sha(cp), 'evidence_sha256': sha(ep), 'questions': 128,
               'questions_with_evidence': sum(bool(e['background']) for e in evidence.values()),
               'reference_answers_supplied': False, 'test_used': False}
    rp = args.output / 'registration.json'
    if rp.exists() and json.loads(rp.read_bytes()) != receipt:
        raise ValueError('Changed registered retrieval evidence file')
    rp.write_text(json.dumps(receipt, indent=2) + '\n')
    return selected, trees, evidence, receipt


def run(args):
    rows, trees, evidence, registration = prepare(args)
    if args.prepare_only:
        print(json.dumps(registration), flush=True)
        return
    import modal
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    if tags.get(old.MODEL) != old.MODEL_DIGEST:
        raise ValueError('Changed compact solver')
    path = args.output / 'train.jsonl'
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        saved = [json.loads(line) for line in stream]
        known = {r['id']: r for r in saved}
        if len(known) != len(saved) or set(known) - {r['id'] for r in rows}:
            raise ValueError('Unexpected or duplicate retrieval response')
        lock = threading.Lock()
        done = 0

        def one(row):
            nonlocal done
            qid = row['id']; safe = {**row, 'answer': ''}
            background = evidence[qid]['background']
            prompt = old.repair_prompt(safe, trees[qid][''], feedback(background))
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            if qid not in known:
                reuse = not background
                node = trees[qid]['repair'] if reuse else actor(request, safe, prompt, seed_for(qid), 256)
                item = {'id': qid, 'kind': 'solver', 'path': 'repair', 'protocol': PROTOCOL,
                        'response': node, 'exact_control_reused': reuse,
                        'background_sha256': evidence[qid]['background_sha256']}
                # Validate the exact empty-feedback control before committing it.
                if (node['prompt_sha256'], node['seed'], node.get('output_cap', 256 if reuse else None), node['model_digest'], node['temperature']) != (
                        prompt_hash, seed_for(qid), 256, old.MODEL_DIGEST, .65):
                    raise ValueError('Incompatible retrieval repair/control response')
                with lock:
                    stream.write(json.dumps(item) + '\n'); stream.flush(); known[qid] = item
            item = known[qid]; node = item['response']
            if (node['prompt_sha256'], node['seed'], node.get('output_cap', 256 if item['exact_control_reused'] else None), node['model_digest'], node['temperature']) != (
                    prompt_hash, seed_for(qid), 256, old.MODEL_DIGEST, .65):
                raise ValueError('Changed saved retrieval repair')
            if item['background_sha256'] != evidence[qid]['background_sha256']:
                raise ValueError('Saved response used different feedback')
            with lock:
                done += 1
                if done % 16 == 0 or done == 128:
                    print(f'Textbook-feedback training pilot {done}/128', flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(one, rows))
    if len(known) != 128:
        raise ValueError('Incomplete pilot; never summarize a partial selected cohort')

    def metrics(group):
        methods = {}
        for method, paths in [('Stop', ('',)), ('Repair once', ('', 'repair')),
                              ('Repair twice', ('', 'repair', 'repair/repair')),
                              ('Textbook repair', ('', 'textbook'))]:
            rights, totals = [], []
            for r in group:
                nodes = [known[r['id']]['response'] if p == 'textbook' else trees[r['id']][p] for p in paths]
                rights.append(int(correct(nodes[-1], r['answer'])))
                totals.append(sum(n['input_tokens'] + n['output_tokens'] for n in nodes))
            methods[method] = {'n': len(group), 'correct': sum(rights), 'accuracy': sum(rights) / len(group),
                               'mean_solver_tokens': sum(totals) / len(group),
                               'mean_compute': 4 * sum(totals) / len(group) / 3.09}
        wins = losses = 0
        for r in group:
            first = correct(trees[r['id']]['repair'], r['answer'])
            second = correct(known[r['id']]['response'], r['answer'])
            wins += bool(second and not first); losses += bool(first and not second)
        return {'methods': methods, 'wins_vs_repair_once': wins, 'losses_vs_repair_once': losses,
                'mean_extra_input_tokens_vs_plain_repair': sum(known[r['id']]['response']['input_tokens'] - trees[r['id']]['repair']['input_tokens'] for r in group) / len(group),
                'mean_cpu_retrieval_seconds': sum(evidence[r['id']]['retrieval_seconds'] for r in group) / len(group)}

    summary = {'split': 'training only', 'test_used': False, 'n': 128,
               'protocol': PROTOCOL, 'new_feedback_channel': True,
               'registration': registration, 'bank_sha256': sha(path),
               'exact_controls_reused': sum(v['exact_control_reused'] for v in known.values()),
               'all': metrics(rows),
               'subjects': {s: metrics([r for r in rows if r['subject'] == s]) for s in COUNTS},
               'notes': 'Exploratory matched training pilot, not a fresh-data performance claim. CPU retrieval cost reported separately.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())
