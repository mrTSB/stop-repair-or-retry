"""Generate complete paired trajectory trees with an optional local MLX judge."""
import json
import hashlib
import os
from pathlib import Path
import re
import time
import random
import urllib.error
import urllib.request

from .core import ACTIONS, read_jsonl, write_jsonl

SOLVER = '''Solve the problem carefully. Return a JSON object with keys "reasoning", "answer", and "confidence". The answer must be a short final answer. Confidence must be exactly low, medium, or high. Do not include other text.\n\nProblem: {question}'''
REPAIR = '''Check the solution below and revise it if needed. Return a JSON object with keys "reasoning", "answer", and "confidence". The answer must be a short final answer. Confidence must be exactly low, medium, or high. Do not include other text.\n\nProblem: {question}\n\nCurrent solution: {solution}'''
JUDGE = '''Choose exactly one action: stop (submit current answer), repair (check current solution), or retry (solve again independently). You have {remaining} solver calls left. Maximize probability of a correct final answer minus {penalty} per 1000 total tokens, including judge and solver tokens. Return only JSON: {{"action": "stop|repair|retry"}}.\n\n{context}'''


def parse_object(text):
    # Some models wrap JSON in markdown or emit a preamble.
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict): return obj
        except json.JSONDecodeError:
            pass
    # Recover only explicitly named, short scalar fields when the model emits
    # invalid JSON (e.g. unquoted confidence or LaTeX escapes in reasoning).
    # Never infer an answer from free-form reasoning or incomplete fields.
    recovered = {}
    for field in ('answer', 'confidence', 'action'):
        pattern = r'"' + field + r'"\s*:\s*(?:"([^"\r\n]*)"|([a-zA-Z]+|[+-]?(?:\d+(?:\.\d+)?|\.\d+)))\s*(?=,|\}|\s*"(?:answer|confidence|action)"\s*:)'
        match = re.search(pattern, text)
        if match:
            recovered[field] = match.group(1) if match.group(1) is not None else match.group(2)
    return recovered


class MLXBackend:
    def __init__(self, model, max_tokens=512, temperature=0.7, seed=0):
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as e:
            raise RuntimeError('Install MLX support with: uv sync --extra mlx') from e
        self.mx = mx
        mx.random.seed(seed)
        self.model, self.tokenizer = load(model)
        self.sampler = make_sampler(temp=temperature)
        self.max_tokens = max_tokens

    def seed(self, seed):
        self.mx.random.seed(seed)

    def generate(self, prompt):
        from mlx_lm import stream_generate
        tokenizer = self.tokenizer
        try:
            formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        except (AttributeError, ValueError, TypeError):
            formatted = prompt
        start = time.perf_counter()
        segments, last = [], None
        for chunk in stream_generate(self.model, tokenizer, prompt=formatted, max_tokens=self.max_tokens, sampler=self.sampler):
            segments.append(chunk.text)
            last = chunk
        elapsed = time.perf_counter() - start
        text = ''.join(segments)
        # stream_generate reports prompt and cumulative generation token counts.
        return {'text': text, 'input_tokens': last.prompt_tokens if last else len(tokenizer.encode(formatted)), 'output_tokens': last.generation_tokens if last else 0, 'latency_seconds': elapsed}


class OllamaBackend:
    """Use a running local Ollama server, including its Apple GPU backend."""
    def __init__(self, model, max_tokens=512, temperature=0.7, seed=0, url='http://127.0.0.1:11434'):
        self.model, self.max_tokens, self.temperature, self.url = model, max_tokens, temperature, url.rstrip('/')
        self.rng = random.Random(seed)

    def seed(self, seed):
        self.rng.seed(seed)

    def generate(self, prompt):
        body = json.dumps({'model': self.model, 'prompt': prompt, 'stream': False, 'format': 'json', 'think': False, 'options': {'temperature': self.temperature, 'num_predict': self.max_tokens, 'seed': self.rng.randrange(2**31)}}).encode()
        request = urllib.request.Request(self.url + '/api/generate', data=body, headers={'Content-Type': 'application/json'})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Ollama returned HTTP {exc.code}: {exc.read().decode()}') from exc
        return {'text': result.get('response', ''), 'input_tokens': result.get('prompt_eval_count', 0), 'output_tokens': result.get('eval_count', 0), 'latency_seconds': time.perf_counter() - start}


def solve(backend, question, previous=None):
    prompt = SOLVER.format(question=question) if previous is None else REPAIR.format(question=question, solution=previous.get('text', ''))
    result = backend.generate(prompt)
    parsed = parse_object(result['text'])
    c = str(parsed.get('confidence', '')).lower()
    answer = parsed.get('answer', '')
    valid = answer is not None and bool(str(answer).strip()) and c in ('low', 'medium', 'high')
    result.update(answer=str(answer) if answer is not None else '', confidence=c if c in ('low', 'medium', 'high') else 'low', valid=valid)
    return result


def judge(backend, row, path, mode, penalty):
    tree = row['tree']
    parts = path.split('/') if path else []
    paths = [''] + ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
    if mode == 'full':
        context = 'Problem: ' + row['question'] + '\nHistory:\n' + '\n'.join(f'{p or "initial"}: {tree[p]["text"]}' for p in paths)
    else:
        from .policies import features
        context = f'Observable features (confidence, calls used, previous action, answer changed): {features(tree, path)}'
    prompt = JUDGE.format(remaining=2-len(parts), penalty=penalty, context=context)
    result = backend.generate(prompt)
    action = str(parse_object(result['text']).get('action', '')).lower()
    result['valid'] = action in ('stop', *ACTIONS)
    result['action'] = action if result['valid'] else 'stop'
    return result


def generate_rows(questions, backend, output, judge_backend=None, penalty=0.01, seed=0, config=None, resume=False):
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f'{output} already exists; use --resume to continue safely')
    rows = read_jsonl(output) if output.exists() else []
    if len(rows) > len(questions):
        raise ValueError('Saved output contains more questions than the input')
    for old, question in zip(rows, questions):
        if any(old.get(k) != question.get(k) for k in ('id', 'question', 'answer')) or old.get('generation_config') != config:
            raise ValueError('Saved output does not match the input order or generation configuration')
    for question in questions[len(rows):]:
        question_seed = int.from_bytes(hashlib.sha256(f'{seed}:{question["id"]}'.encode()).digest()[:4], 'big')
        if hasattr(backend, 'seed'):
            backend.seed(question_seed)
        if judge_backend is not None and judge_backend is not backend and hasattr(judge_backend, 'seed'):
            judge_backend.seed(question_seed ^ 0x5A5A5A5A)
        row = {'id': question['id'], 'question': question['question'], 'answer': question['answer'], 'tree': {}, 'judge_penalty': penalty if judge_backend else None, 'generation_config': config}
        tree = row['tree']
        tree[''] = solve(backend, row['question'])
        def expand(path, depth):
            if depth == 2: return
            for action in ACTIONS:
                child = path + ('/' if path else '') + action
                tree[child] = solve(backend, row['question'], tree[path] if action == 'repair' else None)
                expand(child, depth + 1)
        expand('', 0)
        if judge_backend:
            for path, node in tree.items():
                if path and len(path.split('/')) >= 2: continue
                node['judges'] = {mode: judge(judge_backend, row, path, mode, penalty) for mode in ('full', 'compact')}
        rows.append(row)
        # Atomic replacement preserves all previously completed questions on interruption.
        temporary = output.with_name(output.name + '.tmp')
        write_jsonl(temporary, rows)
        os.replace(temporary, output)
        print(f'Generated {len(rows)}/{len(questions)}: {row["id"]}', flush=True)
    return rows
