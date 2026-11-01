"""Report trajectory quality, confidence calibration, and action transitions."""
import argparse
import json
from collections import Counter, defaultdict
from .core import ACTIONS, confidence, correct, read_jsonl, walk


def diagnose(rows):
    obs = defaultdict(Counter)
    transitions = defaultdict(Counter)
    invalid_solver = invalid_judge = judges = nodes = 0
    duration = 0.0
    for row in rows:
        tree, gold = row['tree'], row['answer']
        for path, node in walk(tree):
            nodes += 1
            invalid_solver += not node.get('valid', True)
            duration += node.get('latency_seconds', 0)
            state = int(correct(node, gold))
            obs[confidence(node)]['n'] += 1
            obs[confidence(node)]['correct'] += state
            for j in node.get('judges', {}).values():
                judges += 1
                invalid_judge += not j.get('valid', True)
                duration += j.get('latency_seconds', 0)
            for a in ACTIONS:
                child = path + ('/' if path else '') + a
                if child in tree:
                    ns = int(correct(tree[child], gold))
                    transitions[a][f'{state}_to_{ns}'] += 1
    return {'questions': len(rows), 'solver_calls': nodes, 'invalid_solver_calls': invalid_solver, 'judge_calls': judges, 'invalid_judge_calls': invalid_judge, 'cached_generation_seconds': duration, 'confidence': {k: dict(v) for k, v in obs.items()}, 'action_transitions': {k: dict(v) for k, v in transitions.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory_file')
    args = p.parse_args()
    print(json.dumps(diagnose(read_jsonl(args.trajectory_file)), indent=2))

if __name__ == '__main__': main()
