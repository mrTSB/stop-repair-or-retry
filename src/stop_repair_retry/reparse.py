"""Reparse saved raw responses after an audited parser improvement."""
import argparse
from .core import ACTIONS, read_jsonl, walk, write_jsonl
from .generate import parse_object


def reparse(rows):
    for row in rows:
        for _, node in walk(row['tree']):
            p = parse_object(node.get('text', ''))
            answer = p.get('answer', '')
            c = str(p.get('confidence', '')).lower()
            node['answer'] = str(answer) if answer is not None else ''
            node['confidence'] = c if c in ('low', 'medium', 'high') else 'low'
            node['valid'] = answer is not None and bool(str(answer).strip()) and c in ('low', 'medium', 'high')
            for judge in node.get('judges', {}).values():
                action = str(parse_object(judge.get('text', '')).get('action', '')).lower()
                judge['valid'] = action in ('stop', *ACTIONS)
                judge['action'] = action if judge['valid'] else 'stop'
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    if args.input == args.output: parser.error('Output must differ from input to preserve raw parser results')
    write_jsonl(args.output, reparse(read_jsonl(args.input)))

if __name__ == '__main__': main()
