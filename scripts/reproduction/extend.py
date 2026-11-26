"""Repeat an archived register/fit or register/replay recipe in a new directory.

The archived registration supplies its input paths automatically. Source files
and completed source runs remain byte-identical. Use --set NAME=VALUE to replace
an input path when extending with a separately prepared dataset or diagnostic.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('fit', 'replay'), default='fit')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists():
        parser.error('output must be a new directory')
    recipe = json.loads(args.recipe.read_bytes())
    inputs = {**recipe['paths'], 'output': str(output)}
    for item in args.set:
        name, separator, value = item.partition('=')
        if not separator or name not in inputs:
            parser.error('--set must name an input in the archived recipe')
        inputs[name] = str(Path(value).resolve())
    arguments = [part for name, value in inputs.items() for part in ('--' + name.replace('_', '-'), value)]
    command = [sys.executable, str(root / 'scripts/reproduction/run.py'), '--root', str(root), str(args.script)]
    subprocess.run([*command, 'register', *arguments], cwd=root, check=True)
    fingerprint = hashlib.sha256((output / 'registration.json').read_bytes()).hexdigest()
    subprocess.run([*command, args.phase, *arguments, '--registration-sha256', fingerprint], cwd=root, check=True)


if __name__ == '__main__':
    main()
