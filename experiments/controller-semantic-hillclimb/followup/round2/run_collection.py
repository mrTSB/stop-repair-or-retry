"""Prioritize complete plain pairs, then finish all ordinary review pairs."""

import argparse
import importlib.util
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'dev'), required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    # The historical project also has a module named collect. Load this
    # continuation explicitly without replacing the archived module.
    path = Path(__file__).with_name('collect.py')
    spec = importlib.util.spec_from_file_location('round2_paired_collector', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.mode = 'plain'
    module.run(args)
    args.mode = 'ordinary'
    module.run(args)
