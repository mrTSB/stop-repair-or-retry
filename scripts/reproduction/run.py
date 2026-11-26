"""Run frozen experiment code using a verified bundle's automatic path map."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from portable import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('script', help='Repository-relative Python script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'reproduction-bundle.json').read_bytes())
    script = root / args.script
    if not script.is_file() or not script.resolve().is_relative_to(root):
        parser.error('script must be an existing file in the bundle')
    script_string, root_string = str(script), str(root)
    install(root_string, manifest['path_mappings'])
    sys.path.insert(0, str(Path(script_string).parent))
    sys.path.insert(0, str(Path(root_string) / 'src'))
    sys.argv = [script_string, *args.arguments]
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    runpy.run_path(script_string, run_name='__main__')


if __name__ == '__main__':
    main()
