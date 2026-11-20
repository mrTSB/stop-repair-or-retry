"""Complete paired training/development branches using the unchanged old protocol."""

import argparse
from pathlib import Path

from margin_data import PREVIOUS_CODE
import extended_acquire as old
from transport import resilient


if __name__ == '__main__':
    import modal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--app',default='srr-hybrid-margin-20261119')
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error('Use one or two collection workers')
    worker=modal.Cls.from_name(args.app,'Ollama')()
    old.request=resilient(worker,args.data/'transport-errors.jsonl')
    args.backend='local'  # Inject the private worker directly; do not select the retired app.
    args.phase='full'
    for split in ('train','dev'):
        args.split=split
        old.collect(args)
