"""Load a frozen observer and run one TRAIN input with its reference removed."""

import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/controller-semantic-hillclimb/followup/round2'))
import development_math_specialization  # Establish archived model imports.


def main():
    import torch
    torch.set_num_threads(2)
    folder = ROOT / 'runs/controller-semantic-followup/round2'
    selected = json.loads((folder / 'development-math-specialization-v1/summary.json').read_bytes())['selected_A']
    model = pickle.loads(Path(selected['model_path']).read_bytes())
    observer = model.math.classifier
    observer.device = 'cpu'
    rows = [json.loads(line) for line in (folder / 'data/splits/train-questions.jsonl').read_bytes().splitlines()]
    row = next(row for row in rows if row['subject'] == 'math')
    observations = (json.loads(line) for line in (folder / 'data/train-observations.jsonl').read_bytes().splitlines())
    node = next(item['response'] for item in observations
                if item['id'] == row['id'] and item['kind'] == 'solver' and item['path'] == '')
    observer._cache = {}
    values = observer.values({**row, 'answer': ''}, '', {'': node})
    print(json.dumps({'passed': True, 'observer_class': type(observer).__module__ + '.' + type(observer).__name__,
                      'output_shapes': [list(value.shape) for value in values[:3]],
                      'token_slots': int(values[3]), 'device': 'cpu', 'training_example': True,
                      'reference_supplied_to_encoder': False, 'test_used': False}), flush=True)


if __name__ == '__main__':
    main()
