import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import concrete_checkpoint as checkpoint


class CheckpointTests(unittest.TestCase):
    def test_changed_completed_artifact_is_rejected(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            item = folder/'summary.json'
            item.write_text('{"correct":153}')
            (folder/'complete.json').write_text(json.dumps({'artifacts': {'summary.json': checkpoint.sha(item)}}))
            # Isolate path rendering; verification still uses the actual bytes.
            from unittest.mock import patch
            with patch.object(checkpoint, 'ROOT', folder):
                checkpoint.sealed(folder)
                item.write_text('{"correct":205}')
                with self.assertRaises(ValueError): checkpoint.sealed(folder)

    def test_missing_or_duplicate_question_and_compute_difference_rejected(self):
        rows = [{'id': str(i), 'correct': int(i < 153), 'compute': 1000+i} for i in range(256)]
        result = checkpoint.metrics(rows)
        self.assertEqual(result['correct'], 153)
        checkpoint.same(result, result)
        with self.assertRaises(ValueError): checkpoint.metrics(rows[:-1])
        with self.assertRaises(ValueError): checkpoint.metrics([*rows[:-1], rows[0]])
        with self.assertRaises(ValueError): checkpoint.same(result, {**result, 'mean_compute': result['mean_compute']-1})


if __name__ == '__main__': unittest.main()
