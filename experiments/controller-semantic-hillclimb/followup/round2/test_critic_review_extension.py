from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import development_critic_review_extended as extension


class CriticExtensionTests(unittest.TestCase):
    def test_complete_cache_requires_each_registered_forward(self):
        plan = {'id': 'q', 'kind': 'solver', 'path': 'guided', 'input_sha256': 'i'}
        record = {'feature_plan': [plan]}; key = extension.diagnostic.inputs.fingerprint(plan)
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), registration_sha256='r'); feature = args.output/'features'; feature.mkdir()
            with self.assertRaisesRegex(ValueError, 'Incomplete'): extension.restored(args, record, True)
            extension.io.append(feature/'attempts.jsonl', {'key': key, 'registration_sha256': 'r'})
            with self.assertRaisesRegex(ValueError, 'Unresolved'): extension.restored(args, record)
            extension.io.append(feature/'observations.jsonl', {'key': key, 'registration_sha256': 'r',
                'outputs': [[0.]*4, [0.]*8, [0.]*8], 'slots': 100})
            self.assertEqual(len(extension.restored(args, record, True)), 1)
            extension.io.append(feature/'attempts.jsonl', {'key': key, 'registration_sha256': 'r'})
            with self.assertRaisesRegex(ValueError, 'Changed'): extension.restored(args, record)


if __name__ == '__main__': unittest.main()
