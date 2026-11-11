"""Offline regressions for the independently imported Modal worker module."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import patch

DRIVER = Path(__file__).resolve().parents[1] / 'experiments/llm-controller/modal_pilot.py'


class ModalPilotImportTests(unittest.TestCase):
    def test_shallow_remote_path_import_has_no_local_repository_dependency(self):
        class InertImage:
            @classmethod
            def debian_slim(cls, **kwargs):
                return cls()

            def pip_install(self, *args):
                return self

        class InertApp:
            def __init__(self, *args):
                pass

            def cls(self, **kwargs):
                return lambda cls: cls

        modal = types.ModuleType('modal')
        modal.Image = InertImage
        modal.App = InertApp
        modal.enter = lambda: lambda function: function
        modal.method = lambda: lambda function: function
        namespace = {'__name__': 'remote_worker_import_test', '__file__': '/root/modal_pilot.py'}
        source = DRIVER.read_text()
        with patch.dict('sys.modules', {'modal': modal}):
            exec(compile(source, '/root/modal_pilot.py', 'exec'), namespace)
        self.assertNotIn('ROOT', namespace)
        self.assertIn('Controller', namespace)
        for name in ('load', 'metadata', 'decide'):
            self.assertNotIn('ROOT', getattr(namespace['Controller'], name).__code__.co_names)
        # The client-only experiment import must stay inside main, never run on workers.
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('stop_repair_retry'))
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith('stop_repair_retry') for alias in node.names))


if __name__ == '__main__':
    unittest.main()
