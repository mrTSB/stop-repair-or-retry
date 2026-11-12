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
            for driver in (DRIVER, DRIVER.with_name('modal_full_evaluation.py')):
                with self.subTest(driver=driver.name):
                    namespace = {'__name__': 'remote_worker_import_test', '__file__': '/root/' + driver.name}
                    exec(compile(driver.read_text(), namespace['__file__'], 'exec'), namespace)
                    self.assertNotIn('ROOT', namespace)
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


    def test_full_backend_exactly_matches_pilot(self):
        pilot = ast.parse(DRIVER.read_text())
        full = ast.parse(DRIVER.with_name('modal_full_evaluation.py').read_text())
        pilot_class = next(n for n in pilot.body if isinstance(n, ast.ClassDef))
        full_class = next(n for n in full.body if isinstance(n, ast.ClassDef))
        self.assertEqual(ast.dump(pilot_class, include_attributes=False),
                         ast.dump(full_class, include_attributes=False))

    def test_watchdog_syntax_and_full_dev_freeze_test_order(self):
        tree = ast.parse(DRIVER.with_name('modal_full_evaluation.py').read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'watchdog_source'
                                  for t in n.targets))
        call = assignment.value
        self.assertEqual(call.func.attr, 'replace')
        source = ast.literal_eval(call.func.value).replace(
            ast.literal_eval(call.args[0]), ast.literal_eval(call.args[1]))
        compile(source, '<watchdog syntax only>', 'exec')
        self.assertIn("'--profile','gancone','--yes'", source)
        stages = []

        class Stages(ast.NodeVisitor):
            def visit_Call(self, node):
                f = node.func
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'experiment':
                    if f.attr in ('run', 'evaluate', 'freeze_run', 'completed_records'):
                        stages.append((f.attr, node))
                self.generic_visit(node)

        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        Stages().visit(main)
        self.assertEqual([s[0] for s in stages],
                         ['run', 'evaluate', 'freeze_run', 'run', 'completed_records', 'evaluate'])
        self.assertEqual(ast.literal_eval(stages[0][1].args[3]), 'dev')
        self.assertEqual(ast.literal_eval(stages[3][1].args[3]), 'test')
        self.assertIsNone(ast.literal_eval(stages[0][1].args[4]))
        self.assertIsNone(ast.literal_eval(stages[3][1].args[4]))
        self.assertTrue(any(k.arg == 'freeze' for k in stages[3][1].keywords))

if __name__ == '__main__':
    unittest.main()
