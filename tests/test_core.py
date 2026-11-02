import unittest
import tempfile
import random
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from stop_repair_retry.core import Model, correct, write_jsonl
from stop_repair_retry.cli import POLICIES, evaluate
from stop_repair_retry.generate import generate_rows, parse_object
from stop_repair_retry.policies import TabularQ, majority, simulate, tune


def row(i, initial='0', repair='1', retry='0'):
    def node(answer, conf='medium'):
        return {'answer': answer, 'confidence': conf, 'valid': True, 'input_tokens': 10, 'output_tokens': 5, 'latency_seconds': .1}
    tree = {'': node(initial)}
    for a in ('repair', 'retry'):
        tree[a] = node(repair if a == 'repair' else retry)
        for b in ('repair', 'retry'):
            tree[a + '/' + b] = node(repair if b == 'repair' else retry)
    return {'id': str(i), 'answer': '1', 'question': 'test', 'tree': tree}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(i, initial='1' if i % 2 else '0') for i in range(20)]
        self.model = Model.fit(self.rows)
        self.q = TabularQ(self.rows)

    def test_bayes_observation_probabilities_and_horizon(self):
        b = self.model.initial_belief('medium')
        self.assertTrue(0 < b < 1)
        self.assertAlmostEqual(sum(self.model.update(b, 'repair', o)[1] for o in ('low', 'medium', 'high')), 1)
        self.assertEqual(self.model.values(b, 0, 0), {'stop': b})
        self.assertEqual(max(self.model.values(b, 1, 0), key=self.model.values(b, 1, 0).get), 'repair')

    def test_costs_and_forced_budget(self):
        r = simulate(self.rows[0], 'repair', self.model, self.q, .1)
        self.assertEqual(r['actions'], ['repair', 'repair'])
        self.assertEqual(r['tokens'], 45)
        self.assertEqual(r['correct'], 1)
        self.assertEqual(simulate(self.rows[0], 'stop', self.model, self.q, .1)['tokens'], 15)

    def test_judge_cost_included_even_when_stopping(self):
        sample = row('x')
        sample['tree']['']['judges'] = {'full': {'action': 'stop', 'input_tokens': 100, 'output_tokens': 3, 'latency_seconds': 1}}
        r = simulate(sample, 'judge_full', self.model, self.q, .1)
        self.assertEqual(r['tokens'], 118)
        self.assertEqual(r['judge_calls'], 1)

    def test_majority_and_schedule(self):
        sample = row('x', initial='1', retry='0')
        self.assertEqual(majority(sample)['correct'], 0)
        chosen = tune(self.rows, 'schedule', self.model, self.q, 0)
        self.assertEqual(chosen[0], 'repair')

    def test_parse_bad_json_and_markdown(self):
        self.assertEqual(parse_object('```json\n{"action":"retry"}\n```')['action'], 'retry')
        self.assertEqual(parse_object('invalid'), {})
        self.assertEqual(parse_object('{"answer": "40 cm"\n"confidence": "high"}'), {'answer': '40 cm', 'confidence': 'high'})
        self.assertEqual(parse_object('{"reasoning":"\\(2+2\\)", "answer":"4", "confidence": high}'), {'answer': '4', 'confidence': 'high'})
        self.assertEqual(parse_object('{"answer":"4'), {})

    def test_numeric_grading_is_exact_and_does_not_extract_from_prose(self):
        for prediction, expected in [('1,000.0', '1000'), ('$0.5', '1/2'), ('.5', '0.5'), ('-2.00', '-2'), ('40 cm', '40'), ('52%', '52'), ('24 children more', '24'), ('(1/3)*2*(2*12)', '16'), ('w = 5', '5')]:
            self.assertTrue(correct({'answer': prediction}, expected))
        self.assertFalse(correct({'answer': 'about 5'}, '5'))
        self.assertFalse(correct({'answer': '5 apples and 6 oranges'}, '5'))
        self.assertFalse(correct({'answer': '__import__("os").getpid()'}, '5'))
        self.assertFalse(correct({'answer': '50'}, '5'))
        self.assertFalse(correct({'answer': '5', 'valid': False}, '5'))

    def test_generated_tree_covers_all_controller_paths(self):
        class FakeBackend:
            def generate(self, prompt):
                text = '{"action":"stop"}' if 'Choose exactly one action' in prompt else '{"answer":"1","confidence":"high"}'
                return {'text': text, 'input_tokens': 10, 'output_tokens': 2, 'latency_seconds': 0}
        with tempfile.TemporaryDirectory() as d:
            rows = generate_rows([{'id': 'x', 'question': 'test', 'answer': '1'}], FakeBackend(), Path(d) / 'tree.jsonl', FakeBackend())
        self.assertEqual(len(rows[0]['tree']), 7)
        self.assertEqual(sum(len(n.get('judges', {})) for n in rows[0]['tree'].values()), 6)
        model, qmodel = Model.fit(rows), TabularQ(rows)
        for policy in ('stop', 'repair', 'retry', 'one_step', 'supervised', 'pomdp', 'judge_full', 'judge_compact'):
            self.assertEqual(simulate(rows[0], policy, model, qmodel, .01)['correct'], 1)

    def test_resume_preserves_completed_rows_and_rejects_config_changes(self):
        class Backend:
            def seed(self, value): self.value = value
            def generate(self, prompt):
                return {'text': '{"answer":"1","confidence":"high"}', 'input_tokens': 1, 'output_tokens': 1}
        questions = [{'id': str(i), 'question': 'test', 'answer': '1'} for i in range(2)]
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'tree.jsonl'
            first = generate_rows(questions[:1], Backend(), output, config={'model': 'a'})
            resumed = generate_rows(questions, Backend(), output, config={'model': 'a'}, resume=True)
            self.assertEqual(resumed[0], first[0])
            self.assertEqual(len(resumed), 2)
            with self.assertRaises(ValueError):
                generate_rows(questions, Backend(), output, config={'model': 'b'}, resume=True)
            with self.assertRaises(FileExistsError):
                generate_rows(questions, Backend(), output)

    def test_per_question_seed_is_reproducible_after_resume(self):
        class Backend:
            def seed(self, value): self.rng = random.Random(value)
            def generate(self, prompt):
                return {'text': '{"answer":"%s","confidence":"high"}' % self.rng.randrange(100000), 'input_tokens': 1, 'output_tokens': 1}
        questions = [{'id': str(i), 'question': 'test', 'answer': '1'} for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            complete = generate_rows(questions, Backend(), root/'full', seed=42, config={'x': 1})
            generate_rows(questions[:1], Backend(), root/'partial', seed=42, config={'x': 1})
            resumed = generate_rows(questions, Backend(), root/'partial', seed=42, config={'x': 1}, resume=True)
            self.assertEqual(complete, resumed)

    def test_complete_evaluation_includes_all_policies(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split in ('train', 'dev', 'test'):
                rows = [row(split + str(i), initial=str(i % 2)) for i in range(4)]
                for sample in rows:
                    sample['judge_penalty'] = .01
                    for path, node in sample['tree'].items():
                        if not path or '/' not in path:
                            node['judges'] = {mode: {'action': 'stop', 'input_tokens': 2, 'output_tokens': 1} for mode in ('full', 'compact')}
                write_jsonl(root / f'{split}.jsonl', rows)
            args = Namespace(train=root/'train.jsonl', dev=root/'dev.jsonl', test=root/'test.jsonl', output=root/'results', penalty=.01, policies=','.join(POLICIES))
            with redirect_stdout(StringIO()):
                evaluate(args)
            import json
            summary = json.loads((root/'results'/'summary.json').read_text())
            self.assertEqual({r['policy'] for r in summary['results']}, set(POLICIES))
            self.assertEqual(len((root/'results'/'per_question.jsonl').read_text().splitlines()), 4 * len(POLICIES))

    def test_interactive_episode_nodes_and_metrics(self):
        from stop_repair_retry.interactive import _node, _simulate
        root = _node(0, 'medium', 10, 2, .1, 2, 1)
        repair = _node(100, 'high', 20, 3, .2, 4, 0)
        retry = _node(0, 'low', 30, 4, .3, 6, 0)
        sample = {'id': 'interactive-1', 'answer': '1', 'tree': {'': root, 'repair': repair, 'retry': retry}}
        outcome = _simulate(sample, 'repair', self.model, self.q, .01)
        self.assertEqual(outcome['correct'], 1)
        self.assertEqual(outcome['reward'], 1)
        self.assertEqual(outcome['tokens'], 35)
        self.assertEqual(outcome['env_steps'], 6)

    def test_interactive_action_prompt_receives_remaining_full_attempt_budget(self):
        from stop_repair_retry.interactive import _run_steps

        class Adapter:
            done = False
            score = 0
            observation = 'state'
            task = {'task': 'boil'}
            def valid_actions(self): return ['go']
            def task_description(self): return 'task'
            def step(self, action):
                self.done = True
                return 'done', True

        class Backend:
            def generate(self, prompt):
                self.prompt = prompt
                return {'text': '{"action":"go","confidence":"high"}', 'input_tokens': 1, 'output_tokens': 1}

        backend = Backend()
        _run_steps(backend, Adapter(), [], 1, 'solve', attempt_budget=6, steps_before=0)
        self.assertIn('Actions remaining in this attempt: 6', backend.prompt)
        _run_steps(backend, Adapter(), [], 4, 'repair', attempt_budget=6, steps_before=2)
        self.assertIn('Actions remaining in this attempt: 4', backend.prompt)

    def test_alfworld_goal_extraction_uses_reset_instruction(self):
        from stop_repair_retry.interactive import _extract_alfworld_goal
        observation = 'Welcome to TextWorld\nYou are in a room.\n\nYour task is to: put a pencil in shelf.'
        self.assertEqual(_extract_alfworld_goal(observation), 'Your task is to: put a pencil in shelf.')

    def test_science_state_change_menu_hides_unrelated_focus_actions(self):
        from stop_repair_retry.interactive import _science_task_actions
        actions = ['look around', 'go hallway', 'focus on air', 'focus on tin', 'look at tin']
        filtered = _science_task_actions('boil', 'Your task is to boil tin.', actions)
        self.assertIn('focus on tin', filtered)
        self.assertNotIn('focus on air', filtered)
        self.assertIn('go hallway', filtered)
        self.assertIn('look at tin', filtered)

    def test_scienceworld_split_sample_is_disjoint_and_stratified(self):
        from stop_repair_retry.interactive import scienceworld_tasks
        train = scienceworld_tasks('train', 42, 4)
        dev = scienceworld_tasks('dev', 42, 4)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(dev), 4)
        self.assertFalse({row['id'] for row in train} & {row['id'] for row in dev})
        self.assertEqual(len({row['task'] for row in train}), 4)

if __name__ == '__main__': unittest.main()
