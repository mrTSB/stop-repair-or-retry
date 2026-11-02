"""Episode-level stop/repair/retry experiments for interactive text worlds.

The benchmark action labels are interventions at one fixed checkpoint:
stop and submit the current partial score, repair by continuing the same live
episode, or retry by resetting the same task and spending a fresh episode
budget. Each task is reset to the same official instance for paired branches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import random
import statistics
import time
from pathlib import Path
from typing import Any

from .cli import bootstrap_interval
from .core import ACTIONS, Model, confidence, correct, read_jsonl, write_jsonl
from .generate import MLXBackend, parse_object
from .policies import TabularQ, best, choose

POLICIES = (
    "stop", "repair", "retry", "tuned_fixed", "threshold", "one_step",
    "supervised", "judge_full", "judge_compact", "hybrid",
)
SPLIT_SIZES = {"train": 300, "dev": 100, "test": 500}
ALFWORLD_EPISODE_STEPS = 50
MODEL_DEFAULT = (
    ".cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit/"
    "snapshots/4f83f8f146fdf28b512a06562b671d7af4fab457"
)


def _task_seed(seed: int, task_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{task_id}".encode()).digest()[:4], "big")


def _science_task_actions(task_name: str, task_description: str, actions: list[str]) -> list[str]:
    if task_name not in ("boil", "freeze", "change-the-state-of-matter-of") and not task_name.startswith("chemistry-mix"):
        return actions
    goal_sentence = task_description.lower().split(".", 1)[0]
    target_words = set(re.findall(r"[a-z]+", goal_sentence)) - {
        "your", "task", "is", "to", "the", "of", "a", "an", "and", "or",
        "on", "in", "matter", "change", "state", "boil", "freeze", "mix",
        "chemistry", "please", "first", "focus", "substance",
    }
    if not target_words:
        return actions
    return [
        action for action in actions
        if not action.startswith("focus on ")
        or any(word in action.lower().split() for word in target_words)
    ]


def _sample_stratified(groups: dict[str, list[Any]], size: int, seed: int) -> list[Any]:
    """Round-robin task families, then sample variation IDs without replacement."""
    rng = random.Random(seed)
    keys = sorted(groups)
    shuffled = {key: list(groups[key]) for key in keys}
    for values in shuffled.values():
        rng.shuffle(values)
    selected: list[Any] = []
    cursor = 0
    while len(selected) < size:
        key = keys[cursor % len(keys)]
        values = shuffled[key]
        if values:
            selected.append((key, values.pop()))
        elif not any(shuffled.values()):
            raise ValueError(f"Requested {size} items from only {len(selected)} available")
        cursor += 1
    return selected


def scienceworld_tasks(split: str, seed: int, size: int) -> list[dict[str, Any]]:
    manifest = Path(".cache/scienceworld-variations.json")
    if manifest.exists():
        split_sets = json.loads(manifest.read_text())
    else:
        from scienceworld import ScienceWorldEnv

        env = ScienceWorldEnv(envStepLimit=100)
        try:
            split_sets = {}
            for task in env.task_names:
                env.load(task, 0, "")
                split_sets[task] = {
                    name: sorted(int(item) for item in method())
                    for name, method in (
                        ("train", env.getVariationsTrain),
                        ("dev", env.getVariationsDev),
                        ("test", env.getVariationsTest),
                    )
                }
        finally:
            env._gateway.shutdown()
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(split_sets, indent=2) + "\n")
    groups: dict[str, list[int]] = {}
    for task, values in split_sets.items():
        sets = {name: set(ids) for name, ids in values.items()}
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            if sets[left] & sets[right]:
                raise ValueError(f"ScienceWorld {task} variation leakage: {left}/{right}")
        groups[task] = values[split]
    chosen = _sample_stratified(groups, min(size, sum(map(len, groups.values()))), seed)
    return [
        {"id": f"scienceworld-{split}-{task}-{variation:04d}", "task": task, "variation": variation}
        for task, variation in chosen
    ]


def _alfworld_config(data_root: Path) -> dict[str, Any]:
    root = str(data_root.resolve())
    return {
        "dataset": {
            "data_path": f"{root}/json_2.1.1/train",
            "eval_id_data_path": f"{root}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{root}/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {"domain": f"{root}/logic/alfred.pddl", "grammar": f"{root}/logic/alfred.twl2"},
        "env": {
            "type": "AlfredTWEnv", "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6], "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "general": {"training_method": "dqn", "use_cuda": False},
        "rl": {"training": {"max_nb_steps_per_episode": 50}},
    }


def _alfworld_env(data_root: Path, split: str, game_files: list[str]):
    """Create the official ALFWorld wrapper with an explicit selected list."""
    os.environ["ALFWORLD_DATA"] = str(data_root.resolve())
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    mode = {"train": "train", "dev": "eval_in_distribution", "test": "eval_out_of_distribution"}[split]

    class SelectedGames(AlfredTWEnv):
        def collect_game_files(self, verbose=False):
            self.game_files = list(game_files)
            self.num_games = len(self.game_files)

    return SelectedGames(_alfworld_config(data_root), train_eval=mode)


def _alfworld_game_files(split: str, data_root: Path) -> list[str]:
    index = Path(".cache") / f"alfworld-{split}-games.json"
    if index.exists():
        return json.loads(index.read_text())
    from alfworld.agents.environment import get_environment

    mode = {"train": "train", "dev": "eval_in_distribution", "test": "eval_out_of_distribution"}[split]
    scanner = get_environment("AlfredTWEnv")(_alfworld_config(data_root), train_eval=mode)
    files = sorted(scanner.game_files)
    del scanner
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(files, indent=2) + "\n")
    return files


def alfworld_tasks(split: str, seed: int, size: int, data_root: Path) -> list[dict[str, Any]]:
    data_root = data_root.resolve()
    for rel in ("logic/alfred.pddl", "logic/alfred.twl2", "json_2.1.1/train", "json_2.1.1/valid_seen", "json_2.1.1/valid_unseen"):
        if not (data_root / rel).exists():
            raise FileNotFoundError(f"Missing ALFWorld asset {data_root / rel}; run scripts/prepare_interactive_benchmarks.sh")
    os.environ["ALFWORLD_DATA"] = str(data_root)
    all_paths = _alfworld_game_files(split, data_root)
    rng = random.Random(seed)
    chosen = sorted(all_paths) if size >= len(all_paths) else sorted(rng.sample(all_paths, size))
    return [
        {"id": f"alfworld-{split}-{hashlib.sha256(path.encode()).hexdigest()[:12]}", "game_file": path}
        for path in chosen
    ]


class ScienceWorldAdapter:
    def __init__(self):
        from scienceworld import ScienceWorldEnv

        # Preserve the simulator's standard 100-action hard limit; the study
        # uses a shorter, explicit attempt budget inside that environment.
        self.env = ScienceWorldEnv(envStepLimit=100)
        self.task: dict[str, Any] | None = None
        self.observation = ""
        self.done = False
        self.score = 0
        self.history: list[tuple[str, str]] = []
        self.legal_actions: list[str] = []
        self.objects: list[str] = []
        self.templates: list[str] = []

    def reset(self, task: dict[str, Any]) -> None:
        if self.task != task:
            self.task = task
            self.env.load(task["task"], task["variation"], "")
        self.observation, info = self.env.reset()
        self.initial_observation = self.observation
        self.legal_actions = []
        self.objects = list(self.env.getPossibleObjects())
        self.templates = list(self.env.getPossibleActions())
        self.raw_score = int(info.get("score", 0))
        self.score = max(0, min(100, self.raw_score))
        self.done = False
        self.history = []

    def valid_actions(self) -> list[str]:
        # The wrapper recomputes all legal combinations on every step. Expose
        # grounded commands from the official grammar and measure invalid calls.
        if self.done:
            return []
        task_text = self.task_description().lower()
        obs_text = self.observation.lower()
        mentioned = [obj for obj in self.objects if obj.lower() in obs_text or obj.lower() in task_text]
        actions = []
        for template in self.templates:
            slots = template.count("OBJ")
            if slots == 0:
                actions.append(template)
            elif slots == 1:
                actions.extend(template.replace("OBJ", obj) for obj in mentioned[:12])
            elif slots == 2:
                for first in mentioned[:8]:
                    for second in mentioned[:8]:
                        actions.append(template.replace("OBJ", first, 1).replace("OBJ", second, 1))
        actions = list(dict.fromkeys(actions))
        actions = _science_task_actions(self.task["task"], self.task_description(), actions)
        if len(actions) <= 48:
            return actions
        task_words = set(task_text.replace('.', ' ').split())
        obs_words = set(obs_text.replace('.', ' ').split())
        relevant = [action for action in actions if sum(word in action.lower() for word in task_words if len(word) > 3)]
        if len(relevant) < 24:
            relevant = actions
        if len(relevant) <= 48:
            return relevant
        ranked = sorted(
            relevant,
            key=lambda action: (
                -sum(word in action.lower() for word in task_words if len(word) > 3),
                -sum(word in action.lower() for word in obs_words if len(word) > 3),
                action,
            ),
        )
        # Always keep common navigation/information commands.
        pinned = [a for a in actions if a in ('look around', 'inventory', 'task', 'wait')]
        return list(dict.fromkeys(pinned + ranked[:48]))[:48]

    def task_description(self) -> str:
        return self.env.taskdescription()

    def step(self, action: str) -> tuple[str, bool]:
        self.observation, _reward, self.done, info = self.env.step(action)
        self.legal_actions = []
        self.raw_score = int(info["score"])
        self.score = max(0, min(100, self.raw_score))
        self.done = bool(self.done or self.raw_score < 0)
        return self.observation, self.done

    def close(self) -> None:
        self.env._gateway.shutdown()


class AlfWorldAdapter:
    def __init__(self, data_root: Path, split: str, game_files: list[str], active_game: str):
        data_root = data_root.resolve()
        os.environ["ALFWORLD_DATA"] = str(data_root)
        self.raw = _alfworld_env(data_root, split, game_files)
        self.raw.game_files = list(game_files)
        self.raw.num_games = 1
        self.env = self.raw.init_env(batch_size=1)
        self.game_file = active_game
        self.split = split
        self.observation = ""
        self.goal_text = ""
        self.done = False
        self.score = 0
        self.history: list[tuple[str, str]] = []

    @staticmethod
    def _first(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    def reset(self) -> None:
        observations, _infos = self.env.reset()
        self.observation = str(self._first(observations))
        self.goal_text = _extract_alfworld_goal(self.observation)
        self.infos = _infos
        self.done = False
        self.score = 0
        self.history = []

    def valid_actions(self) -> list[str]:
        return list(self.infos.get("admissible_commands", [[]])[0])

    def task_description(self) -> str:
        # ALFWorld emits the instruction at reset, then replaces observations
        # with action feedback. Preserve the original instruction across steps.
        return self.goal_text

    def step(self, action: str) -> tuple[str, bool]:
        observations, rewards, dones, infos = self.env.step([action])
        self.infos = infos
        self.observation = str(self._first(observations))
        self.done = bool(self._first(dones))
        won = bool(self._first(infos.get("won", [False])))
        self.score = 100 if won else 0
        return self.observation, self.done

    def close(self) -> None:
        self.env.close()


def _extract_alfworld_goal(initial_observation: str) -> str:
    for line in initial_observation.splitlines():
        if line.lower().startswith("your task is"):
            return line.strip()
    return initial_observation.split("\n", 1)[0].strip()


def _node(score: float, confidence_value: str, input_tokens: int, output_tokens: int,
          latency: float, action_steps: int, invalid_actions: int) -> dict[str, Any]:
    success = int(score >= 100)
    return {
        "answer": str(success), "valid": True, "confidence": confidence_value,
        "score": score / 100.0, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "latency_seconds": latency, "env_steps": action_steps, "invalid_actions": invalid_actions,
    }


def _action_prompt(adapter: Any, history: list[tuple[str, str]], style: str, max_actions: int) -> str:
    allowed = adapter.valid_actions()
    # ScienceWorld's template-level menus can contain hundreds of legal object
    # combinations. Use the action list supplied by its own simulator API.
    actions = "\n".join(f"- {action}" for action in allowed)
    recent = history[-8:]
    trace = "\n".join(f"Action: {a}\nObservation: {o}" for a, o in recent) or "(no actions yet)"
    if isinstance(adapter, AlfWorldAdapter):
        family = Path(adapter.game_file).parts[-3].split("-")[0]
        task_guides = {
            "look_at_obj_in_light": "Find the named lamp and object. Turn on the lamp, then examine the exact object with that lamp.",
            "pick_clean_then_place_in_recep": "Take the exact requested object, clean it at a sink, then put it in the named destination.",
            "pick_heat_then_place_in_recep": "Take the exact requested object, heat it with an appropriate appliance, then put it in the named destination.",
            "pick_cool_then_place_in_recep": "Take the exact requested object, cool it in a fridge, then put it in the named destination.",
            "pick_and_place_simple": "Take the exact requested object, go to the named destination, and put it there.",
            "pick_two_obj_and_place": "Take two instances of the requested object, then put them in the named destination.",
            "pick_and_place_with_movable_recep": "Move the named receptacle to the destination, then put the requested object into it.",
        }
        guidance = (
            "Keep the original task goal fixed even when the latest observation is only feedback. "
            + task_guides.get(family, "Follow the exact object and destination named in the task.")
            + " Do not manipulate unrelated objects or return an object to where it started."
        )
    else:
        task_name = adapter.task.get("task", "")
        if task_name in ("boil", "freeze", "change-the-state-of-matter-of"):
            task_guide = "Focus on the exact substance named by the task, then use the appropriate heating or cooling equipment to change its state."
        elif task_name.startswith("chemistry-mix"):
            task_guide = "Use only the exact ingredient names in the task; locate them, bring them together, and mix them in a suitable container."
        elif task_name.startswith("find-"):
            task_guide = "Inspect candidate objects and use their properties to identify one matching the requested category; do not manipulate the agent or scenery."
        else:
            task_guide = "Identify the exact target and desired outcome in the task before choosing an action."
        guidance = (
            "Keep the task goal fixed and choose an action that advances it. " + task_guide +
            " Never focus on or move unrelated furniture, the agent, or air."
        )
    return (
        "You control an agent in an interactive text environment. Follow the task and issue one legal action. "
        "Return only a JSON object with string fields action and confidence; confidence must be low, medium, or high.\n"
        f"Strategy: {style}\nTask: {adapter.task_description()}\nGuidance: {guidance}\n"
        f"Current observation:\n{adapter.observation}\nRecent history:\n{trace}\n"
        f"Actions remaining in this attempt: {max_actions}\n"
        f"Available actions (use one exactly):\n{actions}"
    )


def _judge_prompt(row: dict[str, Any], mode: str, penalty: float) -> str:
    root = row["tree"][""]
    context = row.get("checkpoint_context", "")
    if mode == "full":
        visible = context
    else:
        visible = (
            f"Task family: {row.get('task_family', 'unknown')}\n"
            f"Actions already used: {root['env_steps']}\n"
            f"Last action confidence: {root['confidence']}\n"
            f"Checkpoint observation: {row.get('checkpoint_observation', '')}"
        )
    return (
        "Choose exactly one controller action: stop, repair, or retry. Stop submits the current partial task score. "
        "Repair continues from the current simulator state. Retry resets this same task and starts a fresh attempt. "
        f"Maximize task success minus {penalty} per 1000 generated tokens. Return only JSON {{\"action\":\"stop|repair|retry\"}}.\n"
        + visible
    )


def _llm_action(backend: Any, adapter: Any, history: list[tuple[str, str]], style: str,
                remaining: int) -> tuple[str, str, dict[str, Any]]:
    output = backend.generate(_action_prompt(adapter, history, style, remaining))
    obj = parse_object(output["text"])
    action = str(obj.get("action", "")).strip()
    conf = str(obj.get("confidence", "low")).lower()
    if conf not in ("low", "medium", "high"):
        conf = "low"
    output["valid"] = bool(action)
    output["action"] = action
    output["confidence"] = conf
    return action, conf, output


def _run_steps(backend: Any, adapter: Any, history: list[tuple[str, str]], limit: int,
               style: str, attempt_budget: int, steps_before: int = 0) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    input_tokens = output_tokens = invalid_actions = 0
    latency = 0.0
    last_confidence = "low"
    used = 0
    for _ in range(limit):
        if adapter.done or adapter.score >= 100:
            break
        remaining = max(0, attempt_budget - steps_before - used)
        action, last_confidence, result = _llm_action(backend, adapter, history, style, remaining)
        input_tokens += int(result.get("input_tokens", 0))
        output_tokens += int(result.get("output_tokens", 0))
        latency += float(result.get("latency_seconds", 0))
        if not action:
            invalid_actions += 1
            action = "look" if isinstance(adapter, AlfWorldAdapter) else "look around"
        elif action not in adapter.valid_actions():
            invalid_actions += 1
        observation, _done = adapter.step(action)
        history.append((action, observation))
        used += 1
    return _node(adapter.score, last_confidence, input_tokens, output_tokens, latency, used, invalid_actions), history


def _generate_science(task: dict[str, Any], split: str, backend: Any, checkpoint: int,
                      penalty: float, judge: bool, seed: int,
                      episode_steps: int = 30) -> dict[str, Any]:
    adapter = ScienceWorldAdapter()
    try:
        adapter.reset(task)
        initial_observation = adapter.observation
        history: list[tuple[str, str]] = []
        backend.seed(_task_seed(seed, task["id"]))
        root, history = _run_steps(backend, adapter, history, checkpoint, "Solve the task directly.", episode_steps)
        remaining_steps = max(0, episode_steps - checkpoint)
        checkpoint_context = _action_prompt(adapter, history, "At a controller checkpoint.", remaining_steps)
        checkpoint_observation = adapter.observation
        root["input_tokens"] += 0
        tree = {"": root}
        if not adapter.done and adapter.score < 100:
            repair, _ = _run_steps(backend, adapter, list(history), max(0, episode_steps - checkpoint),
                                   "Repair the ongoing attempt. Use feedback to correct mistakes and continue.",
                                   episode_steps, checkpoint)
        else:
            repair = _node(adapter.score, root["confidence"], 0, 0, 0.0, 0, 0)
        tree["repair"] = repair

        adapter.reset(task)
        reset_match = adapter.observation == initial_observation
        retry_history: list[tuple[str, str]] = []
        backend.seed(_task_seed(seed ^ 0x5A5A5A5A, task["id"]))
        retry, _ = _run_steps(backend, adapter, retry_history, episode_steps,
                              "Start a fresh independent attempt from the beginning.", episode_steps)
        tree["retry"] = retry

        row = {
            **task, "answer": "1", "tree": tree, "benchmark": "scienceworld", "split": split,
            "task_family": task["task"], "checkpoint_observation": checkpoint_observation,
            "retry_reset_match": reset_match,
            "checkpoint_context": checkpoint_context, "judge_penalty": penalty if judge else None,
            "generation_config": None,
        }
        if judge:
            tree[""]["judges"] = {}
            for mode in ("full", "compact"):
                result = backend.generate(_judge_prompt(row, mode, penalty))
                obj = parse_object(result["text"])
                action = str(obj.get("action", "stop")).lower()
                result["action"] = action if action in ("stop", "repair", "retry") else "stop"
                tree[""]["judges"][mode] = result
        return row
    finally:
        adapter.close()


def _generate_alfworld(task: dict[str, Any], split: str, data_root: Path, backend: Any,
                       checkpoint: int, penalty: float, judge: bool,
                       seed: int, episode_steps: int = ALFWORLD_EPISODE_STEPS) -> dict[str, Any]:
    adapter = AlfWorldAdapter(data_root, split, [task["game_file"]], task["game_file"])
    try:
        adapter.reset()
        initial_observation = adapter.observation
        history: list[tuple[str, str]] = []
        backend.seed(_task_seed(seed, task["id"]))
        root, history = _run_steps(backend, adapter, history, checkpoint, "Solve the household task.", episode_steps)
        remaining_steps = max(0, episode_steps - checkpoint)
        checkpoint_context = _action_prompt(adapter, history, "At a controller checkpoint.", remaining_steps)
        checkpoint_observation = adapter.observation
        tree = {"": root}
        if not adapter.done and adapter.score < 100:
            repair, _ = _run_steps(backend, adapter, list(history), max(0, episode_steps - checkpoint),
                                   "Repair the ongoing attempt. Use feedback to correct mistakes and continue.",
                                   episode_steps, checkpoint)
        else:
            repair = _node(adapter.score, root["confidence"], 0, 0, 0.0, 0, 0)
        tree["repair"] = repair

        adapter.reset()
        if adapter.observation != initial_observation:
            raise RuntimeError(f"ALFWorld reset changed initial state for {task['id']}")
        retry_history: list[tuple[str, str]] = []
        backend.seed(_task_seed(seed ^ 0x5A5A5A5A, task["id"]))
        retry, _ = _run_steps(backend, adapter, retry_history, episode_steps,
                              "Start a fresh independent attempt from the beginning.", episode_steps)
        tree["retry"] = retry

        row = {
            **task, "answer": "1", "tree": tree, "benchmark": "alfworld", "split": split,
            "task_family": Path(task["game_file"]).parts[-3].split("-")[0],
            "checkpoint_observation": checkpoint_observation, "checkpoint_context": checkpoint_context,
            "judge_penalty": penalty if judge else None, "generation_config": None,
        }
        if judge:
            tree[""]["judges"] = {}
            for mode in ("full", "compact"):
                result = backend.generate(_judge_prompt(row, mode, penalty))
                obj = parse_object(result["text"])
                action = str(obj.get("action", "stop")).lower()
                result["action"] = action if action in ("stop", "repair", "retry") else "stop"
                tree[""]["judges"][mode] = result
        return row
    finally:
        adapter.close()


def generate_split(args: argparse.Namespace) -> None:
    data_root = Path(args.alfworld_data)
    size = SPLIT_SIZES[args.split] if args.size is None else args.size
    if args.benchmark == "scienceworld":
        tasks = scienceworld_tasks(args.split, args.selection_seed, size)
        episode_steps = args.episode_steps or 30
        generate_one = lambda task, backend: _generate_science(
            task, args.split, backend, args.checkpoint_steps,
            args.penalty, args.judge, args.seed, episode_steps)
    else:
        tasks = alfworld_tasks(args.split, args.selection_seed, size, data_root)
        episode_steps = args.episode_steps or ALFWORLD_EPISODE_STEPS
        generate_one = lambda task, backend: _generate_alfworld(
            task, args.split, data_root, backend, args.checkpoint_steps,
            args.penalty, args.judge, args.seed, episode_steps)
    config = {
        "benchmark": args.benchmark, "model": args.model, "seed": args.seed,
        "selection_seed": args.selection_seed, "split": args.split,
        "checkpoint_steps": args.checkpoint_steps,
        "episode_steps": episode_steps,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "judge": args.judge, "penalty": args.penalty if args.judge else None,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out = Path(args.output)
    if out.exists() and not args.resume:
        raise FileExistsError(f"{out} exists; use --resume")
    rows = read_jsonl(out) if out.exists() else []
    for old, task in zip(rows, tasks):
        if old.get("id") != task["id"] or old.get("generation_config") != config:
            raise ValueError("Saved interactive trajectories do not match selected tasks/config")
    if len(rows) > len(tasks):
        raise ValueError("Saved trajectories contain more tasks than selected input")
    if len(rows) == len(tasks):
        return
    backend = MLXBackend(args.model, args.max_tokens, args.temperature, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    for task in tasks[len(rows):]:
        row = generate_one(task, backend)
        row["generation_config"] = config
        rows.append(row)
        temporary = out.with_name(out.name + ".tmp")
        write_jsonl(temporary, rows)
        os.replace(temporary, out)
        print(f"Generated {args.benchmark} {args.split} {len(rows)}/{len(tasks)}: {task['id']}", flush=True)


def _simulate(row: dict[str, Any], policy: str, model: Model, qmodel: TabularQ,
              penalty: float, setting: Any = None) -> dict[str, Any]:
    tree = row["tree"]
    root = tree[""]
    belief = model.initial_belief(confidence(root))
    if policy == "tuned_fixed":
        action, judge_node = str(setting), None
    else:
        action, judge_node = choose(policy, tree, "", belief, 1, model, qmodel, penalty, setting)
    selected = root if action == "stop" else tree[action]
    tokens_in = int(root.get("input_tokens", 0))
    tokens_out = int(root.get("output_tokens", 0))
    latency = float(root.get("latency_seconds", 0))
    env_steps = int(root.get("env_steps", 0))
    invalid_actions = int(root.get("invalid_actions", 0))
    if action != "stop":
        tokens_in += int(selected.get("input_tokens", 0))
        tokens_out += int(selected.get("output_tokens", 0))
        latency += float(selected.get("latency_seconds", 0))
        env_steps += int(selected.get("env_steps", 0))
        invalid_actions += int(selected.get("invalid_actions", 0))
    judge_calls = int(judge_node is not None)
    if judge_node:
        tokens_in += int(judge_node.get("input_tokens", 0))
        tokens_out += int(judge_node.get("output_tokens", 0))
        latency += float(judge_node.get("latency_seconds", 0))
    total_tokens = tokens_in + tokens_out
    return {
        "id": row["id"], "correct": int(float(selected.get("score", 0)) >= 1.0),
        "retry_reset_match": row.get("retry_reset_match", True),
        "answer": selected["answer"], "reward": float(selected.get("score", 0)),
        "initial_reward": float(root.get("score", 0)), "input_tokens": tokens_in,
        "output_tokens": tokens_out, "tokens": total_tokens, "env_steps": env_steps,
        "invalid_actions": invalid_actions, "solver_latency_seconds": latency,
        "judge_latency_seconds": float(judge_node.get("latency_seconds", 0)) if judge_node else 0,
        "judge_calls": judge_calls, "actions": [action], "path": action if action != "stop" else "",
        "utility": float(selected.get("score", 0)) - penalty * total_tokens / 1000,
    }


def _tune(rows: list[dict[str, Any]], policy: str, model: Model, qmodel: TabularQ,
          penalty: float) -> Any:
    from itertools import product

    if policy == "tuned_fixed":
        candidates = ["stop", "repair", "retry"]
    elif policy == "threshold":
        candidates = list(product(range(4), ("repair", "retry")))
    elif policy == "hybrid":
        candidates = [0, .005, .01, .02, .05, .1, .2, 1]
    else:
        return None

    def score(setting: Any) -> float:
        if policy == "tuned_fixed":
            results = [_simulate(row, setting, model, qmodel, penalty) for row in rows]
            return statistics.mean(result["utility"] for result in results)
        results = [_simulate(row, policy, model, qmodel, penalty, setting) for row in rows]
        return statistics.mean(result["utility"] for result in results)

    return max(candidates, key=score)


def evaluate(args: argparse.Namespace) -> None:
    train, dev, test = (read_jsonl(path) for path in (args.train, args.dev, args.test))
    if not train or not dev or not test:
        raise ValueError("Train, dev, and test trajectory files must all be nonempty")
    id_sets = [set(row["id"] for row in split) for split in (train, dev, test)]
    if any(id_sets[i] & id_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Interactive train/dev/test task IDs must be disjoint")
    model, qmodel = Model.fit(train), TabularQ(train)
    summaries, details = [], []
    for policy in POLICIES:
        setting = _tune(dev, policy, model, qmodel, args.penalty)
        results = [_simulate(row, policy, model, qmodel, args.penalty, setting) for row in test]
        details.extend({"policy": policy, **result} for result in results)
        summaries.append({
            "policy": policy, "setting_selected_on_dev": setting, "n": len(results),
            "accuracy": statistics.mean(result["correct"] for result in results),
            "accuracy_ci_95": bootstrap_interval([result["correct"] for result in results]),
            "mean_reward": statistics.mean(result["reward"] for result in results),
            "reward_ci_95": bootstrap_interval([result["reward"] for result in results]),
            "mean_tokens": statistics.mean(result["tokens"] for result in results),
            "mean_env_steps": statistics.mean(result["env_steps"] for result in results),
            "mean_invalid_actions": statistics.mean(result["invalid_actions"] for result in results),
            "mean_latency_seconds": statistics.mean(result["solver_latency_seconds"] for result in results),
            "mean_utility": statistics.mean(result["utility"] for result in results),
        })
    from collections import defaultdict
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in details:
        groups[result["policy"]][result["id"]] = result
    baseline = groups["stop"]
    eligible_ids = {row["id"] for row in test if row.get("retry_reset_match", True)}
    paired = []
    for policy, group in groups.items():
        reward_diffs = [group[task_id]["reward"] - baseline[task_id]["reward"] for task_id in eligible_ids]
        success_diffs = [group[task_id]["correct"] - baseline[task_id]["correct"] for task_id in eligible_ids]
        paired.append({
            "policy": policy, "baseline": "stop", "n": len(eligible_ids),
            "excluded_reset_mismatches": len(test) - len(eligible_ids),
            "reward_difference": statistics.mean(reward_diffs) if reward_diffs else None,
            "reward_difference_ci_95": bootstrap_interval(reward_diffs) if reward_diffs else None,
            "accuracy_difference": statistics.mean(success_diffs) if success_diffs else None,
            "accuracy_difference_ci_95": bootstrap_interval(success_diffs) if success_diffs else None,
        })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_question.jsonl", details)
    (output / "summary.json").write_text(json.dumps({"penalty": args.penalty, "results": summaries}, indent=2) + "\n")
    (output / "paired-vs-stop.json").write_text(json.dumps(paired, indent=2) + "\n")
    print(output / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--benchmark", choices=("scienceworld", "alfworld"), required=True)
    gen.add_argument("--split", choices=("train", "dev", "test"), required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--model", default=MODEL_DEFAULT)
    gen.add_argument("--size", type=int)
    gen.add_argument("--selection-seed", type=int, default=20261101)
    gen.add_argument("--seed", type=int, default=101)
    gen.add_argument("--checkpoint-steps", type=int, default=6)
    gen.add_argument("--episode-steps", type=int, help="Override the benchmark episode budget (default: 30 SciWorld, 50 ALFWorld)")
    gen.add_argument("--max-tokens", type=int, default=128)
    gen.add_argument("--temperature", type=float, default=.7)
    gen.add_argument("--penalty", type=float, default=.01)
    gen.add_argument("--judge", action="store_true")
    gen.add_argument("--alfworld-data", default=".cache/alfworld")
    gen.add_argument("--resume", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--train", required=True)
    ev.add_argument("--dev", required=True)
    ev.add_argument("--test", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--penalty", type=float, default=.01)
    args = parser.parse_args()
    if args.command == "generate":
        generate_split(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
