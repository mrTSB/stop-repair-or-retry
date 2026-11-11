# LLM controller experiment preparation

This workflow adds an initial full-history LLM controller to the published cached-tree dataset. It is independent of the older experiment/evaluation CLI: it does not load or fit a POMDP, import old controller results, regenerate solver responses, or use Tanvir's earlier benchmark results. The comparison has exactly five fixed baselines: submit initial, repair once, repair twice, retry once and retry twice.

**Status: full development and test collection completed.** The unchanged Qwen2.5-14B-Instruct controller ran on all 60 dev questions, was frozen, and then ran on all 400 test questions. See [full results and verification](../../results/llm-controller-full-evaluation/README.md) and the earlier [ten-question pilot](../../results/llm-controller-dev-pilot/README.md). The generic HTTP example still leaves endpoint, model, and billing unset; actual collection used private Modal RPC. Mock output is not LLM performance.

## Run without a provider or key

From the repository root, using Python 3.10+ with no added dependencies:

```sh
# Inspect exactly what the controller would see. No network request.
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment preview \
  --config experiments/llm-controller/config.example.json --limit 1

# Exercise persistence, replay, token accounting and reporting offline.
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment run \
  --config experiments/llm-controller/config.example.json \
  --limit 3 --output runs/llm-controller/mock-smoke --mock
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment evaluate \
  --run runs/llm-controller/mock-smoke

# Characterize training action outcomes, confidence counts and reward tradeoffs.
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment landscape \
  --output runs/llm-controller/training-landscape

PYTHONPATH=src python3 -m unittest discover -s tests -p test_llm_experiment.py -v
```

Use a fresh output directory if code, prompt, dataset, model or configuration changes. All generated run files and private configurations should stay under Git-ignored `runs/`. The published dataset/report files remain unchanged.

## Configure the eventual connection

Copy `config.example.json` into `runs/llm-controller/config.json`. Set the full HTTPS chat-completions URL and the explicit model name. The adapter expects the OpenAI-compatible chat-completions request/response shape, not the Responses or Anthropic Messages API. Provider compatibility is not established by offline tests.

Set `response_format` to `json_object`, `json_schema`, or `none` according to the chosen model's supported features. The prompt always requires exactly one JSON action. `output_limit_parameter` supports `max_tokens` or `max_completion_tokens`; set `temperature` to null if that parameter is unsupported. Unsupported options fail visibly instead of being silently removed or falling back to another model. Reasoning models may use the output allowance internally; verify their output cap and usage semantics in the smoke run.

Credentials are read only from the environment variable named by `api_key_env`. Never put a key in the config, prompt, command arguments or repository. In Bash, this reads it without echoing or putting the value into shell history:

```sh
read -rs LLM_CONTROLLER_API_KEY
export LLM_CONTROLLER_API_KEY
```

The endpoint must be HTTPS, with no credentials or query parameters in the URL. Redirects are rejected to avoid forwarding authorization to another origin. The response must contain one choice, a model identifier, and integer `usage.prompt_tokens` / `usage.completion_tokens`. Reasoning-token details are already included in completion totals and are not added twice. Missing usage stops the run; it is never treated as free inference.

### Modal is an option

Modal offers [OpenAI-compatible Endpoints](https://modal.com/docs/guide/endpoints). Dedicated Endpoints bill for [compute resources](https://modal.com/docs/guide/dedicated-endpoints), so they are the route to investigate for compute credits; confirm your particular credit eligibility before deployment. [Shared Endpoints](https://modal.com/docs/guide/shared-endpoints) are billed per token, and Modal states that plan-included credits cannot cover their usage. These details were checked on November 10, 2026 and must be rechecked before spending.

For a Modal Endpoint, copy the URL and served model name from its dashboard request example. Use its **proxy token** as the API credential; this is different from the account token used by the Modal CLI. Keep authentication enabled. The client does not create, resize, stop or deploy an endpoint. Creating a dedicated endpoint can incur compute/startup/idle charges even before this client sends its first request.

### Private Modal pilot

`modal_pilot.py` runs ten development questions (the first ten in existing file order, five math and five science) using private authenticated Modal RPC. It does not need a router key, deploy a public endpoint, or use Shared Endpoint token billing. The client configuration endpoint is descriptive; it is not used for HTTP inference.

The driver pins Qwen2.5-14B-Instruct revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, bfloat16, greedy decoding, and a 128-token output cap. It uses at most one A100-80GB, four CPU cores, and 48 GiB RAM. At most twenty controller decisions and one metadata call are allowed. Function retries are disabled; startup and call timeouts are 600 and 120 seconds. The temporary app exits after collection. Verify shutdown and billing independently afterward.

Before another paid run, inspect prior project charges (including failed startups), recalculate the conservative reservation using current prices, and select the intended account explicitly. The driver refuses existing output directories; do not automatically replay an interrupted run. Its dollar reservation is not a provider-enforced account spending limit. No command here should be run merely to reproduce an already collected result; replay/evaluation is offline.

The run directory retains the frozen protocol, selected IDs, model/backend metadata, request/response journal, token usage including chat-template and EOS tokens, and completion status. Research token cost includes every visited solver response and every controller decision, including stopping. RPC latency includes transport; GPU generation time is retained separately, and loading time is not per-decision latency.

### Full dev → freeze → test collection

`modal_full_evaluation.py` collects all 60 development questions, evaluates them offline, saves the freeze evidence, and then collects all 400 test questions using the exact same config and controller. It does not choose settings from the development scores. The model backend is identical to the pilot; only operational limits and the descriptive app address differ.

Both phases share one temporary GPU app. The global maximum is 920 controller decisions (120 dev + 800 test), and a separate watchdog process requests app shutdown after 1,800 seconds. This watchdog survives parent-process interruption; confirm actual shutdown with Modal afterward. The watchdog and request limits are operational protections, not a provider-enforced dollar cap. An initial prior-spend reservation plus the entire new compute/fees reservation must be below $20 before the app is created. The driver requires `MODAL_PROFILE=gancone`, an explicit `--execute`, a fresh output directory, and a reconciled `--prior-reserved-usd` value.

Do not execute the collector to reproduce archived results: use offline evaluation or the portable verification script. Raw evidence binds dev and test to a freeze and preserves all decisions, usage, and failures. Original absolute paths remain in manifests for provenance; the portable verifier resolves archived dev/test/freeze files within the bundle and checks their original hashes.

### Spending controls

For `billing_mode: "token"`, provide verified input/output USD per million token rates, a positive per-run `spend_limit_usd`, and a conservative `request_reserve_usd`. Before each request, prior estimated spending plus the reserve must fit the run limit. A request exceeding its reserve halts the run and blocks automatic continuation. These are estimates, **not a hard provider billing cap**: a response may exceed the reserve, and caching, reasoning, provider fees or pricing changes can make estimates differ from invoices. Use a provider-side spending limit too. Limits are per run; previous development runs and other account use are not included automatically.

For `billing_mode: "compute"`, leave all token-price/dollar-budget fields null. Configure the provider's compute budget and resource limits first, then set `compute_budget_confirmed: true`. This is an operator acknowledgment, not verification by this client. The client records research token costs, but cannot infer GPU startup, idle or wall-time charges from tokens. Check the provider ledger separately and stop/scale down resources when appropriate. Do not translate the original Modal credit limit into an invented token price.

Both modes enforce `max_requests` and a `max_run_seconds` launch guard. The latter is checked before calls and includes previously recorded response latency on resume; a request may last up to its separate `timeout_seconds` beyond that guard. It is not a cloud-server lifetime or account-spend limit. No automatic HTTP retries are made.

## Run development before test

After configuring the provider and verifying its account budget, start with a small development smoke run:

```sh
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment run \
  --config runs/llm-controller/config.json --split dev --limit 3 \
  --output runs/llm-controller/dev-smoke-v1 --live
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment evaluate \
  --run runs/llm-controller/dev-smoke-v1
```

Inspect raw action validity, usage, response model, latency and provider charges. Adjust prompts/configuration using train/dev only, with a new run directory for each version. No claim should be made from a three-question smoke run. When a version is ready, run all 60 development questions by omitting `--limit`:

```sh
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment run \
  --config runs/llm-controller/config.json --split dev \
  --output runs/llm-controller/dev-final --live
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment evaluate \
  --run runs/llm-controller/dev-final
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment freeze \
  --dev-run runs/llm-controller/dev-final \
  --output runs/llm-controller/frozen.json
```

Freezing requires a complete live 60-question development run. It records code, prompt, reward design, config, dataset and development-evidence hashes. Keep that development run with its freeze file. The test command refuses missing/mismatched freezes or modified/deleted development evidence. Requested model aliases do not guarantee unchanged model weights; prefer a versioned model when available. Responses record the provider's reported model, and test rejects names not present in the frozen development run.

```sh
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment run \
  --config runs/llm-controller/config.json --split test \
  --freeze runs/llm-controller/frozen.json \
  --output runs/llm-controller/test-frozen --live
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment evaluate \
  --run runs/llm-controller/test-frozen
```

This split contains 400 test questions; at most 800 controller calls are needed, often fewer if it stops early. **Baseline/test outcomes have already been inspected.** The freeze prevents subsequent changes; it does not make this an unseen confirmation set. Use fresh held-out questions if those outcomes informed redesign. The current loader intentionally targets the published 100/60/400 dataset; a future dataset version needs its own manifest and protocol.

## What the controller sees and pays for

`prompt.txt` defines stop/repair/retry, remaining budget, the cost objective, and the strict action-only schema. The request body is built from an explicit allowlist: question, visited raw responses, their observed token counts, remaining solver calls and lambda. It contains no gold answer, reference solution, correctness labels, future nodes, or old controller decisions. Embedded instructions in question/response text are treated as task data.

At most two additional solver calls are allowed. Repair reads the cached response at the repair child; retry reads the fresh-solve child. Both replace the current answer. At depth two the latest answer is submitted automatically without another controller request. This experiment generates **controller decisions only**, using the same original solver outputs for every comparison.

All visited solver tokens and executed controller input/output tokens count, including a controller call that says stop. An invalid action (extra fields, malformed JSON, refusal, Markdown, unknown action) falls back to stop, is flagged, and retains its usage cost. Failed transport or missing usage does not silently become a stop decision or zero cost. Only controller HTTP latency is measured; cached shared-batch solver timing is not reported as serial interactive latency.

## Resume and uncertain requests

Repeating the identical `run` command reuses saved decisions. A durable request record is written before the HTTP call. Successful raw responses are saved before parsing, then accepted responses are journaled. Only one process can own a run directory. After a crash, verify the old process has stopped before removing a stale `.lock` file.

A request without an accepted response blocks resumption: the server may have completed and charged it. The client never retries it automatically. If an authentic provider response/usage receipt is available (or a complete saved `raw_response` event), extract that response JSON and reconcile it:

```sh
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment recover-response \
  --run runs/llm-controller/dev-smoke-v1 \
  --request-id REQUEST_ID_FROM_JOURNAL \
  --response-json runs/llm-controller/provider-response.json \
  --latency-seconds MEASURED_LATENCY
```

The operator must verify that receipt belongs to the specific request; the client cannot prove that association. Never fabricate usage, replace a decision because its outcome was bad, or clear a request merely to retry it. If the response/usage cannot be recovered, preserve the incomplete run and reconcile provider billing before deciding on a new run. There is deliberately no automated paid retry command.

## Reward landscape and future POMDP

`reward_design.json` fixes the objective: final correctness minus lambda times executed solver/controller tokens per 1,000. The current reference lambda is 0.01; the candidate tradeoff grid is 0, .001, .01, .1 and 1. Declare the grid/cost preference before comparative evaluation. Do not select lambda by maximizing utility across different lambdas: smaller penalties mechanically raise utility. Generate separate LLM decisions at each lambda because the objective changes its prompt; reweighting old decisions is only fixed-prompt sensitivity.

At lambda .01, an additional 500-token call costs .005 reward. It is worth executing if its expected net improvement in final correctness exceeds 0.5 percentage points, including the risk of damaging a correct answer. Costs already incurred are sunk at decision time, but still count in end-to-end results. This is a transparent accuracy–cost objective, not proof that .01 represents the right preference or that token counts equal FLOPs/dollars across models.

The `landscape` command reads **training only** and reports empirical action/correctness transition counts, confidence counts and mean action tokens, plus the five fixed baselines across the candidate lambda grid. It does not fit smoothed models, plan actions or optimize a POMDP. Later, a POMDP can learn transitions, confidence observations and expected action costs from train; tuning stays on dev. The reward formula is a declared objective, not something inferred from test outcomes.

## Artifacts and validation

Each run retains the exact request/response journal, requested configuration, model identities, frozen input hashes and parsed outcomes. Evaluation requires complete, consistent visited paths. It writes controller paths, six-method summaries (LLM plus five fixed baselines), and paired question-bootstrap utility differences against each baseline. Domain summaries are included. These exploratory multiple comparisons do not establish a confirmatory win by themselves.

Offline tests cover observation leakage, forced horizon, strict parsing, judge-stop token charges, schema options, redirects, missing usage, interrupted/resumed calls, recovered receipts, request/spend guards, data/model/prompt changes, test freezes and independent baseline replay. Actual provider compatibility, price estimates and usefulness of the prompt remain to be established by a future live development run.
