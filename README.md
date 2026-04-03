# AI Quadruped v2

Config-driven JAX training stack for a quadruped locomotion task. The repo is organized as a very small production-style project: a real domain layer for the robot and environment, packaged runtime code under `ai/`, thin root launchers, declarative configs, structured logs, regression tests, and a working live frontend/backend path.

**Repo Layout**

- `quadruped/`: domain models for body, leg, motor, robot, and environment/task representation.
- `ai/api/`: FastAPI websocket services for live training and single-checkpoint viewing.
- `ai/config/`: typed YAML/JSON runtime spec loading and validation.
- `ai/infra/`: structured run logging and artifact helpers.
- `ai/quality/`: quality gates and fixed-seed regression tooling.
- `ai/runtime/`: checkpoint and launcher runtime helpers.
- `ai/`: JAX trainer implementation and compatibility exports.
- `configs/`: declarative runtime specs in YAML.
- `frontend/`: React + Vite viewer that consumes websocket metadata and swarm frames.
- `tests/`: config validation, smoke quality checks, and fixed-seed regression tests.
- `checkpoints/`: saved model artifacts.
- `logs/`: per-run structured artifacts and metrics. This directory is gitignored.
- `.github/workflows/quality-gates.yml`: CI job that runs the unittest suite on push and PRs.

**Architecture**

The runtime is split into three layers:

1. Domain layer: [quadruped/robot.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/robot.py), [quadruped/leg.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/leg.py), [quadruped/motor.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/motor.py), and [quadruped/environment.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/environment.py) define the robot and task in a logical, real-world shape.
2. Execution layer: [ai/jax_trainer.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/jax_trainer.py) converts the resolved spec and domain models into JAX arrays and runs training or rollouts.
3. Service layer: [ai/api/live.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/api/live.py), [ai/api/single.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/api/single.py), [main.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/main.py), [run_single.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/run_single.py), and the React frontend expose live viewers and headless workflows.

The important constraint is that the domain layer is real and explicit, but the simulation hot path stays functional for JAX `jit` and batched execution.

**Runtime Spec**

The trainer now reads a resolved environment/task spec from YAML or JSON. The default spec lives at [configs/default.yaml](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/configs/default.yaml), and a faster validation profile lives at [configs/smoke.yaml](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/configs/smoke.yaml).

The JAX simulator remains functional and batch-oriented, but the runtime is now fed through explicit domain objects in [quadruped/robot.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/robot.py), [quadruped/leg.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/leg.py), [quadruped/motor.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/motor.py), and [quadruped/environment.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/environment.py). That keeps the repo logically modeled without putting Python objects into the JAX hot path.

Supported sections:

- `terrain`: stepped arena or flat terrain, field bounds, step count/width/height, floor height.
- `goals`: radial random goals or a fixed goal.
- `spawn_policy`: origin, fixed points, or uniform spawn box.
- `friction`: static/kinetic foot friction and body friction.
- `robot`: body dimensions, masses, leg geometry, motor limits, and contact sampling.
- `physics`: gravity, contact stiffness/damping, substep budget, and sleep thresholds.
- `episode`: episode length, brain timestep, lifespan, selection cadence, and goal radius.
- `reward`: progress/noise/tipping/climbing reward parameters.
- `training`: ES population size, sigma, learning rate, and elite count.
- `quality_gates`: fast runtime checks and performance thresholds.
- `logging`: log level and artifact filenames.

Checkpoint resumes are now config-aware. A resume attempt will fail if the checkpoint was produced with a different resolved runtime spec.

**Install**

```bash
python3 -m pip install -r requirements.txt
```

For the frontend:

```bash
cd frontend
npm install
```

**Quick Start**

Run the fast validation profile end to end:

```bash
python3 run_quality_gates.py --config configs/smoke.yaml
python3 train_headless.py --config configs/smoke.yaml --generations 1
python3 main.py --config configs/smoke.yaml
```

If you only want the live checkpoint viewer instead of continuous training:

```bash
python3 run_single.py --config configs/smoke.yaml
```

**Run Quality Gates**

Fast runtime validation against a config:

```bash
python3 run_quality_gates.py --config configs/smoke.yaml
```

The built-in quality suite covers:

- invalid spawn detection
- collision sanity checks
- determinism checks
- unstable-state detection
- performance budget checks

Fixed-seed regression coverage is enforced in the test suite using the smoke config baseline at [tests/fixtures/smoke_regression_baseline.json](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/tests/fixtures/smoke_regression_baseline.json).

Run the repo tests locally:

```bash
python3 -m unittest discover -s tests -v
```

**Run Live UI**

Live training viewer:

```bash
python3 main.py --config configs/default.yaml
```

Single-checkpoint viewer:

```bash
python3 run_single.py --config configs/default.yaml
```

Both launchers start:

- a FastAPI websocket backend on port `8000`
- the Vite frontend on port `5173`

They now use the same runtime config pattern as the headless path, so terrain and robot geometry shown in the frontend come from backend metadata instead of duplicated frontend constants.

**How Config Flows Through The System**

1. A YAML or JSON file is loaded through [ai/config/schema.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/config/schema.py).
2. That config is converted into domain objects in `quadruped/`.
3. The JAX trainer applies those values to its runtime constants and cached tensors.
4. APIs and the frontend viewer receive metadata derived from the same resolved config.
5. Checkpoints embed the resolved config and are rejected on resume if the active config is incompatible.

**Train Headless**

```bash
python3 train_headless.py --config configs/default.yaml --generations 100 --seed 42
```

Useful variants:

```bash
python3 train_headless.py --config configs/smoke.yaml --quality-only
python3 train_headless.py --config configs/default.yaml --resume checkpoints/latest.npz
python3 train_headless.py --config configs/default.yaml --population-size 64 --episode-seconds 20
```

By default the headless entrypoint runs the fast quality gates before training starts. Use `--skip-quality-gates` only when you intentionally want to bypass that preflight.

**Run Artifacts**

Each headless run writes a timestamped artifact directory under `logs/` containing:

- `resolved_config.yaml`: the exact runtime spec used for the run
- `events.jsonl`: structured event logs
- `metrics.jsonl`: generation-level metrics
- `quality_report.json`: preflight validation results
- `summary.json`: terminal run summary

Training checkpoints continue to be written to `checkpoints/`:

- `latest.npz`
- `best.npz`
- `best_single.npz`
- `top_01.npz` through `top_N.npz`

Old checkpoints with incompatible parameter shapes or mismatched configs are rejected during load instead of failing later during rollout.

**Plot Rewards**

```bash
python3 plot_rewards.py --checkpoint checkpoints/latest.npz
```

**Notes**

- `train_headless.py --save-every` is retained only for CLI compatibility. Numbered generation checkpoints are not written.
- The live viewer entrypoints depend on [server.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/server.py), [server_single.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/server_single.py), and [frontend/index.html](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/frontend/index.html), which are present again in this repo.
- The launchers prefer the current Python interpreter first. That avoids silently using a stale repo-local venv when it differs from the environment you are actively running.
