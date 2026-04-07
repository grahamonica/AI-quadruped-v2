# AI Quadruped v2

Config-driven quadruped training stack. The repo is organized as follows: a domain layer for the robot and environment, packaged runtime code under `ai/`, declarative configs, structured logs, regression tests, and one live viewer stack. The runtime uses JAX for policy math and optimization, and MuJoCo for rollout and playback.

**Repo Layout**

- `quadruped/`: domain models for body, leg, motor, robot, and environment/task representation.
- `ai/sim/`: runtime rollout code, MuJoCo integration, and internal simulator helpers.
- `ai/api/`: FastAPI websocket service for the frontend viewer.
- `ai/config/`: typed YAML/JSON runtime spec loading and validation.
- `ai/infra/`: structured run logging and artifact helpers.
- `ai/quality/`: quality gates and fixed-seed regression tooling.
- `ai/runtime/`: checkpoint and launcher runtime helpers.
- `ai/`: the single trainer implementation, simulator-facing runtime glue, and compatibility exports.
- `configs/`: declarative runtime specs in YAML.
- `frontend/`: React + Vite viewer that consumes websocket metadata and single-model replay frames.
- `tests/`: config validation, smoke quality checks, and fixed-seed regression tests.
- `checkpoints/`: saved model artifacts.
- `logs/`: per-run structured artifacts and metrics. This directory is gitignored.
- `.github/workflows/quality-gates.yml`: CI job that runs the unittest suite on push and PRs.

**Architecture**

The runtime is split into four layers:

1. Domain layer: [quadruped/robot.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/robot.py), [quadruped/leg.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/leg.py), [quadruped/motor.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/motor.py), and [quadruped/environment.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/environment.py) define the robot and task in a logical, real-world shape.
2. Config layer: [ai/config/schema.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/config/schema.py) resolves YAML/JSON into a typed runtime spec.
3. Runtime layer: [ai/sim/mujoco_backend.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/sim/mujoco_backend.py) owns rollout execution, and [ai/sim/mujoco_model_builder.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/sim/mujoco_model_builder.py) is the only place that translates domain objects into MuJoCo MJCF. [ai/sim/jax_backend.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/sim/jax_backend.py) remains only as an internal reference helper for quality checks.
4. Training and service layer: [ai/jax_trainer.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/jax_trainer.py) contains the single ES trainer implementation. That trainer keeps optimization and policy math in JAX and always drives rollout through MuJoCo. The service layer exposes the viewer and headless workflows on top of that shared trainer.


**Runtime Backend**

The runtime uses the following backend configuration:

```yaml
simulator:
  backend: unified
  render: false
  deterministic_mode: true
  mujoco:
    timestep_s: 0.0025
    integrator: implicitfast
    solver: Newton
    solver_iterations: 100
    line_search_iterations: 50
    noslip_iterations: 4
    contact_margin_m: 0.002
    actuator_force_limit: 12.0
    velocity_servo_gain: 6.0
    joint_range_rad: [-1.1, 1.1]
```

`backend: unified` means:

- JAX owns the policy parameter vector, ES update step, and other compute-heavy math.
- MuJoCo owns environment rollout, playback, and contact dynamics.

The rollout path is built from the existing domain objects. The flow is:

1. `config` -> typed `RuntimeSpec`
2. `RuntimeSpec` -> `QuadrupedRobot` and `SimulationEnvironment`
3. domain models -> MuJoCo MJCF in [ai/sim/mujoco_model_builder.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/sim/mujoco_model_builder.py)
4. compiled model -> `MuJoCoBackend`
5. runtime backend -> trainer, quality gates, APIs, and frontend metadata

The frontend still consumes the same websocket frame shape, and checkpoints remain config-aware.

**Runtime Spec**

The trainer reads a resolved environment/task spec from YAML or JSON. The main configs live at [configs/default.yaml](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/configs/default.yaml) and [configs/smoke.yaml](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/configs/smoke.yaml).

The runtime is fed through explicit domain objects in [quadruped/robot.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/robot.py), [quadruped/leg.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/leg.py), [quadruped/motor.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/motor.py), and [quadruped/environment.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/quadruped/environment.py). MuJoCo uses that shared source of truth and compiles it into MJCF instead of duplicating dimensions or terrain constants in rollout code.

Supported sections:

- `simulator`: runtime mode and MuJoCo solver/control settings.
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

Checkpoint resumes are config-aware. A resume attempt only fails when the resolved runtime spec differs in a meaningful way.

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

If you only want the viewer for the current model:

```bash
python3 main.py --config configs/smoke.yaml
```

**Run Quality Gates**

Fast runtime validation against a config:

```bash
python3 run_quality_gates.py --config configs/smoke.yaml
```

The built-in quality suite supports two profiles:

- `runtime`: validates the actual unified MuJoCo-backed runtime.
- `reference`: runs faster internal reference checks against the legacy JAX simulator helpers.

The smoke config uses the `runtime` profile. That suite covers:

- invalid spawn detection
- collision sanity checks
- determinism checks
- unstable-state detection
- MJCF/model compilation succeeds
- reset pose is contact-safe
- zero-action rollout stays numerically stable
- identical seed reproduces the same rollout
- the internal JAX reference path and the runtime path show bounded drift on a fixed smoke rollout
- MuJoCo rollout time stays inside the configured budget

Fixed-seed regression coverage is enforced in the test suite using the smoke config baseline at [tests/fixtures/smoke_regression_baseline.json](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/tests/fixtures/smoke_regression_baseline.json).

Run the repo tests locally:

```bash
python3 -m unittest discover -s tests -v
```

**Run Viewer**

Unified viewer launcher:

```bash
python3 main.py --config configs/default.yaml
```

This launcher starts the only UI stack in the repo:

- a FastAPI websocket backend, default port `8000`
- the Vite frontend, default port `5173`

Both ports are configurable:

```bash
python3 main.py --config configs/default.yaml --api-port 8001 --frontend-port 5174
```

It uses the same runtime config pattern as the headless path, so terrain and robot geometry shown in the frontend come from backend metadata instead of duplicated frontend constants. The viewer contract stays simulator-agnostic; the backend adapts itself to the same metadata and single-model frame shape.

**How Config Flows Through The System**

1. A YAML or JSON file is loaded through [ai/config/schema.py](/Users/monicagraham/Desktop/GitHub/AI-quadruped-v2/ai/config/schema.py).
2. That config is converted into domain objects in `quadruped/`.
3. The single ES trainer applies those values to its runtime constants and cached tensors.
4. The runtime backend is constructed from those same resolved objects and attached under that trainer.
5. APIs and the frontend viewer receive metadata derived from the same resolved config.
6. Checkpoints embed the resolved config so the runtime can reject genuinely incompatible resumes.

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

Old checkpoints with incompatible parameter shapes or genuinely mismatched configs are rejected during load instead of failing later during rollout. Automatic resume skips incompatible artifacts and starts fresh; explicit `--resume` still fails loudly so you do not accidentally resume the wrong run.

**Runtime Notes**

- The runtime backend always uses MuJoCo for rollout.
- JAX is still used internally for policy math, optimizer updates, and reference/regression tooling.
- The smoke config intentionally uses `terrain.kind: flat` to keep the runtime quality gates narrow and stable.

**Plot Rewards**

```bash
python3 plot_rewards.py --checkpoint checkpoints/latest.npz
```

**Notes**

- `train_headless.py --save-every` is retained only for CLI compatibility. Numbered generation checkpoints are not written.
- The launchers prefer the current Python interpreter first. That avoids silently using a stale repo-local venv when it differs from the environment you are actively running.
