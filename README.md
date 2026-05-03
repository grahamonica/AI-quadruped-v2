# Multi-Brain Quadruped Sim

MuJoCo and JAX playground for embodied control experiments.

The current stable path is quadruped training and playback with static MJCF/STL assets, a JAX evolution-strategy trainer, and a native MuJoCo viewer. The battlebot assets are available as a standalone CAD-to-MJCF scene for geometry work. Stunt commands such as front flip, back flip, and side roll are experimental placeholders; they are not treated as working behaviors or quality gates yet.

## Repository Map

- `assets/mujoco/`: canonical MuJoCo XML and mesh assets.
- `assets/mujoco/scene.xml`: quadruped scene with shared ground.
- `assets/mujoco/battlebot_scene.xml`: battlebot scene with shared ground.
- `assets/mujoco/meshes/quadruped/`: checked-in quadruped STL meshes.
- `assets/mujoco/meshes/battlebot/`: STEP-generated battlebot STL meshes.
- `battlebot_step/`: source STEP files for battlebot assets.
- `brains/sim/`: MuJoCo backend, action projection, and asset loading.
- `brains/jax_trainer.py`: JAX ES trainer.
- `brains/runtime/`: playback, quality gates, logging, and checkpoint metadata.
- `configs/`: runtime YAML configs.
- `tests/`: unittest suite used by CI.
- `tools/generate_mujoco_from_step.py`: battlebot STEP-to-STL/MJCF generator.
- `tools/view_battlebot.py`: native viewer entry point for the battlebot scene.
- `train_headless.py`: headless training and quality-gate entry point.
- `train.sbatch`: SLURM wrapper for cluster training.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

For STEP conversion, install `pythonocc-core` in a conda environment:

```bash
conda install -c conda-forge pythonocc-core
```

Runtime simulation does not need `pythonocc-core`; it only loads the generated MJCF/STL files.

## Quality And CI

GitHub Actions builds the Docker image and runs the default container command:

```bash
docker build -t ai-quadruped:test .
docker run --rm ai-quadruped:test
```

That command runs:

```bash
python -m unittest discover -s tests -v
```

For a faster local loop:

```bash
python3 -m unittest tests.test_control -v
python3 -m unittest discover -s tests -v
```

The suite is intentionally lean. It covers config validation, control projection, static MuJoCo asset geometry, checkpoint metadata, backend construction, and one smoke quality-gate pass. Placeholder stunt behaviors, notebook registry helpers, plugin experiments, and fixed-seed reward snapshots are not CI gates.

## Run Quality Gates

```bash
python3 train_headless.py --config configs/smoke.yaml --quality-only
```

Quality gates compile the MuJoCo model, check reset pose and zero-action stability, verify deterministic playback, and run a short performance budget check.

## Train

Local smoke run:

```bash
python3 train_headless.py --config configs/smoke.yaml --generations 1
```

Default run:

```bash
python3 train_headless.py --config configs/default.yaml --generations 100
```

Checkpoints are written under:

```text
checkpoints/<model_type>_<log_id>/
```

Reusing `--log-id` resumes from that run's `latest.npz` when compatible.

For SLURM:

```bash
sbatch train.sbatch
```

Useful SLURM overrides:

```bash
CONFIG=configs/default.yaml MODEL_TYPE=shared_trunk_es GENERATIONS=200 sbatch train.sbatch
LOG_ID=my_run RESUME=checkpoints/shared_trunk_es_my_run/latest.npz sbatch train.sbatch
EXTRA_ARGS="--population-size 64 --episode-seconds 20" sbatch train.sbatch
```

## Run The Viewer

On macOS, the MuJoCo passive viewer must run on the main thread, so use `mjpython` from the `mujoco` package. On Linux, regular `python3` works.

```bash
# macOS
.venv/bin/mjpython main.py --config configs/smoke.yaml

# Linux
python3 main.py --config configs/smoke.yaml
```

With no `--model`, the viewer shows the static reset scene. To play a saved checkpoint:

```bash
.venv/bin/mjpython main.py --list
.venv/bin/mjpython main.py --model quadruped_es_001
.venv/bin/mjpython main.py --model quadruped_es_001 --goal 1.5,0.5
```

Viewer keys:

- Arrow keys: nudge the goal and restart the episode.
- `R`: restart the current episode.
- `G`: print the current goal.
- Close the window to exit.

## MuJoCo Assets

Open scenes directly:

```bash
python3 -m mujoco.viewer --mjcf assets/mujoco/scene.xml
python3 -m mujoco.viewer --mjcf assets/mujoco/battlebot_scene.xml
```

Regenerate battlebot assets from STEP:

```bash
python3 tools/generate_mujoco_from_step.py
```

The generator writes:

- `assets/mujoco/battlebot.xml`
- `assets/mujoco/battlebot_scene.xml`
- `assets/mujoco/meshes/battlebot/*.stl`
- `assets/mujoco/meshes/battlebot/manifest.json`

The battlebot assembly uses CAD-derived placement constants for the weapon and two wheel geoms. The single wheel mesh is instantiated twice in MJCF, once per side.

## Notes

- The quadruped is the training target used by tests and quality gates.
- The battlebot scene is currently an asset/viewing target, not a trainer target.
- Command primitives are useful for control-layer experiments, but stunt commands are not validated as solved behaviors yet.
