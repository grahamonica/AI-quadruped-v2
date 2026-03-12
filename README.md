# AI Quadruped v2

Minimal Python quadruped simulation inspired by the KT2 Kungfu Turtle layout.

## Physics Scope

- The floor is an immovable plane at `z = 0`.
- Gravity is always applied using `F = ma`.
- The body is simulated as a 3D rigid body with translation and roll/pitch/yaw rotation.
- Each foot can enter `airborne`, `static`, or `kinetic` contact in 3D.
- Contact forces at the feet and body corners generate both linear force and tipping torque.
- If the center of mass moves outside the active support region, the robot can rotate and fall.

## Run UI

```bash
./venv/bin/python main.py
```

This starts the training server and the React frontend.

## Train Headless

```bash
./venv/bin/python train_headless.py --generations 100 --save-every 5 --episode-seconds 60
```

This runs the ES trainer without the websocket server or frontend and writes checkpoints to `checkpoints/latest.npz`, `checkpoints/best.npz`, and periodic `checkpoints/generation_*.npz` files.

Use `--progress-every-steps N` to print step-level progress while training, and `--resume checkpoints/latest.npz` to continue from a saved model.
