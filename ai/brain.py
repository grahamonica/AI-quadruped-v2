"""Leaky Integrate-and-Fire spiking neural network brain for the quadruped.

Inputs (48 total):
  [0:3]   goal_xyz          – goal point world coords
  [3:6]   com_xyz           – total centre of mass world coords
  [6:9]   body_com_xyz      – body centre of mass world coords
  [9:21]  leg_foot_xyz      – world coords of each leg tip (4×3)
  [21:33] leg_com_xyz       – world coords of each leg COM  (4×3)
  [33:36] body_imu_rad      – body IMU roll/pitch/yaw
  [36:48] leg_imu_rad       – each leg IMU roll/pitch/yaw   (4×3)

Outputs (4): one target angular velocity per leg, in [-1, 1] (scaled by caller).

LIF integration
---------------
Discrete LIF: v_new = v*decay + i_in*scale, scale=dt/τ, decay=1-scale.
Steady state: v_ss = i_in.  With V_THRESH=0.01 and typical |i_in|~0.1-0.3,
neurons fire roughly every 5-20 steps.  A spike trace (EMA) smooths the
sparse spikes into a continuous rate signal for the linear output layer.

Exploration noise
-----------------
Gaussian noise is added to the motor outputs each step.  This drives
random leg movement even early in training so ES has diverse behaviour
to evaluate.  The noise scale is fixed at MOTOR_NOISE_SCALE.
"""
from __future__ import annotations

import numpy as np


# Network sizes
N_IN = 48
N_HID = 64
N_OUT = 4

# LIF parameters
TAU_MEM = 0.020      # membrane time constant (s)
V_THRESH = 0.01      # spike threshold — neurons fire every ~7 steps at typical inputs
V_RESET = 0.0        # reset potential after spike
DT = 0.010           # default motor update timestep (s) — 10 ms / 100 Hz

# Exploration noise added to each motor output step
MOTOR_NOISE_SCALE = 0.20   # in tanh(·) space → ≈1.2 rad/s jitter at MOTOR_SCALE=6


class SNNBrain:
    """LIF hidden layer with spike-trace output and per-step exploration noise."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        rng = self._rng
        self.w_in_hid: np.ndarray = rng.normal(0.0, 1.0 / N_IN**0.5, (N_HID, N_IN)).astype(np.float32)
        self.b_hid: np.ndarray = rng.uniform(-0.2, 0.2, N_HID).astype(np.float32)
        self.w_hid_out: np.ndarray = rng.normal(0.0, 4.0 / N_HID**0.5, (N_OUT, N_HID)).astype(np.float32)
        self.b_out: np.ndarray = np.zeros(N_OUT, dtype=np.float32)
        # LIF membrane state
        self._v_hid: np.ndarray = np.zeros(N_HID, dtype=np.float32)
        # Spike trace (EMA) — smooths sparse spikes into a rate signal
        self._trace: np.ndarray = np.zeros(N_HID, dtype=np.float32)
        self._trace_decay: float = 0.70

    def reset(self) -> None:
        self._v_hid[:] = 0.0
        self._trace[:] = 0.0

    def step(self, obs: np.ndarray, dt: float = DT, noise: bool = True) -> np.ndarray:
        """Run one timestep; returns motor commands in [-1, 1]."""
        scale = dt / TAU_MEM
        decay = 1.0 - scale

        # LIF membrane update
        i_in = self.w_in_hid @ obs + self.b_hid
        self._v_hid = self._v_hid * decay + i_in * scale

        # Spike + reset
        spikes = (self._v_hid >= V_THRESH).astype(np.float32)
        self._v_hid[spikes > 0] = V_RESET

        # Spike trace (rate code)
        self._trace = self._trace * self._trace_decay + spikes * (1.0 - self._trace_decay)

        # Output readout + optional exploration noise
        out = self.w_hid_out @ self._trace + self.b_out
        if noise:
            out = out + self._rng.standard_normal(N_OUT).astype(np.float32) * MOTOR_NOISE_SCALE
        return np.tanh(out)

    # ── ES parameter access ────────────────────────────────────────────────────
    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.w_in_hid.ravel(), self.b_hid,
            self.w_hid_out.ravel(), self.b_out,
        ])

    def set_params(self, params: np.ndarray) -> None:
        n0 = N_HID * N_IN
        n1 = n0 + N_HID
        n2 = n1 + N_OUT * N_HID
        self.w_in_hid = params[:n0].reshape(N_HID, N_IN).astype(np.float32)
        self.b_hid = params[n0:n1].astype(np.float32)
        self.w_hid_out = params[n1:n2].reshape(N_OUT, N_HID).astype(np.float32)
        self.b_out = params[n2:].astype(np.float32)

    def param_count(self) -> int:
        return N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT
