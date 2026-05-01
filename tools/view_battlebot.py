"""Launch MuJoCo's native passive viewer on the battlebot scene."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer


SCENE = Path(__file__).resolve().parents[1] / "assets" / "mujoco" / "battlebot_scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.perf_counter() - step_start
            sleep_for = model.opt.timestep - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
