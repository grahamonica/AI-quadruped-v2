"""Static MuJoCo asset loading and compiled-model metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


MUJOCO_ROOT = Path(__file__).resolve().parents[2] / "assets" / "mujoco"
SCENE_XML_PATH = MUJOCO_ROOT / "scene.xml"

LEG_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
LEG_ROTATION_AXIS_BODY = (0.0, 1.0, 0.0)
TORSO_BODY_NAME = "torso"
TORSO_GEOM_NAME = "torso_geom"
GROUND_GEOM_NAME = "ground"
LEG_BODY_NAMES = tuple(f"{name}_leg" for name in LEG_NAMES)
LEG_GEOM_NAMES = tuple(f"{name}_leg_foot" for name in LEG_NAMES)
LEG_JOINT_NAMES = tuple(f"{name}_hinge" for name in LEG_NAMES)
FOOT_SITE_NAMES = tuple(f"{name}_foot_site" for name in LEG_NAMES)
ACTUATOR_NAMES = tuple(f"{name}_motor" for name in LEG_NAMES)


@dataclass(frozen=True)
class RobotGeometry:
    body_half_extents_m: tuple[float, float, float]
    leg_length_m: float
    leg_radius_m: float
    reset_height_m: float
    floor_height_m: float
    mount_points_body: tuple[tuple[float, float, float], ...]
    total_mass_kg: float

    @property
    def body_size_m(self) -> tuple[float, float, float]:
        return tuple(float(value * 2.0) for value in self.body_half_extents_m)


def load_mujoco_model() -> mujoco.MjModel:
    """Compile the canonical MJCF scene from disk."""
    return mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))


def _name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"MuJoCo model is missing {obj_type.name} named {name!r}.")
    return int(obj_id)


def _neutral_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    if model.nq >= 7:
        data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(model, data)
    return data


def _mesh_geom_bounds(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        half = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        return center - half, center + half

    start = int(model.mesh_vertadr[mesh_id])
    stop = start + int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start:stop], dtype=np.float64)
    geom_xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    geom_xpos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    world_vertices = geom_xpos[None, :] + vertices @ geom_xmat.T
    return world_vertices.min(axis=0), world_vertices.max(axis=0)


def robot_geometry_from_model(model: mujoco.MjModel) -> RobotGeometry:
    """Derive robot dimensions from the compiled static MJCF model."""
    data = _neutral_data(model)
    torso_body_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
    torso_geom_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, TORSO_GEOM_NAME)
    ground_geom_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM_NAME)
    leg_body_ids = [_name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in LEG_BODY_NAMES]
    leg_geom_ids = [_name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in LEG_GEOM_NAMES]
    foot_site_ids = [_name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITE_NAMES]

    torso_min, torso_max = _mesh_geom_bounds(model, data, torso_geom_id)
    body_half_extents = tuple(float(value) for value in ((torso_max - torso_min) * 0.5))

    body_xmat = np.asarray(data.xmat[torso_body_id], dtype=np.float64).reshape(3, 3)
    body_xpos = np.asarray(data.xpos[torso_body_id], dtype=np.float64)
    mount_points: list[tuple[float, float, float]] = []
    leg_lengths: list[float] = []
    for leg_body_id, foot_site_id in zip(leg_body_ids, foot_site_ids, strict=True):
        mount_world = np.asarray(data.xpos[leg_body_id], dtype=np.float64)
        foot_world = np.asarray(data.site_xpos[foot_site_id], dtype=np.float64)
        mount_body = body_xmat.T @ (mount_world - body_xpos)
        mount_points.append(tuple(float(value) for value in mount_body))
        leg_lengths.append(float(np.linalg.norm(foot_world - mount_world)))

    first_leg_min, first_leg_max = _mesh_geom_bounds(model, data, leg_geom_ids[0])
    first_leg_dims = first_leg_max - first_leg_min
    leg_radius = float(max(first_leg_dims[0], first_leg_dims[1]) * 0.5)

    robot_min_z = min(_mesh_geom_bounds(model, data, geom_id)[0][2] for geom_id in [torso_geom_id, *leg_geom_ids])
    floor_height = float(data.geom_xpos[ground_geom_id][2])
    reset_height = float(floor_height - robot_min_z)
    total_mass = float(model.body_subtreemass[torso_body_id])

    return RobotGeometry(
        body_half_extents_m=body_half_extents,
        leg_length_m=max(leg_lengths),
        leg_radius_m=leg_radius,
        reset_height_m=reset_height,
        floor_height_m=floor_height,
        mount_points_body=tuple(mount_points),
        total_mass_kg=total_mass,
    )
