#!/usr/bin/env python3
"""Generate MuJoCo STL/XML assets from STEP CAD files.

Thanks to Wanqi Xiao for writing an instructional article that this is based off of.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.dom import minidom
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_ROOT = PROJECT_ROOT / "assets" / "mujoco"
DEFAULT_STEP_DIR = PROJECT_ROOT / "battlebot_step"
DEFAULT_MESH_DIR = MUJOCO_ROOT / "meshes" / "battlebot"
DEFAULT_ROBOT_XML = MUJOCO_ROOT / "battlebot.xml"
DEFAULT_SCENE_XML = MUJOCO_ROOT / "battlebot_scene.xml"
DEFAULT_MANIFEST = DEFAULT_MESH_DIR / "manifest.json"
GROUND_GRID_XML = MUJOCO_ROOT / "ground_grid.xml"
DEFAULT_MESH_SCALE = ("0.001", "0.001", "0.001")
BATTLEBOT_CHASSIS_RGBA = "0.82 0.82 0.78 1"
BATTLEBOT_BLACK_RGBA = "0.02 0.018 0.015 1"
BATTLEBOT_WHEEL_RGBA = "0.015 0.014 0.013 1"
BATTLEBOT_WEAPON_POS = (-0.0104775, -0.013284057, 0.009372451)
BATTLEBOT_WEAPON_QUAT = (0.5, 0.5, 0.5, 0.5)
BATTLEBOT_WHEEL_LAYOUT = (
    ("rupture_mini_wheel_left_geom", (-0.021191381, 0.004314427, 0.005547544), (0.70710678, 0.0, -0.70710678, 0.0)),
    ("rupture_mini_wheel_right_geom", (0.021191381, 0.004314427, 0.005547544), (0.70710678, 0.0, 0.70710678, 0.0)),
)


@dataclass(frozen=True)
class CadPart:
    index: int
    source_step: str
    kind: str
    stl_file: str
    mesh_name: str
    geom_name: str
    mass_kg: float


def _sanitize_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip().lower()).strip("_")
    if not name:
        name = "part"
    if name[0].isdigit():
        name = f"part_{name}"
    return name


def _load_occ() -> dict[str, Any]:
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_SOLID
        from OCC.Core.TopExp import TopExp_Explorer
    except ImportError as exc:
        raise SystemExit(
            "pythonocc-core is required for STEP conversion.\n"
            "Install it in a CAD/conversion environment, for example:\n"
            "  conda install -c conda-forge pythonocc-core\n"
            "Then rerun this script. Runtime simulation does not need this dependency."
        ) from exc

    return {
        "BRepMesh_IncrementalMesh": BRepMesh_IncrementalMesh,
        "IFSelect_RetDone": IFSelect_RetDone,
        "STEPControl_Reader": STEPControl_Reader,
        "StlAPI_Writer": StlAPI_Writer,
        "TopAbs_SHELL": TopAbs_SHELL,
        "TopAbs_SOLID": TopAbs_SOLID,
        "TopExp_Explorer": TopExp_Explorer,
    }


def _shape_hash(shape: Any) -> int:
    try:
        return int(shape.HashCode(2_147_483_647))
    except Exception:
        return id(shape)


def _explore_shapes(root_shape: Any, shape_kind: Any, occ: dict[str, Any]) -> list[Any]:
    explorer = occ["TopExp_Explorer"](root_shape, shape_kind)
    shapes: list[Any] = []
    seen: set[int] = set()
    while explorer.More():
        shape = explorer.Current()
        key = _shape_hash(shape)
        if key not in seen:
            seen.add(key)
            shapes.append(shape)
        explorer.Next()
    return shapes


def _select_shapes(root_shape: Any, part_mode: str, occ: dict[str, Any]) -> list[tuple[str, Any]]:
    solids = [("solid", shape) for shape in _explore_shapes(root_shape, occ["TopAbs_SOLID"], occ)]
    shells = [("shell", shape) for shape in _explore_shapes(root_shape, occ["TopAbs_SHELL"], occ)]

    if part_mode == "solids":
        return solids
    if part_mode == "shells":
        return shells
    if part_mode == "solids-and-shells":
        return solids + shells
    if part_mode == "shells-if-no-solids":
        return solids if solids else shells
    raise ValueError(f"Unsupported part_mode {part_mode!r}.")


def _read_step_shape(step_path: Path, occ: dict[str, Any]) -> Any:
    reader = occ["STEPControl_Reader"]()
    status = reader.ReadFile(str(step_path))
    if status != occ["IFSelect_RetDone"]:
        raise RuntimeError(f"OpenCascade could not read STEP file: {step_path}")
    transferred = reader.TransferRoots()
    if transferred <= 0:
        raise RuntimeError(f"OpenCascade found no transferable roots in STEP file: {step_path}")
    return reader.OneShape()


def _write_shape_stl(shape: Any, stl_path: Path, linear_deflection: float, angular_deflection: float, occ: dict[str, Any]) -> None:
    mesh = occ["BRepMesh_IncrementalMesh"](shape, float(linear_deflection), False, float(angular_deflection), True)
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("OpenCascade meshing did not complete.")

    writer = occ["StlAPI_Writer"]()
    if hasattr(writer, "SetASCIIMode"):
        writer.SetASCIIMode(False)
    ok = writer.Write(shape, str(stl_path))
    if ok is False:
        raise RuntimeError(f"OpenCascade STL writer failed for {stl_path}")
    if not stl_path.exists() or stl_path.stat().st_size == 0:
        raise RuntimeError(f"OpenCascade did not create a non-empty STL: {stl_path}")


def extract_step_parts(
    step_paths: list[Path],
    mesh_dir: Path,
    *,
    part_mode: str,
    linear_deflection: float,
    angular_deflection: float,
    total_mass_kg: float,
) -> list[CadPart]:
    occ = _load_occ()
    mesh_dir.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[Path, str, Any]] = []
    for step_path in step_paths:
        root_shape = _read_step_shape(step_path, occ)
        file_shapes = _select_shapes(root_shape, part_mode, occ)
        print(f"{step_path.name}: detected {len(file_shapes)} exported {part_mode} part(s)")
        for kind, shape in file_shapes:
            selected.append((step_path, kind, shape))

    if not selected:
        raise RuntimeError("No solids or shells were selected from the STEP inputs.")

    mass_per_part = float(total_mass_kg) / float(len(selected))
    parts: list[CadPart] = []
    for index, (step_path, kind, shape) in enumerate(selected, start=1):
        base_name = _sanitize_name(step_path.stem)
        part_name = f"{base_name}_{kind}_{index:03d}"
        stl_path = mesh_dir / f"{part_name}.stl"
        _write_shape_stl(shape, stl_path, linear_deflection, angular_deflection, occ)
        part = CadPart(
            index=index,
            source_step=str(step_path),
            kind=kind,
            stl_file=stl_path.name,
            mesh_name=f"{part_name}_mesh",
            geom_name=f"{part_name}_geom",
            mass_kg=mass_per_part,
        )
        parts.append(part)
        print(f"created {stl_path}")

    return parts


def _relative_posix(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def _format_float(value: float) -> str:
    return f"{value:.9g}"


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(_format_float(value) for value in values)


def _find_part(parts: list[CadPart], token: str) -> CadPart | None:
    matches = [part for part in parts if token in part.mesh_name.lower()]
    return matches[0] if len(matches) == 1 else None


def _add_mesh_geom(
    body: ET.Element,
    *,
    name: str,
    mesh: str,
    mass_kg: float,
    friction: str,
    rgba: str,
    pos: tuple[float, float, float] | None = None,
    quat: tuple[float, float, float, float] | None = None,
) -> None:
    attrs = {
        "name": name,
        "type": "mesh",
        "mesh": mesh,
        "mass": _format_float(mass_kg),
        "friction": friction,
        "rgba": rgba,
    }
    if pos is not None:
        attrs["pos"] = _format_vec(pos)
    if quat is not None:
        attrs["quat"] = _format_vec(quat)
    ET.SubElement(body, "geom", **attrs)


def _write_battlebot_assembly_geoms(body: ET.Element, parts: list[CadPart]) -> bool:
    chassis = _find_part(parts, "chassis")
    weapon = _find_part(parts, "weapon")
    wheel = _find_part(parts, "wheel")
    if chassis is None or weapon is None or wheel is None:
        return False

    total_mass_kg = sum(part.mass_kg for part in parts)
    _add_mesh_geom(
        body,
        name=chassis.geom_name,
        mesh=chassis.mesh_name,
        mass_kg=0.5 * total_mass_kg,
        friction="0.9 0.01 0.001",
        rgba=BATTLEBOT_CHASSIS_RGBA,
    )
    _add_mesh_geom(
        body,
        name=weapon.geom_name,
        mesh=weapon.mesh_name,
        pos=BATTLEBOT_WEAPON_POS,
        quat=BATTLEBOT_WEAPON_QUAT,
        mass_kg=0.25 * total_mass_kg,
        friction="0.9 0.01 0.001",
        rgba=BATTLEBOT_BLACK_RGBA,
    )
    for geom_name, pos, quat in BATTLEBOT_WHEEL_LAYOUT:
        _add_mesh_geom(
            body,
            name=geom_name,
            mesh=wheel.mesh_name,
            pos=pos,
            quat=quat,
            mass_kg=0.125 * total_mass_kg,
            friction="1.1 0.02 0.001",
            rgba=BATTLEBOT_WHEEL_RGBA,
        )
    return True


def write_robot_xml(
    parts: list[CadPart],
    output_xml: Path,
    mesh_dir: Path,
    *,
    model_name: str,
    mesh_scale: tuple[float, float, float],
    body_pos: tuple[float, float, float],
    rgba: str,
) -> None:
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", model=model_name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="true")
    ET.SubElement(
        root,
        "option",
        timestep="0.002500",
        gravity="0 0 -9.810000",
        integrator="implicitfast",
        solver="Newton",
    )

    asset = ET.SubElement(root, "asset")
    scale = _format_vec(mesh_scale)
    for part in parts:
        stl_path = mesh_dir / part.stl_file
        ET.SubElement(
            asset,
            "mesh",
            name=part.mesh_name,
            file=_relative_posix(stl_path, output_xml.parent),
            scale=scale,
        )

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="4", margin="0.002000", solref="0.005 1")
    ET.SubElement(default, "joint", armature="0.01")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")

    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        name=model_name,
        pos=_format_vec(body_pos),
    )
    ET.SubElement(body, "freejoint", name="root_free")
    if model_name != "battlebot" or not _write_battlebot_assembly_geoms(body, parts):
        for part in parts:
            _add_mesh_geom(
                body,
                name=part.geom_name,
                mesh=part.mesh_name,
                mass_kg=part.mass_kg,
                friction="0.9 0.01 0.001",
                rgba=rgba,
            )

    pretty_xml = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    output_xml.write_text(pretty_xml, encoding="utf-8")
    print(f"created {output_xml}")


def write_scene_xml(robot_xml: Path, scene_xml: Path, *, ground_xml: Path) -> None:
    scene_xml.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", model=f"{robot_xml.stem}_scene")
    ET.SubElement(root, "include", file=_relative_posix(ground_xml, scene_xml.parent))
    ET.SubElement(root, "include", file=_relative_posix(robot_xml, scene_xml.parent))
    pretty_xml = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    scene_xml.write_text(pretty_xml, encoding="utf-8")
    print(f"created {scene_xml}")


def write_manifest(parts: list[CadPart], manifest_path: Path, args: argparse.Namespace) -> None:
    manifest = {
        "model_name": args.model_name,
        "mesh_dir": str(args.mesh_dir),
        "robot_xml": str(args.robot_xml),
        "scene_xml": None if args.no_scene else str(args.scene_xml),
        "ground_xml": str(GROUND_GRID_XML),
        "part_mode": args.part_mode,
        "linear_deflection": args.linear_deflection,
        "angular_deflection": args.angular_deflection,
        "mesh_scale": args.mesh_scale,
        "total_mass_kg": args.total_mass_kg,
        "parts": [asdict(part) for part in parts],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"created {manifest_path}")


def _parse_vec3(values: list[str], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} requires exactly 3 numeric values.")
    return float(values[0]), float(values[1]), float(values[2])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        dest="step_paths",
        action="append",
        type=Path,
        help="STEP file to convert. May be passed more than once. Defaults to battlebot_step/*.step.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help="Directory for generated battlebot STL meshes.",
    )
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=DEFAULT_ROBOT_XML,
        help="Generated robot MJCF. Defaults to assets/mujoco/battlebot.xml.",
    )
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=DEFAULT_SCENE_XML,
        help="Generated scene MJCF that includes ground_grid.xml and the robot XML.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Generated conversion manifest.",
    )
    parser.add_argument("--model-name", default="battlebot")
    parser.add_argument(
        "--part-mode",
        choices=("solids", "shells", "solids-and-shells", "shells-if-no-solids"),
        default="shells-if-no-solids",
        help="Use shells mode if a STEP file visually has separate shells inside a composite solid.",
    )
    parser.add_argument("--linear-deflection", type=float, default=0.001)
    parser.add_argument("--angular-deflection", type=float, default=0.5)
    parser.add_argument("--total-mass-kg", type=float, default=1.0)
    parser.add_argument(
        "--mesh-scale",
        nargs=3,
        default=DEFAULT_MESH_SCALE,
        metavar=("SX", "SY", "SZ"),
        help="STEP-to-MuJoCo mesh scale. Defaults to 0.001 for millimeter CAD coordinates.",
    )
    parser.add_argument("--body-pos", nargs=3, default=("0", "0", "0.05"), metavar=("X", "Y", "Z"))
    parser.add_argument("--rgba", default="0.82 0.82 0.82 1")
    parser.add_argument("--no-scene", action="store_true", help="Only write the robot XML, not a scene XML.")
    args = parser.parse_args(argv)
    args.mesh_scale = _parse_vec3(list(args.mesh_scale), "mesh-scale")
    args.body_pos = _parse_vec3(list(args.body_pos), "body-pos")
    if args.total_mass_kg <= 0.0:
        parser.error("--total-mass-kg must be > 0")
    if args.linear_deflection <= 0.0:
        parser.error("--linear-deflection must be > 0")
    if args.angular_deflection <= 0.0:
        parser.error("--angular-deflection must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    step_paths = args.step_paths
    if not step_paths:
        step_paths = sorted(DEFAULT_STEP_DIR.glob("*.step"))
    step_paths = [path.resolve() for path in step_paths]
    missing = [path for path in step_paths if not path.exists()]
    if missing:
        raise SystemExit("Missing STEP input(s): " + ", ".join(str(path) for path in missing))
    if not step_paths:
        raise SystemExit(f"No STEP files found in {DEFAULT_STEP_DIR}")

    mesh_dir = args.mesh_dir.resolve()
    robot_xml = args.robot_xml.resolve()
    scene_xml = args.scene_xml.resolve()
    manifest_path = args.manifest.resolve()
    args.mesh_dir = mesh_dir
    args.robot_xml = robot_xml
    args.scene_xml = scene_xml
    args.manifest = manifest_path

    parts = extract_step_parts(
        step_paths,
        mesh_dir,
        part_mode=args.part_mode,
        linear_deflection=args.linear_deflection,
        angular_deflection=args.angular_deflection,
        total_mass_kg=args.total_mass_kg,
    )
    write_robot_xml(
        parts,
        robot_xml,
        mesh_dir,
        model_name=_sanitize_name(args.model_name),
        mesh_scale=args.mesh_scale,
        body_pos=args.body_pos,
        rgba=args.rgba,
    )
    if not args.no_scene:
        write_scene_xml(robot_xml, scene_xml, ground_xml=GROUND_GRID_XML)
    write_manifest(parts, manifest_path, args)
    print(f"conversion complete with {len(parts)} part(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
