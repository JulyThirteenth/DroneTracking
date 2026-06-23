"""Text-scene parsing and Isaac Sim geometry helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def ned_to_enu(point) -> np.ndarray:
    north, east, down = np.asarray(point, dtype=float).reshape(3)
    return np.array([east, north, -down], dtype=float)


def select_text_file(
    path: Path,
    *,
    index: int | None,
    list_only: bool,
    label: str,
) -> Path | None:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".txt":
            raise ValueError(f"Expected a .txt {label} file: {path}")
        if list_only:
            print(path)
            return None
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"{label.capitalize()} path not found: {path}")

    files = sorted(item for item in path.iterdir() if item.suffix.lower() == ".txt")
    if not files:
        raise FileNotFoundError(f"No .txt {label} files under {path}")

    for file_index, item in enumerate(files):
        print(f"[{file_index}] {item.name}")
    if list_only:
        return None

    if index is None:
        index = int(input(f"Select {label} index: ").strip())
    if not 0 <= index < len(files):
        raise ValueError(f"{label} index must be in [0, {len(files) - 1}]")
    return files[index]


def _numeric_rows(path: Path):
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield line_number, line.replace(",", " ").split()


def load_waypoints_ned(path: Path) -> list[np.ndarray]:
    points = []
    for line_number, columns in _numeric_rows(path):
        if len(columns) < 3:
            raise ValueError(f"Invalid waypoint row at {path}:{line_number}")
        points.append(np.asarray(columns[:3], dtype=float))
    return points


def _pivot(values: np.ndarray) -> list[float] | None:
    return values.tolist() if np.all(np.isfinite(values)) else None


def parse_scene(path: Path) -> tuple[list[dict], list[dict]]:
    cuboids: list[dict] = []
    gates: list[dict] = []

    for line_number, columns in _numeric_rows(path):
        kind = columns[0].lower()
        try:
            values = np.asarray(columns[2:], dtype=float)
        except ValueError as error:
            raise ValueError(f"Invalid number at {path}:{line_number}") from error

        if kind == "cuboid" and len(columns) == 16:
            cuboids.append(
                dict(
                    name=columns[1],
                    center=values[0:3],
                    size=values[3:6],
                    yaw=values[6],
                    pivot=_pivot(values[7:10]),
                    color=values[10:13],
                    opacity=values[13],
                )
            )
        elif kind == "gate" and len(columns) == 17:
            gates.append(
                dict(
                    name=columns[1],
                    center=values[0:3],
                    yaw=values[3],
                    pivot=_pivot(values[4:7]),
                    height=values[7],
                    outer=values[8],
                    inner=values[9],
                    thickness=values[10],
                    color=values[11:14],
                    opacity=values[14],
                )
            )
        else:
            raise ValueError(f"Invalid {kind!r} row at {path}:{line_number}")

    return cuboids, gates


def _rotated_center(center, yaw_deg: float, pivot):
    center = np.asarray(center, dtype=float)
    if pivot is None or yaw_deg == 0.0:
        return center

    angle = math.radians(float(yaw_deg))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    pivot = np.asarray(pivot, dtype=float)
    result = center.copy()
    result[:2] = (center[:2] - pivot[:2]) @ rotation.T + pivot[:2]
    return result


def spawn_cuboid(
    stage, path: str, *, center, size, yaw=0.0, pivot=None, color, opacity
):
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Gf, UsdGeom

    if prim_utils.is_prim_path_valid(path):
        return

    center = _rotated_center(center, float(yaw), pivot)
    half_yaw = 0.5 * math.radians(float(yaw))

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    transform = UsdGeom.Xformable(cube.GetPrim())
    transform.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    transform.AddOrientOp().Set(
        Gf.Quatf(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    )
    transform.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))

    geometry = UsdGeom.Gprim(cube.GetPrim())
    geometry.CreateDisplayColorAttr().Set([Gf.Vec3f(*map(float, color))])
    geometry.CreateDisplayOpacityAttr().Set([float(opacity)])


def spawn_gate(stage, path: str, **spec) -> None:
    center = np.asarray(spec.pop("center"), dtype=float)
    yaw = float(spec.pop("yaw"))
    pivot = spec.pop("pivot") or center
    height = float(spec.pop("height"))
    outer = float(spec.pop("outer"))
    inner = float(spec.pop("inner"))
    thickness = float(spec.pop("thickness"))
    edge = (outer - inner) / 2.0
    x, y, _ = center

    parts = {
        "base": ([x, y, (height - outer) / 2.0], [thickness, outer, height - outer]),
        "bottom": ([x, y, height - outer + edge / 2.0], [thickness, inner, edge]),
        "top": ([x, y, height - edge / 2.0], [thickness, inner, edge]),
        "left": (
            [x, y - (outer + inner) / 4.0, height - outer / 2.0],
            [thickness, edge, outer],
        ),
        "right": (
            [x, y + (outer + inner) / 4.0, height - outer / 2.0],
            [thickness, edge, outer],
        ),
    }
    for name, (part_center, size) in parts.items():
        spawn_cuboid(
            stage,
            f"{path}/{name}",
            center=part_center,
            size=size,
            yaw=yaw,
            pivot=pivot,
            **spec,
        )


def load_scene(stage, scene_path: Path, root: str = "/World/layout/PreEnv") -> None:
    import isaacsim.core.utils.prims as prim_utils

    if not prim_utils.is_prim_path_valid(root):
        prim_utils.create_prim(root, "Xform")

    cuboids, gates = parse_scene(scene_path)
    for item in cuboids:
        name = item.pop("name")
        spawn_cuboid(stage, f"{root}/{name}", **item)

    gate_root = f"{root}/Gates"
    if not prim_utils.is_prim_path_valid(gate_root):
        prim_utils.create_prim(gate_root, "Xform")
    for item in gates:
        name = item.pop("name")
        spawn_gate(stage, f"{gate_root}/{name}", **item)


def draw_waypoints(waypoints_ned, path: str = "/World/layout/Waypoints") -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    positions = [Gf.Vec3f(*map(float, ned_to_enu(point))) for point in waypoints_ned]
    points = UsdGeom.Points.Define(stage, path)
    points.CreatePointsAttr(positions)
    points.CreateWidthsAttr([0.05] * len(positions))

    geometry = UsdGeom.Gprim(points.GetPrim())
    geometry.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
    geometry.CreateDisplayOpacityAttr().Set([0.5])
