from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def ned_to_enu(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([v[1], v[0], -v[2]], dtype=float)


def _prim_utils():
    import isaacsim.core.utils.prims as prim_utils

    return prim_utils


def load_waypoints_ned(path: Path):
    waypoints = []
    if not path.exists():
        return waypoints

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2]) if len(parts) >= 3 else 0.0
        waypoints.append(np.array([x, y, z], dtype=float))
    return waypoints


def _read_scene_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"scene file not found: {path}")

    rows: list[list[str]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError(f"scene file parse error at {path}:{lineno}")
        rows.append(parts)
    return rows


def _optional_pivot(pivot_xyz: np.ndarray):
    if np.all(np.isfinite(pivot_xyz)):
        return pivot_xyz.tolist()
    return None


def load_scene_specs(path: Path) -> tuple[list[dict], list[dict]]:
    """
    Load scene specs from a single txt file.

    Cuboid row (16 cols):
    cuboid name cx cy cz sx sy sz yaw_deg pivot_x pivot_y pivot_z r g b opacity

    Gate row (17 cols):
    gate name cx cy cz yaw_deg pivot_x pivot_y pivot_z height outer inner thick r g b opacity
    """
    cuboids: list[dict] = []
    gates: list[dict] = []
    for parts in _read_scene_rows(path):
        kind = parts[0].lower()
        if kind == "cuboid":
            if len(parts) != 16:
                raise ValueError(
                    f"cuboid parse error at {path}, expected 16 columns but got {len(parts)}"
                )
            name = parts[1]
            vals = np.asarray(parts[2:], dtype=float)
            cuboids.append(
                {
                    "name": name,
                    "center": vals[0:3].tolist(),
                    "size": vals[3:6].tolist(),
                    "yaw_deg": float(vals[6]),
                    "pivot": _optional_pivot(vals[7:10]),
                    "color": (float(vals[10]), float(vals[11]), float(vals[12])),
                    "opacity": float(vals[13]),
                }
            )
            continue
        if kind == "gate":
            if len(parts) != 17:
                raise ValueError(
                    f"gate parse error at {path}, expected 17 columns but got {len(parts)}"
                )
            name = parts[1]
            vals = np.asarray(parts[2:], dtype=float)
            gates.append(
                {
                    "name": name,
                    "center": vals[0:3].tolist(),
                    "yaw_deg": float(vals[3]),
                    "pivot": _optional_pivot(vals[4:7]),
                    "height": float(vals[7]),
                    "outer": float(vals[8]),
                    "inner": float(vals[9]),
                    "thick": float(vals[10]),
                    "color": (float(vals[11]), float(vals[12]), float(vals[13])),
                    "opacity": float(vals[14]),
                }
            )
            continue
        raise ValueError(f"unknown scene type {parts[0]!r} in {path}")
    return cuboids, gates


def rotate_z(point, yaw_deg: float, pivot):
    point = np.asarray(point, dtype=float).reshape(3)
    pivot = np.asarray(pivot, dtype=float).reshape(3)
    yaw = np.deg2rad(float(yaw_deg))
    rot = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return (point - pivot) @ rot.T + pivot


def spawn_cuboid(
    stage,
    prim_path: str,
    *,
    center,
    size,
    yaw_deg: float = 0.0,
    pivot=None,
    color=(0.2, 0.8, 0.2),
    opacity: float = 0.35,
):
    """
    Spawn an axis-aligned (optionally Z-yawed) cuboid using a scaled USD Cube.

    - `center`: (x,y,z) in IsaacSim world frame (ENU, meters).
    - `size`: (sx,sy,sz) edge lengths in meters.
    - `pivot`: if provided, rotates `center` around pivot by `yaw_deg` (matches env_utils behavior).
    """
    prim_utils = _prim_utils()
    if prim_utils.is_prim_path_valid(prim_path):
        return

    from pxr import Gf, UsdGeom

    center = np.asarray(center, dtype=float).reshape(3)
    size = np.asarray(size, dtype=float).reshape(3)
    if pivot is not None and float(yaw_deg) != 0.0:
        center = rotate_z(center, yaw_deg=float(yaw_deg), pivot=pivot)

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)

    xform = UsdGeom.Xformable(cube.GetPrim())
    quat_xyzw = Rotation.from_euler("Z", float(yaw_deg), degrees=True).as_quat()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))
    xform.AddOrientOp().Set(
        Gf.Quatf(
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        )
    )
    xform.AddScaleOp().Set(Gf.Vec3f(*size.tolist()))

    gprim = UsdGeom.Gprim(cube.GetPrim())
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*map(float, color))])
    gprim.CreateDisplayOpacityAttr().Set([float(opacity)])


def spawn_gate(
    stage,
    prim_path: str,
    *,
    center,
    yaw_deg: float = 0.0,
    pivot=None,
    height: float = 2.374,
    outer: float = 1.9,
    inner: float = 1.5,
    thick: float = 0.2,
    color=(0.2, 0.4, 0.9),
    opacity: float = 0.20,
):
    """
    Spawn a "gate" that matches `examples/pre/env_utils.py::plot_gate` geometry.

    Gate is composed of 5 cuboids (base, bottom, upper, left, right).
    """
    prim_utils = _prim_utils()
    if prim_utils.is_prim_path_valid(prim_path):
        return

    center = np.asarray(center, dtype=float).reshape(3)
    cx, cy, _cz = map(float, center.tolist())
    if pivot is None:
        pivot = center

    base_h = float(height - outer)
    edge = float((outer - inner) / 2.0)

    base_center = [cx, cy, base_h / 2.0]
    bottom_center = [cx, cy, base_h + edge / 2.0]
    upper_center = [cx, cy, height - edge / 2.0]
    left_center = [cx, cy - (outer + inner) / 4.0, base_h + outer / 2.0]
    right_center = [cx, cy + (outer + inner) / 4.0, base_h + outer / 2.0]

    spawn_cuboid(
        stage,
        f"{prim_path}/base",
        center=base_center,
        size=[thick, outer, base_h],
        yaw_deg=yaw_deg,
        pivot=pivot,
        color=color,
        opacity=opacity,
    )
    spawn_cuboid(
        stage,
        f"{prim_path}/bottom",
        center=bottom_center,
        size=[thick, inner, edge],
        yaw_deg=yaw_deg,
        pivot=pivot,
        color=color,
        opacity=opacity,
    )
    spawn_cuboid(
        stage,
        f"{prim_path}/upper",
        center=upper_center,
        size=[thick, inner, edge],
        yaw_deg=yaw_deg,
        pivot=pivot,
        color=color,
        opacity=opacity,
    )
    spawn_cuboid(
        stage,
        f"{prim_path}/left",
        center=left_center,
        size=[thick, edge, outer],
        yaw_deg=yaw_deg,
        pivot=pivot,
        color=color,
        opacity=opacity,
    )
    spawn_cuboid(
        stage,
        f"{prim_path}/right",
        center=right_center,
        size=[thick, edge, outer],
        yaw_deg=yaw_deg,
        pivot=pivot,
        color=color,
        opacity=opacity,
    )


def generate_scene(
    stage,
    *,
    env_root: str = "/World/layout/PreEnv",
    middle_wall_name: str = "MiddleWall",
    gate_root_name: str = "Gates",
    scene_file: Path | None = None,
):
    """
    Create scene content from a single scene txt file.
    """
    prim_utils = _prim_utils()
    if not prim_utils.is_prim_path_valid(env_root):
        prim_utils.create_prim(env_root, "Xform")

    scene_dir = Path(__file__).resolve().parent.parent / "scenes"
    scene_path = scene_dir / "scene.txt" if scene_file is None else Path(scene_file)
    cuboid_specs, gate_specs = load_scene_specs(scene_path)

    for spec in cuboid_specs:
        name = middle_wall_name if spec["name"] == "MiddleWall" else spec["name"]
        spawn_cuboid(
            stage,
            f"{env_root}/{name}",
            center=spec["center"],
            size=spec["size"],
            yaw_deg=spec["yaw_deg"],
            pivot=spec["pivot"],
            color=spec["color"],
            opacity=spec["opacity"],
        )

    gate_root = f"{env_root}/{gate_root_name}"
    if not prim_utils.is_prim_path_valid(gate_root):
        prim_utils.create_prim(gate_root, "Xform")

    for spec in gate_specs:
        spawn_gate(
            stage,
            f"{gate_root}/{spec['name']}",
            center=spec["center"],
            yaw_deg=spec["yaw_deg"],
            pivot=spec["pivot"],
            height=spec["height"],
            outer=spec["outer"],
            inner=spec["inner"],
            thick=spec["thick"],
            color=spec["color"],
            opacity=spec["opacity"],
        )


def generate_waypoint(
    stage,
    *,
    task_path: Path | None = None,
    waypoints_ned=None,
    parent_path: str = "/World/layout/Waypoints",
    radius: float = 0.05,
    color=(0.0, 1.0, 0.0),
    opacity: float = 0.1,
):
    """
    Create transparent waypoint sphere markers (no collisions).

    Pass either:
    - `task_path`: file path with NED waypoints (x y z per line)
    - `waypoints_ned`: iterable of 3D points in NED
    """
    if waypoints_ned is None:
        if task_path is None:
            raise ValueError("Either task_path or waypoints_ned must be provided")
        waypoints_ned = load_waypoints_ned(Path(task_path))

    waypoints_ned = list(waypoints_ned)
    if not waypoints_ned:
        return 0

    prim_utils = _prim_utils()
    if not prim_utils.is_prim_path_valid(parent_path):
        prim_utils.create_prim(parent_path, "Xform")

    from pxr import Gf, UsdGeom

    created = 0
    for i, wp_ned in enumerate(waypoints_ned):
        wp_enu = ned_to_enu(wp_ned)
        prim_path = f"{parent_path}/wp_{i:02d}"
        if prim_utils.is_prim_path_valid(prim_path):
            continue
        prim_utils.create_prim(
            prim_path,
            "Sphere",
            position=wp_enu,
            attributes={"radius": float(radius)},
        )
        prim = stage.GetPrimAtPath(prim_path)
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*map(float, color))])
        gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
        created += 1
    return created


