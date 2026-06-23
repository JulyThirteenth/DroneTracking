from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_SPAWN_ENU = [0.0, 0.0, 0.07]


def prompt_index(*, count: int, prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please input an integer.")
            continue
        if 0 <= idx < count:
            return idx
        print(f"Index out of range: {idx} (0..{count - 1})")


def resolve_scene_path(path: Path, *, script_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    for candidate in (
        path.resolve(),
        (script_dir / path).resolve(),
        (script_dir.parent / path).resolve(),
    ):
        if candidate.exists():
            return candidate
    return path.resolve()


def spawn_position_from_arg(spawn: list[float] | None) -> list[float]:
    return [float(v) for v in (spawn or DEFAULT_SPAWN_ENU)]


def select_txt_scene(
    scene_arg: Path,
    *,
    scene_index: int | None,
    list_scenes: bool,
    script_dir: Path,
) -> Path | None:
    return select_txt_file(
        scene_arg,
        file_index=scene_index,
        list_files=list_scenes,
        script_dir=script_dir,
        title="Scenes dir",
        prompt="Select scene index: ",
        missing_message="No txt scene files found under",
        unsupported_message="unsupported txt scene path",
    )


def select_waypoint_file(
    waypoints_arg: Path,
    *,
    waypoint_index: int | None,
    list_waypoints: bool,
    script_dir: Path,
) -> Path | None:
    return select_txt_file(
        waypoints_arg,
        file_index=waypoint_index,
        list_files=list_waypoints,
        script_dir=script_dir,
        title="Waypoints dir",
        prompt="Select waypoint index: ",
        missing_message="No waypoint txt files found under",
        unsupported_message="unsupported waypoint path",
    )


def select_txt_file(
    path_arg: Path,
    *,
    file_index: int | None,
    list_files: bool,
    script_dir: Path,
    title: str,
    prompt: str,
    missing_message: str,
    unsupported_message: str,
) -> Path | None:
    scene_path = resolve_scene_path(path_arg, script_dir=script_dir)

    if scene_path.is_dir():
        scene_files = sorted(
            p
            for p in scene_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        )
        if not scene_files:
            raise FileNotFoundError(f"{missing_message}: {scene_path}")

        if list_files or file_index is None:
            print(f"{title}: {scene_path}")
            for i, txt_path in enumerate(scene_files):
                print(f"[{i}] {txt_path.name}")

        if list_files:
            return None

        if file_index is None:
            file_index = prompt_index(
                count=len(scene_files),
                prompt=prompt,
            )
        elif file_index < 0 or file_index >= len(scene_files):
            raise ValueError(
                f"index out of range: {file_index} (0..{len(scene_files) - 1})"
            )

        return scene_files[file_index]

    if not scene_path.exists():
        raise FileNotFoundError(f"path not found: {scene_path}")

    if scene_path.suffix.lower() == ".txt":
        if list_files:
            print(f"Txt file: {scene_path}")
            return None
        return scene_path

    raise ValueError(f"{unsupported_message}: {scene_path}")


def select_behavior1k_scene(
    scene_root: Path,
    *,
    scene_index: int | None,
    list_scenes: bool,
    script_dir: Path,
) -> tuple[Path | None, Path]:
    root = resolve_scene_path(scene_root, script_dir=script_dir)
    scenes_file = root / "scenes.txt"
    if not scenes_file.exists():
        raise FileNotFoundError(f"behavior1k scenes.txt not found: {scenes_file}")

    scene_files: list[Path] = []
    for raw in scenes_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        scene_path = Path(line)
        if not scene_path.is_absolute():
            scene_path = scenes_file.parent / scene_path
        if scene_path.exists():
            scene_files.append(scene_path.resolve())

    if not scene_files:
        raise FileNotFoundError(f"No behavior1k scene files found under: {root}")

    if list_scenes or scene_index is None:
        print(f"Behavior1k scenes: {scenes_file}")
        for i, scene_path in enumerate(scene_files):
            print(f"[{i}] {scene_path.parent.name}")

    if list_scenes:
        return None, root / "MTL"

    if scene_index is None:
        scene_index = prompt_index(
            count=len(scene_files),
            prompt="Select behavior1k scene index: ",
        )
    elif scene_index < 0 or scene_index >= len(scene_files):
        raise ValueError(
            f"scene_index out of range: {scene_index} (0..{len(scene_files) - 1})"
        )

    return scene_files[scene_index], root / "MTL"




def as_vec3(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


def ned_to_enu(value) -> np.ndarray:
    value = as_vec3(value)
    return np.array([value[1], value[0], -value[2]], dtype=float)


def load_waypoints_ned(path: Path) -> list[np.ndarray]:
    waypoints: list[np.ndarray] = []
    if not path.exists():
        return waypoints

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) >= 3:
            waypoints.append(np.asarray(parts[:3], dtype=float))
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
    import isaacsim.core.utils.prims as prim_utils

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
    import isaacsim.core.utils.prims as prim_utils

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
    gate_root_name: str = "Gates",
    scene_file: Path | None = None,
):
    """
    Create scene content from a single scene txt file.
    """
    import isaacsim.core.utils.prims as prim_utils

    if not prim_utils.is_prim_path_valid(env_root):
        prim_utils.create_prim(env_root, "Xform")

    scene_dir = Path(__file__).resolve().parent.parent / "scenes"
    scene_path = scene_dir / "scene.txt" if scene_file is None else Path(scene_file)
    cuboid_specs, gate_specs = load_scene_specs(scene_path)

    for spec in cuboid_specs:
        spawn_cuboid(
            stage,
            f"{env_root}/{spec['name']}",
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
    waypoints_ned,
    parent_path: str = "/World/layout/Waypoints",
    radius: float = 0.05,
    color=(0.0, 1.0, 0.0),
    opacity: float = 0.1,
):
    """
    Draw waypoint markers in the viewport only.

    These markers are not USD geometry, so they do not collide and do not appear
    in camera RGB/depth rendering. `stage` and `parent_path` are kept only for
    call-site compatibility.

    `waypoints_ned` is an iterable of NED points.
    """
    waypoints_ned = list(waypoints_ned)
    if not waypoints_ned:
        return 0

    from isaacsim.util.debug_draw import _debug_draw

    _ = stage, parent_path
    points = [tuple(ned_to_enu(wp_ned).tolist()) for wp_ned in waypoints_ned]
    rgba = tuple(map(float, color)) + (float(opacity),)
    size = max(1.0, float(radius) * 200.0)

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_points(points, [rgba] * len(points), [size] * len(points))
    return len(points)
