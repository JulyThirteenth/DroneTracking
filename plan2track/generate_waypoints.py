#!/usr/bin/env python3
"""Interactive waypoint generator based on minimum-snap interpolation.

Default mode opens an Open3D GUI editor. Use `--headless` for CLI generation.
Inputs are specified in ENU coordinates. Saved waypoint files are written in NED
to match the existing task-file format used by the tracker.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent

for extra_path in (_THIS_DIR / "minimum_snap", _ROOT / "isaacsim"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from minimum_snap import MinSnap, Polynomial
from sim_utils import load_scene_specs, rotate_z


DEFAULT_WAYPOINTS_ENU = np.array(
    [
        [14.0, 15.0, 1.424],
        [26.0, 14.05, 1.424],
        [23.5, 5.95, 1.424],
        [7.25, 6.40, 1.424],
        [9.0, 15.0, 1.424],
    ],
    dtype=float,
)
DEFAULT_SEGMENT_TIMES = np.full(DEFAULT_WAYPOINTS_ENU.shape[0] - 1, 10.0, dtype=float)
DEFAULT_RESOLUTION_M = 0.1
DEFAULT_TASKS_DIR = _THIS_DIR / "tasks"
DEFAULT_WAYPOINTS_DIR = _THIS_DIR / "waypoints"
DEFAULT_OUTPUT = DEFAULT_WAYPOINTS_DIR / "demo_waypoint_1.txt"
DEFAULT_TASK_OUTPUT = DEFAULT_TASKS_DIR / "demo_task_1_convert.txt"
DEFAULT_SCENE_FILE: Path | None = _ROOT / "scenes"


@dataclass
class GenerationResult:
    control_points_enu: np.ndarray
    segment_times: np.ndarray
    resolution_m: float
    sampled_enu: np.ndarray
    scene_file: Path | None


@dataclass
class WaypointRowWidgets:
    row: object
    index_label: object
    x_edit: object
    y_edit: object
    z_edit: object
    dt_edit: object


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    return np.stack([arr[:, 1], arr[:, 0], -arr[:, 2]], axis=1)


def _parse_waypoints_text(value: str) -> np.ndarray:
    normalized = value.replace(";", "\n")
    rows: list[list[float]] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [float(x.strip()) for x in stripped.split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError(
                "Each waypoint must contain exactly 3 values in 'x, y, z' format."
            )
        rows.append(parts)
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 3:
        raise ValueError("Need at least 2 waypoints shaped as x,y,z per line.")
    return arr


def _parse_times_text(value: str, *, expected: int) -> np.ndarray:
    parts = [float(x.strip()) for x in value.replace(";", ",").split(",") if x.strip()]
    if len(parts) == 1:
        parts = parts * expected
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} segment times, got {len(parts)}.")
    arr = np.asarray(parts, dtype=float)
    if np.any(arr <= 0.0):
        raise ValueError("All segment times must be > 0.")
    return arr


def _format_waypoints_text(points_enu: np.ndarray) -> str:
    pts = np.asarray(points_enu, dtype=float).reshape(-1, 3)
    return "; ".join(f"{x:.3f}, {y:.3f}, {z:.3f}" for x, y, z in pts)


def _format_times_text(times: np.ndarray) -> str:
    return ", ".join(f"{float(t):.3f}" for t in np.asarray(times, dtype=float).reshape(-1))


def _segment_times_to_row_times(segment_times: np.ndarray) -> np.ndarray:
    seg = np.asarray(segment_times, dtype=float).reshape(-1)
    return np.concatenate([[0.0], seg])


def _row_times_to_segment_times(row_times: np.ndarray) -> np.ndarray:
    row = np.asarray(row_times, dtype=float).reshape(-1)
    if row.size < 2:
        return np.zeros((0,), dtype=float)
    if np.any(row[1:] <= 0.0):
        raise ValueError("All waypoint arrival times after the first row must be > 0.")
    return row[1:]


def _resolve_scene_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / "scene.txt"
    return candidate


def _dense_sample_minimum_snap(
    waypoints_enu: np.ndarray,
    segment_times: np.ndarray,
    *,
    resolution_m: float,
) -> np.ndarray:
    piece_num = int(waypoints_enu.shape[0] - 1)
    zeros = np.zeros((1, 3), dtype=float)
    coeffs = MinSnap.gen_poly_traj(
        waypoints_enu,
        segment_times,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
    )

    dense_points: list[np.ndarray] = []
    for seg_idx in range(piece_num):
        poly = Polynomial(coeffs[seg_idx].T)
        chord = float(
            np.linalg.norm(waypoints_enu[seg_idx + 1] - waypoints_enu[seg_idx])
        )
        dt = float(segment_times[seg_idx])
        sample_count = max(
            int(math.ceil(max(chord, resolution_m) / max(resolution_m / 5.0, 1e-3))),
            200,
        )
        ts = np.linspace(0.0, dt, sample_count + 1, dtype=float)
        if seg_idx > 0:
            ts = ts[1:]
        dense_points.append(poly.pos(ts).T)
    return np.vstack(dense_points)


def _resample_polyline(points_enu: np.ndarray, resolution_m: float) -> np.ndarray:
    pts = np.asarray(points_enu, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pts.copy()
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        return pts[:1].copy()

    s_query = np.arange(0.0, total, resolution_m, dtype=float)
    if s_query.size == 0 or abs(s_query[-1] - total) > 1e-9:
        s_query = np.append(s_query, total)

    out = np.zeros((s_query.size, 3), dtype=float)
    for idx, s in enumerate(s_query):
        seg_i = int(np.searchsorted(cum, s, side="right") - 1)
        seg_i = max(0, min(seg_i, pts.shape[0] - 2))
        seg_len = float(seg[seg_i])
        if seg_len <= 1e-9:
            out[idx] = pts[seg_i]
            continue
        alpha = (s - float(cum[seg_i])) / seg_len
        out[idx] = pts[seg_i] + alpha * (pts[seg_i + 1] - pts[seg_i])
    return out


def compute_generation(
    *,
    waypoints_enu: np.ndarray,
    segment_times: np.ndarray,
    resolution_m: float,
    scene_file: Path | None,
) -> GenerationResult:
    dense_enu = _dense_sample_minimum_snap(
        waypoints_enu,
        segment_times,
        resolution_m=resolution_m,
    )
    sampled_enu = _resample_polyline(dense_enu, resolution_m)
    return GenerationResult(
        control_points_enu=np.asarray(waypoints_enu, dtype=float).reshape(-1, 3),
        segment_times=np.asarray(segment_times, dtype=float).reshape(-1),
        resolution_m=float(resolution_m),
        sampled_enu=sampled_enu,
        scene_file=scene_file,
    )


def _write_waypoints_ned(path: Path, sampled_enu: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sampled_ned = enu_to_ned(sampled_enu)
    with path.open("w", encoding="utf-8") as f:
        f.write("# x y z (NED)\n")
        for x, y, z in sampled_ned:
            f.write(f"{x:.3f}, {y:.3f}, {z:.3f},\n")


def _write_task_file(path: Path, waypoints_enu: np.ndarray, row_times: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(waypoints_enu, dtype=float).reshape(-1, 3)
    row_times = np.asarray(row_times, dtype=float).reshape(-1)
    if pts.shape[0] != row_times.shape[0]:
        raise ValueError("Task save failed: waypoint count and row-time count do not match.")
    with path.open("w", encoding="utf-8") as f:
        f.write("# task waypoints (ENU)\n")
        f.write("# columns: x, y, z, dt_from_prev_s\n")
        for (x, y, z), dt in zip(pts, row_times):
            f.write(f"{x:.3f}, {y:.3f}, {z:.3f}, {dt:.3f}\n")


def _read_task_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(x.strip()) for x in line.split(",") if x.strip()]
            if len(parts) != 4:
                raise ValueError(
                    "Task file must contain 4 comma-separated values per line: x, y, z, dt_from_prev_s"
                )
            rows.append(parts)
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 4:
        raise ValueError("Task file must contain at least 2 waypoint rows.")
    waypoints_enu = arr[:, :3]
    row_times = arr[:, 3]
    row_times[0] = 0.0
    if np.any(row_times[1:] <= 0.0):
        raise ValueError("All dt_from_prev_s values after the first row must be > 0.")
    return waypoints_enu, row_times


def _make_box_transformed(
    o3d,
    *,
    center,
    size,
    yaw_deg: float = 0.0,
    pivot=None,
    color=(0.5, 0.5, 0.5),
):
    center = np.asarray(center, dtype=float).reshape(3)
    size = np.asarray(size, dtype=float).reshape(3)
    if pivot is not None and float(yaw_deg) != 0.0:
        center = rotate_z(center, yaw_deg=float(yaw_deg), pivot=pivot)
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=float(size[0]),
        height=float(size[1]),
        depth=float(size[2]),
    )
    mesh.translate((-size[0] / 2.0, -size[1] / 2.0, -size[2] / 2.0))
    yaw = math.radians(float(yaw_deg))
    rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    mesh.rotate(rot, center=(0.0, 0.0, 0.0))
    mesh.translate(center)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.asarray(color, dtype=float).reshape(3))
    return mesh


def _make_gate_meshes(o3d, gate: dict) -> list:
    center = np.asarray(gate["center"], dtype=float).reshape(3)
    pivot = gate.get("pivot")
    yaw_deg = float(gate["yaw_deg"])
    height = float(gate["height"])
    outer = float(gate["outer"])
    inner = float(gate["inner"])
    thick = float(gate["thick"])
    color = gate.get("color", (0.2, 0.4, 0.9))

    cx, cy, _cz = map(float, center.tolist())
    if pivot is None:
        pivot = center

    base_h = float(height - outer)
    edge = float((outer - inner) / 2.0)

    specs = [
        ([cx, cy, base_h / 2.0], [thick, outer, base_h]),
        ([cx, cy, base_h + edge / 2.0], [thick, inner, edge]),
        ([cx, cy, height - edge / 2.0], [thick, inner, edge]),
        ([cx, cy - (outer + inner) / 4.0, base_h + outer / 2.0], [thick, edge, outer]),
        ([cx, cy + (outer + inner) / 4.0, base_h + outer / 2.0], [thick, edge, outer]),
    ]

    return [
        _make_box_transformed(
            o3d,
            center=box_center,
            size=box_size,
            yaw_deg=yaw_deg,
            pivot=pivot,
            color=color,
        )
        for box_center, box_size in specs
    ]


def _add_scene_preview(app_window, result: GenerationResult) -> None:
    o3d = app_window.o3d
    scene = app_window.scene_widget.scene
    app_window.clear_preview()

    mesh_material = app_window.rendering.MaterialRecord()
    mesh_material.shader = "defaultLit"
    line_material = app_window.rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = 3.0
    point_material = app_window.rendering.MaterialRecord()
    point_material.shader = "defaultUnlit"
    point_material.point_size = 8.0

    bounds_min = np.min(result.sampled_enu, axis=0)
    bounds_max = np.max(result.sampled_enu, axis=0)

    resolved_scene = _resolve_scene_path(result.scene_file)
    if resolved_scene is not None and str(resolved_scene).strip():
        cuboids, gates = load_scene_specs(resolved_scene)
    else:
        cuboids, gates = ([], [])

    for idx, cuboid in enumerate(cuboids):
        mesh = _make_box_transformed(
            o3d,
            center=cuboid["center"],
            size=cuboid["size"],
            yaw_deg=float(cuboid["yaw_deg"]),
            pivot=cuboid.get("pivot"),
            color=cuboid.get("color", (0.2, 0.8, 0.2)),
        )
        name = f"cuboid_{idx}"
        scene.add_geometry(name, mesh, mesh_material)
        app_window.geometry_names.append(name)
        bbox = mesh.get_axis_aligned_bounding_box()
        bounds_min = np.minimum(bounds_min, bbox.get_min_bound())
        bounds_max = np.maximum(bounds_max, bbox.get_max_bound())

    gate_mesh_idx = 0
    for gate in gates:
        for mesh in _make_gate_meshes(o3d, gate):
            name = f"gate_{gate_mesh_idx}"
            gate_mesh_idx += 1
            scene.add_geometry(name, mesh, mesh_material)
            app_window.geometry_names.append(name)
            bbox = mesh.get_axis_aligned_bounding_box()
            bounds_min = np.minimum(bounds_min, bbox.get_min_bound())
            bounds_max = np.maximum(bounds_max, bbox.get_max_bound())

    control_pcd = o3d.geometry.PointCloud()
    control_pcd.points = o3d.utility.Vector3dVector(result.control_points_enu)
    control_pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.5, 0.0]], dtype=float), (result.control_points_enu.shape[0], 1))
    )
    scene.add_geometry("control_points", control_pcd, point_material)
    app_window.geometry_names.append("control_points")

    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(result.sampled_enu)
    sampled_pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.0, 0.8, 0.2]], dtype=float), (result.sampled_enu.shape[0], 1))
    )
    scene.add_geometry("sampled_points", sampled_pcd, point_material)
    app_window.geometry_names.append("sampled_points")

    control_lines = [[i, i + 1] for i in range(result.control_points_enu.shape[0] - 1)]
    if control_lines:
        control_line_set = o3d.geometry.LineSet()
        control_line_set.points = o3d.utility.Vector3dVector(result.control_points_enu)
        control_line_set.lines = o3d.utility.Vector2iVector(control_lines)
        control_line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[1.0, 0.5, 0.0]], dtype=float), (len(control_lines), 1))
        )
        scene.add_geometry("control_lines", control_line_set, line_material)
        app_window.geometry_names.append("control_lines")

    sampled_lines = [[i, i + 1] for i in range(result.sampled_enu.shape[0] - 1)]
    if sampled_lines:
        sampled_line_set = o3d.geometry.LineSet()
        sampled_line_set.points = o3d.utility.Vector3dVector(result.sampled_enu)
        sampled_line_set.lines = o3d.utility.Vector2iVector(sampled_lines)
        sampled_line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 0.8, 0.2]], dtype=float), (len(sampled_lines), 1))
        )
        scene.add_geometry("sampled_lines", sampled_line_set, line_material)
        app_window.geometry_names.append("sampled_lines")

    bbox = o3d.geometry.AxisAlignedBoundingBox(bounds_min, bounds_max)
    scene.show_axes(True)
    app_window.scene_widget.setup_camera(60.0, bbox, bbox.get_center())


class MinimumSnapGui:
    def __init__(
        self,
        *,
        waypoints_enu: np.ndarray,
        segment_times: np.ndarray,
        resolution_m: float,
        output_path: Path,
        scene_file: Path | None,
    ):
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering

        self.o3d = o3d
        self.gui = gui
        self.rendering = rendering

        app = gui.Application.instance
        app.initialize()

        self.window = app.create_window("Minimum Snap Waypoint Tool", 1600, 960)
        self.geometry_names: list[str] = []
        self.current_result: GenerationResult | None = None
        self.row_widgets: list[WaypointRowWidgets] = []

        em = self.window.theme.font_size
        margin = int(0.5 * em)
        spacing = int(0.25 * em)

        self.panel = gui.Vert(spacing, gui.Margins(margin, margin, margin, margin))
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.98, 0.98, 0.99, 1.0])

        self.status_label = gui.Label("Ready")

        self.scene_file_edit = gui.TextEdit()
        self.scene_file_edit.text_value = "" if scene_file is None else str(scene_file)

        self.output_path_edit = gui.TextEdit()
        self.output_path_edit.text_value = str(output_path)

        self.task_path_edit = gui.TextEdit()
        self.task_path_edit.text_value = str(DEFAULT_TASK_OUTPUT)

        self.resolution_edit = gui.TextEdit()
        self.resolution_edit.text_value = f"{float(resolution_m):.3f}"
        self.selected_idx_edit = gui.TextEdit()
        self.selected_idx_edit.text_value = "0"
        self.table_proxy = gui.WidgetProxy()

        self.panel.add_child(gui.Label("Scene File (.txt, optional)"))
        self.panel.add_child(self.scene_file_edit)
        self.panel.add_child(gui.Label("Waypoint Table (ENU)"))
        self.panel.add_child(
            gui.Label("Each row is x, y, z, dt_from_prev(s). Row 0 time is fixed to 0.")
        )
        self.header = gui.Horiz(spacing)
        for label in ("Idx", "x", "y", "z", "dt_from_prev(s)"):
            self.header.add_child(gui.Label(label))
        self.panel.add_child(self.header)
        self.panel.add_child(self.table_proxy)

        self.waypoint_index_row = gui.Horiz(spacing)
        self.waypoint_index_row.add_child(gui.Label("Selected Idx"))
        self.waypoint_index_row.add_child(self.selected_idx_edit)
        self.panel.add_child(self.waypoint_index_row)

        self.waypoint_button_row = gui.Horiz(spacing)
        self.insert_btn = gui.Button("Insert Before Idx")
        self.delete_btn = gui.Button("Delete Idx")
        self.insert_btn.set_on_clicked(self.on_add_row)
        self.delete_btn.set_on_clicked(self.on_delete_row)
        self.waypoint_button_row.add_child(self.insert_btn)
        self.waypoint_button_row.add_child(self.delete_btn)
        self.panel.add_child(self.waypoint_button_row)

        self.panel.add_child(gui.Label("Sampling Resolution (m)"))
        self.panel.add_child(self.resolution_edit)
        self.panel.add_child(gui.Label("Output Waypoint File (NED .txt)"))
        self.panel.add_child(self.output_path_edit)

        self.panel.add_child(gui.Label("Task Waypoint File (ENU + dt .txt)"))
        self.panel.add_child(self.task_path_edit)

        self.task_button_row = gui.Horiz(spacing)
        self.load_task_btn = gui.Button("Load Task")
        self.save_task_btn = gui.Button("Save Task")
        self.load_task_btn.set_on_clicked(self.on_load_task)
        self.save_task_btn.set_on_clicked(self.on_save_task)
        self.task_button_row.add_child(self.load_task_btn)
        self.task_button_row.add_child(self.save_task_btn)
        self.panel.add_child(self.task_button_row)

        self.button_row = gui.Horiz(spacing)
        self.compute_btn = gui.Button("Compute")
        self.save_btn = gui.Button("Save")
        self.reset_btn = gui.Button("Reset Defaults")
        self.compute_btn.set_on_clicked(self.on_compute)
        self.save_btn.set_on_clicked(self.on_save)
        self.reset_btn.set_on_clicked(self.on_reset_defaults)
        self.button_row.add_child(self.compute_btn)
        self.button_row.add_child(self.save_btn)
        self.button_row.add_child(self.reset_btn)
        self.panel.add_child(self.button_row)
        self.panel.add_child(self.status_label)

        self.window.add_child(self.panel)
        self.window.add_child(self.scene_widget)
        self.window.set_on_layout(self.on_layout)

        self._rebuild_rows(
            np.asarray(waypoints_enu, dtype=float).reshape(-1, 3),
            _segment_times_to_row_times(segment_times),
        )
        self.on_compute()

    def clear_preview(self) -> None:
        for name in self.geometry_names:
            try:
                self.scene_widget.scene.remove_geometry(name)
            except Exception:
                pass
        self.geometry_names.clear()

    def on_layout(self, _layout_context) -> None:
        rect = self.window.content_rect
        panel_width = 520
        self.panel.frame = self.gui.Rect(rect.x, rect.y, panel_width, rect.height)
        self.scene_widget.frame = self.gui.Rect(
            rect.x + panel_width,
            rect.y,
            max(1, rect.width - panel_width),
            rect.height,
        )

    def set_status(self, text: str) -> None:
        self.status_label.text = text

    def _make_row(self, idx: int, point_enu: np.ndarray, dt_to_here: float) -> WaypointRowWidgets:
        row = self.gui.Horiz(4)
        index_label = self.gui.Label(str(idx))
        x_edit = self.gui.TextEdit()
        y_edit = self.gui.TextEdit()
        z_edit = self.gui.TextEdit()
        dt_edit = self.gui.TextEdit()

        point = np.asarray(point_enu, dtype=float).reshape(3)
        x_edit.text_value = f"{float(point[0]):.3f}"
        y_edit.text_value = f"{float(point[1]):.3f}"
        z_edit.text_value = f"{float(point[2]):.3f}"
        dt_edit.text_value = f"{float(dt_to_here):.3f}"
        if idx == 0:
            dt_edit.enabled = False

        row.add_child(index_label)
        row.add_child(x_edit)
        row.add_child(y_edit)
        row.add_child(z_edit)
        row.add_child(dt_edit)
        return WaypointRowWidgets(row, index_label, x_edit, y_edit, z_edit, dt_edit)

    def _build_table_widget(self, waypoints_enu: np.ndarray, row_times: np.ndarray):
        table_container = self.gui.ScrollableVert(
            4, self.gui.Margins(4, 4, 4, 4)
        )
        self.row_widgets.clear()
        for idx, (pt, dt_to_here) in enumerate(zip(waypoints_enu, row_times)):
            row_widget = self._make_row(idx, pt, float(dt_to_here))
            self.row_widgets.append(row_widget)
            table_container.add_child(row_widget.row)
        return table_container

    def _rebuild_rows(self, waypoints_enu: np.ndarray, row_times: np.ndarray) -> None:
        pts = np.asarray(waypoints_enu, dtype=float).reshape(-1, 3)
        row_times = np.asarray(row_times, dtype=float).reshape(-1)
        table_widget = self._build_table_widget(pts, row_times)
        self.table_proxy.set_widget(table_widget)
        if hasattr(self.window, "set_needs_layout"):
            self.window.set_needs_layout()

    def _selected_index(self, *, allow_end: bool = False) -> int:
        value = self.selected_idx_edit.text_value.strip()
        idx = int(value) if value else 0
        row_count = len(self.row_widgets)
        max_idx = row_count if allow_end else max(0, row_count - 1)
        idx = max(0, min(idx, max_idx))
        self.selected_idx_edit.text_value = str(idx)
        return idx

    def _read_table_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        points: list[list[float]] = []
        row_times: list[float] = []
        for idx, row_widget in enumerate(self.row_widgets):
            point = [
                float(row_widget.x_edit.text_value.strip()),
                float(row_widget.y_edit.text_value.strip()),
                float(row_widget.z_edit.text_value.strip()),
            ]
            if idx == 0:
                dt_to_here = 0.0
                row_widget.dt_edit.text_value = "0.000"
            else:
                dt_to_here = float(row_widget.dt_edit.text_value.strip())
            points.append(point)
            row_times.append(dt_to_here)

        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
            raise ValueError("Need at least 2 waypoint rows.")
        return pts, np.asarray(row_times, dtype=float)

    def _read_inputs(self) -> tuple[np.ndarray, np.ndarray, float, Path | None, Path]:
        waypoints_enu, row_times = self._read_table_inputs()
        segment_times = _row_times_to_segment_times(row_times)
        resolution_m = float(self.resolution_edit.text_value.strip())
        if not np.isfinite(resolution_m) or resolution_m <= 0.0:
            raise ValueError("Resolution must be > 0.")
        scene_text = self.scene_file_edit.text_value.strip()
        scene_file = _resolve_scene_path(Path(scene_text) if scene_text else None)
        output_path = Path(self.output_path_edit.text_value.strip())
        return waypoints_enu, segment_times, resolution_m, scene_file, output_path

    def _task_path(self) -> Path:
        text = self.task_path_edit.text_value.strip()
        if not text:
            raise ValueError("Task file path is empty.")
        return Path(text)

    def on_add_row(self) -> None:
        try:
            waypoints_enu, row_times = self._read_table_inputs()
            idx = self._selected_index()
            new_point = np.asarray(waypoints_enu[idx], dtype=float).copy()
            if idx == 0:
                new_dt = float(row_times[1]) if row_times.size > 1 else 10.0
            else:
                new_dt = float(row_times[idx])

            updated_waypoints = np.insert(waypoints_enu, idx, new_point, axis=0)
            updated_row_times = np.insert(row_times, idx, new_dt)
            updated_row_times[0] = 0.0
            if idx == 0 and updated_row_times.size > 1:
                updated_row_times[1] = new_dt

            self._rebuild_rows(updated_waypoints, updated_row_times)
            self.selected_idx_edit.text_value = str(idx)
            self.set_status(f"Inserted row before idx {idx}.")
        except Exception as exc:
            self.set_status(f"Add row failed: {exc}")

    def on_delete_row(self) -> None:
        try:
            waypoints_enu, row_times = self._read_table_inputs()
            if int(waypoints_enu.shape[0]) <= 2:
                raise ValueError("At least 2 waypoints must remain.")
            idx = self._selected_index()
            if idx < 0 or idx >= int(waypoints_enu.shape[0]):
                raise ValueError(f"Row index out of range: {idx}")

            updated_waypoints = np.delete(waypoints_enu, idx, axis=0)
            updated_row_times = np.delete(row_times, idx)
            updated_row_times[0] = 0.0

            self._rebuild_rows(updated_waypoints, updated_row_times)
            self.selected_idx_edit.text_value = str(min(idx, updated_waypoints.shape[0] - 1))
            self.set_status(f"Deleted row {idx}.")
        except Exception as exc:
            self.set_status(f"Delete row failed: {exc}")

    def on_compute(self) -> None:
        try:
            waypoints_enu, segment_times, resolution_m, scene_file, _output_path = (
                self._read_inputs()
            )
            result = compute_generation(
                waypoints_enu=waypoints_enu,
                segment_times=segment_times,
                resolution_m=resolution_m,
                scene_file=scene_file,
            )
            self.current_result = result
            _add_scene_preview(self, result)
            total_len = float(
                np.sum(np.linalg.norm(result.sampled_enu[1:] - result.sampled_enu[:-1], axis=1))
            )
            self.set_status(
                f"Computed {result.sampled_enu.shape[0]} points, length={total_len:.3f} m"
            )
        except Exception as exc:
            self.set_status(f"Compute failed: {exc}")

    def on_save(self) -> None:
        if self.current_result is None:
            self.set_status("Nothing to save. Compute first.")
            return
        try:
            _waypoints, _times, _resolution, _scene, output_path = self._read_inputs()
            _write_waypoints_ned(output_path, self.current_result.sampled_enu)
            self.set_status(f"Saved waypoints to {output_path}")
        except Exception as exc:
            self.set_status(f"Save failed: {exc}")

    def on_save_task(self) -> None:
        try:
            waypoints_enu, row_times = self._read_table_inputs()
            task_path = self._task_path()
            _write_task_file(task_path, waypoints_enu, row_times)
            self.set_status(f"Saved task waypoints to {task_path}")
        except Exception as exc:
            self.set_status(f"Save task failed: {exc}")

    def on_load_task(self) -> None:
        try:
            task_path = self._task_path()
            waypoints_enu, row_times = _read_task_file(task_path)
            self._rebuild_rows(waypoints_enu, row_times)
            self.selected_idx_edit.text_value = "0"
            self.set_status(f"Loaded task waypoints from {task_path}")
            self.on_compute()
        except Exception as exc:
            self.set_status(f"Load task failed: {exc}")

    def on_reset_defaults(self) -> None:
        self.scene_file_edit.text_value = (
            "" if DEFAULT_SCENE_FILE is None else str(DEFAULT_SCENE_FILE)
        )
        self.output_path_edit.text_value = str(DEFAULT_OUTPUT)
        self.task_path_edit.text_value = str(DEFAULT_TASK_OUTPUT)
        self.resolution_edit.text_value = f"{DEFAULT_RESOLUTION_M:.3f}"
        self._rebuild_rows(DEFAULT_WAYPOINTS_ENU, _segment_times_to_row_times(DEFAULT_SEGMENT_TIMES))
        self.on_compute()

    def run(self) -> None:
        self.gui.Application.instance.run()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive minimum-snap waypoint editor and generator."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Generate immediately without GUI.",
    )
    parser.add_argument(
        "--scene-file",
        type=Path,
        default=DEFAULT_SCENE_FILE,
        help="Optional scene txt file for visualization.",
    )
    parser.add_argument(
        "--waypoints-enu",
        type=str,
        default=_format_waypoints_text(DEFAULT_WAYPOINTS_ENU),
        help="ENU control points. Use one 'x, y, z' per line or ';' separators.",
    )
    parser.add_argument(
        "--segment-times",
        type=str,
        default=_format_times_text(DEFAULT_SEGMENT_TIMES),
        help="Segment durations in seconds as 't0, t1, ...'. One value repeats.",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=DEFAULT_RESOLUTION_M,
        help="Spatial sampling resolution in meters.",
    )
    parser.add_argument(
        "--output-waypoints",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output waypoint txt path (written in NED).",
    )
    return parser


def _run_headless(args) -> None:
    waypoints_enu = _parse_waypoints_text(str(args.waypoints_enu))
    segment_times = _parse_times_text(
        str(args.segment_times),
        expected=int(waypoints_enu.shape[0] - 1),
    )
    result = compute_generation(
        waypoints_enu=waypoints_enu,
        segment_times=segment_times,
        resolution_m=float(args.resolution_m),
        scene_file=_resolve_scene_path(args.scene_file),
    )
    _write_waypoints_ned(Path(args.output_waypoints), result.sampled_enu)
    total_length = float(
        np.sum(np.linalg.norm(result.sampled_enu[1:] - result.sampled_enu[:-1], axis=1))
    )
    print(f"Control waypoints (ENU): {result.control_points_enu.shape[0]}")
    print(f"Generated sampled waypoints: {result.sampled_enu.shape[0]}")
    print(f"Approx path length: {total_length:.3f} m")
    print(f"Output waypoint file: {Path(args.output_waypoints)}")


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.headless:
        _run_headless(args)
        return

    try:
        app = MinimumSnapGui(
            waypoints_enu=_parse_waypoints_text(str(args.waypoints_enu)),
            segment_times=_parse_times_text(
                str(args.segment_times),
                expected=int(_parse_waypoints_text(str(args.waypoints_enu)).shape[0] - 1),
            ),
            resolution_m=float(args.resolution_m),
            output_path=Path(args.output_waypoints),
            scene_file=args.scene_file,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for GUI mode. Install it in the current environment "
            "or run with --headless."
        ) from exc

    app.run()


if __name__ == "__main__":
    main()
