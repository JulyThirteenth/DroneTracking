import threading
import time
import json
from pathlib import Path
from typing import Callable, Optional, Literal

import numpy as np
import cv2
from PIL import Image


class TaskRecorder:
    def __init__(
        self,
        session_dir: str,
        get_frame: Callable[[], Optional[np.ndarray]],
        fps: int = 20,
    ):
        self.session_dir = Path(session_dir)
        self._grab_frame = get_frame
        self.fps = fps

        self._recording = False
        self._thread: Optional[threading.Thread] = None
        self._task_start_time: Optional[float] = None

        self.log_entries: list[dict] = []
        self.frame_timestamps: list[tuple[float, int]] = []  # [(abs_time, frame_idx), ...]
        self.frame_count: int = 0

    def __enter__(self):
        (self.session_dir / "frames").mkdir(parents=True, exist_ok=True)

        self._recording = True
        self._task_start_time = time.time()

        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

        print(f"Recording started → {self.session_dir}  ({self.fps} fps)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        # Save logs
        log_path = self.session_dir / "logs.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in self.log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Save frame timestamps
        ts_path = self.session_dir / "frame_timestamps.json"
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump({
                "fps": self.fps,
                "timestamps": [
                    {"abs_time": t, "frame_idx": i}
                    for t, i in self.frame_timestamps
                ],
            }, f, ensure_ascii=False)

        print(f"Recording finished: {self.frame_count} frames, {len(self.log_entries)} log entries")
        return False

    def _record_loop(self):
        interval = 1.0 / self.fps
        while self._recording:
            try:
                frame = self._grab_frame()
                if frame is not None:
                    idx = self.frame_count
                    img = Image.fromarray(frame)
                    img.save(self.session_dir / "frames" / f"{idx:06d}.jpg",
                             quality=90)
                    self.frame_timestamps.append((time.time(), idx))
                    self.frame_count += 1
            except Exception:
                pass
            time.sleep(interval)

    def log(self,
            type: Literal["agent_reply", "tool_call", "tool_reply"],
            text: str = "",
            tool_name: str = "",
            args: str = "") -> None:
        """Record one agent log entry with a relative timestamp."""
        if self._task_start_time is None:
            return
        entry = {
            "timestamp": round(time.time() - self._task_start_time, 3),
            "type": type,
        }
        if text:
            entry["text"] = text
        if tool_name:
            entry["tool_name"] = tool_name
        if args:
            entry["args"] = args
        self.log_entries.append(entry)

    @staticmethod
    def stitch(
        session_dir: str,
        output_name: str = "task.mp4",
        log_area_h: int = 150,
        log_line_h: int = 30,
        codec: str = "mp4v",
        font_path: str = "",
    ) -> Path:
        import textwrap
        from PIL import ImageDraw, ImageFont
        
        session_path = Path(session_dir)
        frames_dir = session_path / "frames"
        log_path = session_path / "logs.jsonl"
        ts_path = session_path / "frame_timestamps.json"
        output_path = session_path / output_name

        with open(log_path, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f]

        with open(ts_path, "r", encoding="utf-8") as f:
            ts_data = json.load(f)
        fps = ts_data.get("fps", 20)
        frame_timestamps = [(entry["abs_time"], entry["frame_idx"]) for entry in ts_data["timestamps"]]


        raw_frames = sorted(frames_dir.glob("*.jpg"))
        if not raw_frames:
            raise FileNotFoundError(f"No frame files found: {frames_dir}")

        first_frame = cv2.imread(str(raw_frames[0]))
        assert first_frame is not None, f"Failed to read {raw_frames[0]}"
        h, w = first_frame.shape[:2]

        canvas_h = h + log_area_h

        fourcc = cv2.VideoWriter.fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, canvas_h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter with codec '{codec}'")

        if font_path:
            font = ImageFont.truetype(font_path, 14)
        else:
            font = ImageFont.load_default(14)

        frame_time_map = {idx: abs_t for abs_t, idx in frame_timestamps}
        task_start = frame_timestamps[0][0] if frame_timestamps else 0
        logs_sorted = sorted(logs, key=lambda x: x["timestamp"])
        prefix_map = {"agent_reply": "[A]", "tool_call": "[T]", "tool_reply": "[R]"}

        log_idx = 0
        max_lines = log_area_h // log_line_h

        for idx, fp in enumerate(raw_frames):
            frame = cv2.imread(str(fp))
            assert frame is not None
            canvas = np.zeros((canvas_h, w, 3), dtype=np.uint8)
            canvas[:h, :] = frame

            frame_abs_t = frame_time_map.get(idx, task_start + idx / fps)
            frame_rel_t = frame_abs_t - task_start

            while log_idx < len(logs_sorted) and logs_sorted[log_idx]["timestamp"] <= frame_rel_t:
                log_idx += 1
            current_logs = logs_sorted[max(0, log_idx - 10):log_idx]

            lines_info = []
            if current_logs:
                latest_t = current_logs[-1]["timestamp"]
            for entry in current_logs:
                is_active = (frame_rel_t - entry["timestamp"] <= 1.0) or (latest_t - entry["timestamp"] <= 1.0)
                
                pfx = prefix_map.get(entry["type"], "  ")
                txt = entry.get("text", "") or f"{entry.get('tool_name', '')}({entry.get('args', '')})"
                paragraphs = [p for p in txt.split('\n') if p.strip()]
                
                for i, paragraph in enumerate(paragraphs):
                    wrapped = textwrap.wrap(paragraph, width=90) or [""]
                    for j, line_str in enumerate(wrapped):
                        if i == 0 and j == 0:
                            lines_info.append({"text": f"{pfx} {line_str}", "is_active": is_active})
                        else:
                            lines_info.append({"text": f"    {line_str}", "is_active": is_active})

            canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(canvas_pil)

            y_cursor = canvas_h - log_line_h
            for line in reversed(lines_info):
                if y_cursor < h and not line["is_active"]:
                    break

                fill_color = (255, 80, 80) if line["is_active"] else (255, 255, 255)
                stroke_w = 2 if line["is_active"] else 0

                try:
                    draw.text((10, y_cursor), line["text"], font=font, fill=fill_color, 
                              stroke_width=stroke_w, stroke_fill=(0, 0, 0))
                except TypeError:
                    draw.text((10, y_cursor), line["text"], font=font, fill=fill_color)

                y_cursor -= log_line_h
            writer.write(cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR))
            if idx % 100 == 0:
                print(f"Render progress: {idx + 1}/{len(raw_frames)}")

        writer.release()
        print(f"MP4 generated: {output_path}  ({fps} fps, {w}x{canvas_h})")
        return output_path
