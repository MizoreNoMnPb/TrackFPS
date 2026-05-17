"""Track player dots: median background subtraction + HoughCircles + x-sort."""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def build_background(map_dir: Path, step: int = 10) -> np.ndarray:
    """Build median background from map frames (static map, no scrolling)."""
    paths = sorted(map_dir.glob("frame_*.jpg"))
    frames = [cv2.imread(str(paths[i])) for i in range(0, len(paths), step)
              if cv2.imread(str(paths[i])) is not None]
    return np.median(frames, axis=0).astype(np.uint8)


def detect_dots(frame, bg, color_lower, color_upper):
    """Detect player dots: foreground (diff from bg) + HoughCircles + color."""
    h, w = frame.shape[:2]

    # Foreground = current frame - median background
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg_gray)
    _, fg = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Only consider fg pixels with the right color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array(color_lower, np.uint8),
                             np.array(color_upper, np.uint8))
    fg_color = fg & color_mask

    # HoughCircles on foreground color mask
    circles = cv2.HoughCircles(fg_color, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=40, param2=10, minRadius=5, maxRadius=9)
    if circles is None:
        return []

    circles = np.uint16(np.around(circles))[0]
    dots = []
    for x, y, r in circles:
        if x < 20 or y < 20 or x > w - 20 or y > h - 20:
            continue
        # Verify blob area in fg_color matches expected dot size
        patch = fg_color[y-8:y+8, x-8:x+8]
        fg_area = (patch > 0).sum()
        if fg_area < 80:  # real 14x14 dot has ~120-180 fg pixels
            continue
        dots.append((int(x), int(y)))

    # Deduplicate nearby
    unique = []
    for x, y in dots:
        if not any(abs(x - ux) < 10 and abs(y - uy) < 10 for ux, uy in unique):
            unique.append((x, y))
    return unique[:5]


def track_dots(map_dir: Path, color_lower: tuple, color_upper: tuple) -> dict:
    """Per-frame detection + x-sort labeling."""
    paths = sorted(map_dir.glob("frame_*.jpg"))
    bg = build_background(map_dir)

    trajectories = {0: [], 1: [], 2: []}

    for fi, fp in enumerate(paths):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        fn = int(fp.stem.replace("frame_", ""))
        dots = detect_dots(frame, bg, color_lower, color_upper)
        dots.sort(key=lambda d: d[0])
        for i in range(min(3, len(dots))):
            trajectories[i].append((dots[i][0], dots[i][1], fn))

    result = {}
    for tid, pts in trajectories.items():
        if len(pts) < 50:
            continue
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        if np.sqrt(dx*dx + dy*dy) > 5:
            result[tid] = pts
    return result


def draw_trajectories(map_region_path: str, trajectories: dict,
                      output_path: str, map_w: int = 854, map_h: int = 852):
    bg = cv2.imread(map_region_path)
    if bg is None:
        bg = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    else:
        bg = cv2.resize(bg, (map_w, map_h))
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]
    for ti, (tid, pts) in enumerate(trajectories.items()):
        c = colors[ti % 3]
        for j in range(1, len(pts)):
            cv2.line(bg, (int(pts[j-1][0]), int(pts[j-1][1])),
                     (int(pts[j][0]), int(pts[j][1])), c, 2)
        for x, y, _ in pts[::30]:
            cv2.circle(bg, (int(x), int(y)), 7, c, 1)
    cv2.imwrite(output_path, bg)
    logger.info(f"Trajectories saved to {output_path}")
