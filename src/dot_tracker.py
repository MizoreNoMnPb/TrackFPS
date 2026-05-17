"""Track player dots: HoughCircles detection + greedy nearest-neighbor matching."""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DotTracker:
    """Track dots with greedy nearest-neighbor matching."""

    def __init__(self, max_gap: int = 5, max_dist: float = 80.0):
        self.max_gap = max_gap
        self.max_dist = max_dist
        self.tracks = {}  # id -> deque of (x, y, frame_num)
        self.next_id = 0
        self._gaps = {}   # id -> consecutive misses

    def update(self, dets: list, frame_num: int):
        """Update tracks. dets = [(x, y), ...]."""
        if not dets:
            for tid in list(self._gaps.keys()):
                self._gaps[tid] += 1
                if self._gaps[tid] > self.max_gap:
                    del self.tracks[tid]
                    del self._gaps[tid]
            return

        # Predict current positions (linear extrapolation from last 2 points)
        predicted = {}
        for tid, pts in self.tracks.items():
            if len(pts) >= 2:
                dx = pts[-1][0] - pts[-2][0]
                dy = pts[-1][1] - pts[-2][1]
                predicted[tid] = (pts[-1][0] + dx, pts[-1][1] + dy)
            elif len(pts) == 1:
                predicted[tid] = (pts[-1][0], pts[-1][1])

        # Greedy matching: nearest neighbor
        matched_dets = set()
        matched_trks = set()
        for tid, (px, py) in sorted(predicted.items(),
                                     key=lambda x: len(self.tracks[x[0]]),
                                     reverse=True):
            best_d = float('inf')
            best_i = -1
            for i, (dx, dy) in enumerate(dets):
                if i in matched_dets:
                    continue
                d = np.sqrt((dx - px)**2 + (dy - py)**2)
                if d < best_d and d < self.max_dist:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                matched_dets.add(best_i)
                matched_trks.add(tid)
                self.tracks[tid].append((dets[best_i][0], dets[best_i][1], frame_num))
                self._gaps[tid] = 0

        # Increment gaps for unmatched tracks
        for tid in list(predicted.keys()):
            if tid not in matched_trks:
                self._gaps[tid] = self._gaps.get(tid, 0) + 1
                if self._gaps[tid] > self.max_gap:
                    del self.tracks[tid]
                    del self._gaps[tid]

        # Create new tracks for unmatched dets
        for i in range(len(dets)):
            if i not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = deque([(dets[i][0], dets[i][1], frame_num)])
                self._gaps[tid] = 0

    def get_trajectories(self) -> dict:
        result = {}
        for tid, pts in self.tracks.items():
            if len(pts) > 0:
                result[tid] = [(x, y, fn) for x, y, fn in pts]
        return result


def detect_dots(frame, color_lower, color_upper):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array(color_lower, np.uint8),
                             np.array(color_upper, np.uint8))
    circles = cv2.HoughCircles(color_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                               param1=50, param2=12, minRadius=5, maxRadius=9)
    if circles is None:
        return []
    circles = np.uint16(np.around(circles))[0]
    dots = []
    for x, y, r in circles:
        if x < 15 or y < 15 or x > w - 15 or y > h - 15:
            continue
        patch = frame[y-7:y+7, x-7:x+7]
        if patch.size == 0:
            continue
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        patch_color = cv2.inRange(patch_hsv, np.array(color_lower, np.uint8),
                                  np.array(color_upper, np.uint8))
        if (patch_color > 0).sum() / patch_color.size > 0.25:
            dots.append((int(x), int(y)))
    unique = []
    for x, y in dots:
        if not any(abs(x - ux) < 10 and abs(y - uy) < 10 for ux, uy in unique):
            unique.append((x, y))
    return unique[:5]


def track_dots(map_dir: Path, color_lower: tuple, color_upper: tuple,
               max_gap: int = 5, max_dist: float = 80.0) -> dict:
    paths = sorted(map_dir.glob("frame_*.jpg"))
    tracker = DotTracker(max_gap=max_gap, max_dist=max_dist)

    for fi, fp in enumerate(paths):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        fn = int(fp.stem.replace("frame_", ""))
        dots = detect_dots(frame, color_lower, color_upper)
        tracker.update(dots, fn)

    traj = tracker.get_trajectories()
    result = {}
    for tid, pts in traj.items():
        if len(pts) < 50:
            continue
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        if np.sqrt(dx*dx + dy*dy) > 10:
            result[tid] = pts
    return result


def draw_trajectories(map_region_path: str, trajectories: dict,
                      output_path: str, map_w: int = 854, map_h: int = 852):
    bg = cv2.imread(map_region_path)
    if bg is None:
        bg = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    else:
        bg = cv2.resize(bg, (map_w, map_h))
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255),
              (255, 0, 255), (0, 128, 255), (255, 128, 0)]
    for ti, (tid, pts) in enumerate(trajectories.items()):
        c = colors[ti % len(colors)]
        for j in range(1, len(pts)):
            cv2.line(bg, (int(pts[j-1][0]), int(pts[j-1][1])),
                     (int(pts[j][0]), int(pts[j][1])), c, 2)
        for x, y, _ in pts[::30]:
            cv2.circle(bg, (int(x), int(y)), 7, c, 1)
    cv2.imwrite(output_path, bg)
    logger.info(f"Trajectories saved to {output_path}")
