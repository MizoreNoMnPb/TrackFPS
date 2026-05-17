"""Analyze game frames: match timer, team table (with ranking), player status, Mandel Brick.

Team identity is tracked by color via temporal-smoothed per-frame detection.
Ranking changes are debounced to avoid false positives from color instability.
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)

# ---- Layout constants (game-view coordinates) ----

TIMER_X, TIMER_Y = 454, 8
TIMER_W, TIMER_H = 92, 36

TABLE_X, TABLE_Y = 8, 208
TABLE_W, TABLE_H = 164, 168
HEADER_H = 18
ROW_H, ROW_GAP = 23, 2
ROW_STRIDE = ROW_H + ROW_GAP
DATA_START_Y = TABLE_Y + HEADER_H
NUM_TEAMS = 6

COLOR_BAR_X, COLOR_BAR_W = 16, 4
TEAM_NAME_X, TEAM_NAME_W = 25, 30
PLAYER_X = [93, 102, 110]
MANDEL_ROW_X = 174

MANDEL_ICON_W, MANDEL_ICON_H = 33, 33
MANDEL_TEXT_H = 20
MANDEL_ICON_X = TIMER_X + TIMER_W + 5
MANDEL_ICON_Y = TIMER_Y + (TIMER_H - MANDEL_ICON_H) // 2
MANDEL_TEXT_X = MANDEL_ICON_X + MANDEL_ICON_W + 6
MANDEL_TEXT_Y = TIMER_Y + (TIMER_H - MANDEL_TEXT_H) // 2

# Temporal smoothing: how many frames to look back
COLOR_WINDOW = 15
COLOR_DEBOUNCE = 5    # frames before ranking change confirmed

# Known hue centers for each team color (OpenCV H values)
HUE_CENTERS = {
    "red":    (0, 10, 170, 180),   # wraps around 0/180
    "orange": (12, 22),
    "yellow": (26, 34),
    "green":  (45, 75),
    "blue":   (100, 125),
    "purple": (135, 160),
    "white":  None,  # white has no hue, detected by low S + high V
}


def row_y(idx: int) -> int:
    return DATA_START_Y + idx * ROW_STRIDE


def _hue_in_range(h: int, lo: int, hi: int) -> bool:
    return lo <= h <= hi


class GameAnalyzer:
    """Extract game UI state with temporal smoothing and debounced ranking."""

    def __init__(self, template_dir: str = "assets/player_state",
                 teams_config: str = None):
        self._color_names = {}
        self._row_color_hist = [deque(maxlen=COLOR_WINDOW) for _ in range(NUM_TEAMS)]
        self._locked_rank = None
        self._pending_rank = None
        self._pending_frames = 0

        # Load player status templates
        self._templates = {}
        for s in ["alive", "knocked", "defeated", "eliminated"]:
            tmpl = cv2.imread(f"{template_dir}/{s}.png", cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                self._templates[s] = cv2.resize(tmpl, (7, 9))
        logger.info(f"Loaded {len(self._templates)} player templates from {template_dir}")

        # Load pre-annotated team data (optional, overrides OCR)
        self._teams_config = None
        if teams_config:
            import json
            with open(teams_config) as f:
                self._teams_config = json.load(f)
            logger.info(f"Loaded teams config: {len(self._teams_config['teams'])} teams")

    def analyze_view(self, view_dir: str) -> dict:
        game_dir = Path(view_dir) / "game"
        if not game_dir.exists():
            return {}
        paths = sorted(game_dir.glob("frame_*.jpg"))
        if not paths:
            return {}
        logger.info(f"Analyzing {len(paths)} game frames in {view_dir}")

        self._reset_state()

        # Find first frame where UI is fully loaded
        warmup = self._find_warmup_end(paths)
        if warmup > 0:
            logger.info(f"  Skipping {warmup} loading frames")
            paths = paths[warmup:]

        # Phase 1: OCR team names
        self._build_color_name_map(paths)

        # Phase 2: process all frames
        prev_players = None
        events = []
        frame_data = []

        for i, fp in enumerate(paths):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            timer = self._read_timer(frame)
            mandel = self._read_mandel(frame)

            # Skip frames where UI is covered (inventory, loading, etc.)
            if not timer["visible"]:
                frame_data.append({
                    "frame": fp.stem,
                    "timer": False,
                    "mandel_present": False,
                    "teams": [],
                })
                continue

            # Smoothed color per row → current ranking
            cur_rank = []
            for ti in range(NUM_TEAMS):
                ry = row_y(ti)
                row = frame[ry:ry + ROW_H, TABLE_X:TABLE_X + TABLE_W]
                raw_color = self._classify_color_peak(row)
                self._row_color_hist[ti].append(raw_color)
                # Majority vote over window
                smoothed = self._majority_vote(self._row_color_hist[ti])
                cur_rank.append(smoothed)

            # Debounce ranking changes
            self._update_ranking(cur_rank, fp.stem, i, events)

            # Use locked rank for team identity (fall back to smoothed)
            active_rank = self._locked_rank if self._locked_rank else cur_rank

            cur_players = {}
            teams = []
            for ti in range(NUM_TEAMS):
                ry = row_y(ti)
                row = frame[ry:ry + ROW_H, TABLE_X:TABLE_X + TABLE_W]
                color = active_rank[ti]
                players = self._read_players(row)
                cur_players[color] = [p["status"] for p in players]
                teams.append({
                    "idx": ti, "color": color,
                    "name": self._color_names.get(color),
                    "players": players,
                    "has_mandel": self._read_row_mandel(row),
                })

            # Player status changes
            if prev_players is not None:
                for color, new_statuses in cur_players.items():
                    if color == "unknown":
                        continue
                    old_statuses = prev_players.get(color)
                    if old_statuses is None:
                        continue
                    for pj in range(3):
                        old_s = old_statuses[pj]
                        new_s = new_statuses[pj]
                        if old_s != new_s and old_s != "unknown" and new_s != "unknown":
                            events.append({
                                "type": "player_status",
                                "frame": fp.stem, "frame_idx": i,
                                "team_color": color,
                                "team_name": self._color_names.get(color),
                                "player_idx": pj,
                                "from": old_statuses[pj], "to": new_statuses[pj],
                            })

            prev_players = cur_players
            frame_data.append({
                "frame": fp.stem,
                "timer": timer["visible"],
                "mandel_present": mandel["present"],
                "teams": teams,
            })

        summary = self._build_summary(frame_data, events)

        # Valid state transitions
        VALID_TRANSITIONS = {
            ("alive", "knocked"),
            ("knocked", "alive"),
            ("knocked", "defeated"),
            ("defeated", "alive"),
            ("alive", "eliminated"),     # last alive, no teammates
            ("knocked", "eliminated"),   # teammates eliminated
            ("defeated", "eliminated"),  # timeout or teammates eliminated
        }
        clean = [e for e in events
                 if e["type"] != "player_status"
                 or (e.get("from"), e.get("to")) in VALID_TRANSITIONS]
        self._export_csv(clean, view_dir)

        return {"frame_data": frame_data, "events": clean, "summary": summary}

    def _export_csv(self, events: list, view_dir: str):
        """Export events to CSV with player names from teams.json."""
        import csv
        out = Path(view_dir) / "events.csv"
        fps = 59.94

        # Load player order from teams config
        player_names = {}
        if self._teams_config:
            for t in self._teams_config["teams"]:
                player_names[t["color"]] = t["players"]

        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "video_time", "game_time", "type",
                        "team", "player", "detail", "extra"])
            for e in events:
                fn = e.get("frame", "")
                vs = int(int(fn.replace("frame_", "")) / fps) if fn else 0
                ts = f"{vs//60}:{vs%60:02d}"
                gt = e.get("game_time_str", "")

                if e["type"] == "player_status":
                    pnames = player_names.get(e.get("team_color", ""), ["?", "?", "?"])
                    pname = pnames[e.get("player_idx", 0)] if e.get("player_idx") is not None else ""
                    w.writerow([
                        fn, ts, gt, e["type"],
                        f'{e.get("team_name","")}({e.get("team_color","")})',
                        pname,
                        e.get("from", ""),
                        e.get("to", ""),
                    ])
                elif e["type"] == "ranking_change":
                    row = e.get("row", 0) + 1  # 1-indexed rank
                    w.writerow([
                        fn, ts, gt, e["type"],
                        f'{e.get("to_name","")}({e.get("to_color","")})',
                        f'Rank #{row}',
                        f'{e.get("from_name","")}({e.get("from_color","")})',
                        f'{e.get("to_name","")}({e.get("to_color","")})',
                    ])
                else:
                    w.writerow([fn, ts, gt, e.get("type",""), "", "", "", ""])

    # ================================================================
    # Temporal smoothing & ranking debounce
    # ================================================================

    def _reset_state(self):
        self._row_color_hist = [deque(maxlen=COLOR_WINDOW) for _ in range(NUM_TEAMS)]
        self._locked_rank = None
        self._pending_rank = None
        self._pending_frames = 0

    def _find_warmup_end(self, paths: list) -> int:
        """Return index of first frame where green timer bracket is visible.

        Timer bracket signals that the full UI has rendered.
        Scans up to 800 frames.
        """
        for i, fp in enumerate(paths[:800]):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            if self._read_timer(frame)["visible"]:
                return i
        return 0

    @staticmethod
    def _majority_vote(window: deque) -> str:
        """Return the most common non-unknown color in the window."""
        votes = {}
        for c in window:
            if c != "unknown":
                votes[c] = votes.get(c, 0) + 1
        if not votes:
            return "unknown"
        best = max(votes, key=votes.get)
        if votes[best] >= 2:  # at least 2 votes for narrow bar
            return best
        return "unknown"

    def _update_ranking(self, cur_rank: list, frame_name: str,
                        idx: int, events: list):
        """Debounced ranking change detection."""
        # If all rows are known and different from locked rank
        if "unknown" in cur_rank:
            self._pending_frames = 0
            self._pending_rank = None
            return

        if self._locked_rank is None:
            # First time we have full data
            self._locked_rank = list(cur_rank)
            return

        # Need at least 5 distinct colors (teams) to consider valid
        if len(set(cur_rank)) < 4:
            self._pending_frames = 0
            self._pending_rank = None
            return

        if cur_rank == self._locked_rank:
            self._pending_frames = 0
            self._pending_rank = None
            return

        # cur_rank differs from locked_rank
        if cur_rank == self._pending_rank:
            self._pending_frames += 1
        else:
            self._pending_rank = list(cur_rank)
            self._pending_frames = 1

        if self._pending_frames >= COLOR_DEBOUNCE:
            # Emit ranking change events
            for ti in range(NUM_TEAMS):
                old_c = self._locked_rank[ti]
                new_c = cur_rank[ti]
                if old_c != new_c:
                    events.append({
                        "type": "ranking_change",
                        "frame": frame_name, "frame_idx": idx,
                        "row": ti,
                        "from_color": old_c,
                        "to_color": new_c,
                        "from_name": self._color_names.get(old_c),
                        "to_name": self._color_names.get(new_c),
                    })
            self._locked_rank = list(cur_rank)
            self._pending_rank = None
            self._pending_frames = 0

    # ================================================================
    # Color classification (hue histogram peak)
    # ================================================================

    def _classify_color_peak(self, row: np.ndarray) -> str:
        """Classify color bar by finding the dominant hue peak."""
        bar = row[:, COLOR_BAR_X:COLOR_BAR_X + COLOR_BAR_W, :]
        if bar.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].flatten()
        s = hsv[:, :, 1].flatten()
        v = hsv[:, :, 2].flatten()

        # Check white first (low S, high V) — before saturated check
        white_mask = (s < 40) & (v > 180)
        white_ratio = white_mask.sum() / white_mask.size
        if white_ratio > 0.3:
            return "white"

        # Only saturated pixels vote for color
        mask = s > 50
        if mask.sum() < 3:
            return "unknown"

        h_masked = h[mask]

        # Build hue histogram
        hist, _ = np.histogram(h_masked, bins=36, range=(0, 180))
        peak_bin = np.argmax(hist)

        if hist[peak_bin] < 2:
            return "unknown"

        peak_h = peak_bin * 5 + 2.5  # center of bin

        # Map to nearest color center
        best_color = "unknown"
        best_score = 0
        for name, ranges in HUE_CENTERS.items():
            if ranges is None:
                continue
            if len(ranges) == 2:
                lo, hi = ranges
                if lo <= peak_h <= hi:
                    score = 1.0 - abs(peak_h - (lo + hi) / 2) / ((hi - lo) / 2)
                    if score > best_score:
                        best_score = score
                        best_color = name
            else:  # red wraps
                lo1, hi1, lo2, hi2 = ranges
                if lo1 <= peak_h <= hi1 or lo2 <= peak_h <= hi2:
                    center = (hi1 + lo2 + 180) / 2
                    if peak_h > 90:
                        dist = min(abs(peak_h - (lo1 + hi1) / 2),
                                   abs(peak_h - (lo2 + hi2) / 2))
                    else:
                        dist = abs(peak_h - (lo1 + hi1) / 2)
                    score = max(0, 1.0 - dist / 15)
                    if score > best_score:
                        best_score = score
                        best_color = name

        return best_color if best_score > 0 else "unknown"

    # ================================================================
    # Color → Name Map (OCR)
    # ================================================================

    def _build_color_name_map(self, paths: list):
        """Build color→name mapping from config or OCR."""
        if self._teams_config:
            # Use pre-annotated data
            self._color_names = {}
            self._player_names = {}
            for t in self._teams_config["teams"]:
                c = t["color"]
                self._color_names[c] = t["name"]
                for pj, pid in enumerate(t["players"]):
                    self._player_names[(c, pj)] = pid
            logger.info(f"Color→Name from config: {self._color_names}")
            return

        # Fallback: OCR (legacy)
        color_names = {}
        for fp in paths[:30]:
            frame = cv2.imread(str(fp))
            if frame is None: continue
            for ti in range(NUM_TEAMS):
                ry = row_y(ti)
                row = frame[ry:ry + ROW_H, TABLE_X:TABLE_X + TABLE_W]
                color = self._classify_color_peak(row)
                if color == "unknown": continue
                name = self._ocr_team_name(row)
                if name in ("?", ""): continue
                if color not in color_names or len(name) > len(color_names[color]):
                    color_names[color] = name
        self._color_names = color_names
        self._player_names = {}
        logger.info(f"Color→Name from OCR: {color_names}")

    def _ocr_team_name(self, row: np.ndarray) -> str:
        name_roi = row[:, TEAM_NAME_X:TEAM_NAME_X + TEAM_NAME_W, :]
        if name_roi.size == 0:
            return "?"
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            results = reader.readtext(name_roi, detail=0,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            return results[0] if results else "?"
        except ImportError:
            return "?"

    # ================================================================
    # Lightweight detectors
    # ================================================================

    def _read_timer(self, frame: np.ndarray) -> dict:
        roi = frame[TIMER_Y:TIMER_Y + TIMER_H, TIMER_X:TIMER_X + TIMER_W]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (40, 80, 40), (80, 255, 255))
        ratio = (green > 0).sum() / green.size
        return {"visible": ratio > 0.02, "green_ratio": round(float(ratio), 4)}

    def _read_players(self, row: np.ndarray) -> list:
        h, w = row.shape[:2]
        gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        players = []
        for px in PLAYER_X:
            x1, x2 = max(0, px - 4), min(w, px + 3)
            y1, y2 = max(0, h // 2 - 3), min(h, h // 2 + 6)
            icon = gray[y1:y2, x1:x2]
            if icon.shape[0] < 7 or icon.shape[1] < 7:
                players.append({"status": "unknown"})
                continue

            # Template matching against 4 status icons
            best_status = "unknown"
            best_score = -1
            for s, tmpl in self._templates.items():
                result = cv2.matchTemplate(icon, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_status = s

            # Require minimum confidence to avoid noise
            if best_score < 0.2:
                best_status = "unknown"

            players.append({
                "status": best_status,
                "score": round(float(best_score), 3),
            })
        return players

    def _read_mandel(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        ir = frame[MANDEL_ICON_Y:MANDEL_ICON_Y+MANDEL_ICON_H,
                   MANDEL_ICON_X:min(w, MANDEL_ICON_X+MANDEL_ICON_W)]
        present = False; ia = 0
        if ir.size > 0:
            o = cv2.inRange(cv2.cvtColor(ir, cv2.COLOR_BGR2HSV), (10,120,120), (25,255,255))
            ia = int((o > 0).sum()); present = ia > 10
        tr = frame[MANDEL_TEXT_Y:MANDEL_TEXT_Y+MANDEL_TEXT_H,
                   MANDEL_TEXT_X:min(w, MANDEL_TEXT_X+200)]
        tb = 0
        if tr.size > 0:
            tb = int(cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY).mean())
        return {"present": present, "icon_area": ia, "text_brightness": tb}

    def _read_row_mandel(self, row: np.ndarray) -> bool:
        h, w = row.shape[:2]
        if MANDEL_ROW_X >= w:
            return False
        icon = row[:, MANDEL_ROW_X:min(w, MANDEL_ROW_X+16), :]
        if icon.size == 0:
            return False
        o = cv2.inRange(cv2.cvtColor(icon, cv2.COLOR_BGR2HSV), (10,120,120), (25,255,255))
        return (o > 0).sum() > 5

    # ================================================================
    # Summary
    # ================================================================

    def _build_summary(self, frame_data: list, events: list) -> dict:
        if not frame_data:
            return {}
        mandel_frames = sum(1 for fd in frame_data if fd.get("mandel_present"))
        ranking_ev = [e for e in events if e["type"] == "ranking_change"]
        player_ev = [e for e in events if e["type"] == "player_status"]
        return {
            "total_frames": len(frame_data),
            "color_name_map": self._color_names,
            "mandel_brick_present": mandel_frames > 0,
            "mandel_frames": mandel_frames,
            "ranking_changes": len(ranking_ev),
            "player_status_events": len(player_ev),
            "total_events": len(events),
        }


def analyze_all_views(output_dir: str = "output",
                      teams_config: str = None) -> dict:
    analyzer = GameAnalyzer(teams_config=teams_config)
    results = {}
    base = Path(output_dir)
    for map_dir in sorted(base.iterdir()):
        if not map_dir.is_dir() or map_dir.name.startswith("."):
            continue
        for game_dir in sorted(map_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            for view_dir in sorted(game_dir.iterdir()):
                if not view_dir.is_dir():
                    continue
                view_path = str(view_dir)
                logger.info(f"Analyzing {view_path}")
                result = analyzer.analyze_view(view_path)
                out_path = view_dir / "game_analysis.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                key = f"{map_dir.name}/{game_dir.name}/{view_dir.name}"
                results[key] = result
    return results
