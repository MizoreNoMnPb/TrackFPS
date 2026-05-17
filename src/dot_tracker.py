"""Track player dots: median background subtraction + nearest-neighbor matching.

Team color ranges (HSV):
  white:  (0, 0, 180) - (180, 40, 255)
  yellow: (25, 80, 100) - (35, 255, 255)
  green:  (40, 80, 60) - (80, 255, 255)
  red:    (0, 100, 60) - (10, 255, 255) | (160, 100, 60) - (180, 255, 255)
  blue:   (100, 100, 60) - (130, 255, 255)
  purple: (130, 80, 60) - (160, 255, 255)
"""

TEAM_COLORS_HSV = {
    "white":  ((0, 0, 180), (180, 40, 255)),
    "yellow": ((25, 80, 100), (35, 255, 255)),
    "green":  ((40, 80, 60), (80, 255, 255)),
    "red":    ((0, 100, 60), (10, 255, 255)),
    "red2":   ((160, 100, 60), (180, 255, 255)),
    "blue":   ((100, 100, 60), (130, 255, 255)),
    "purple": ((130, 80, 60), (160, 255, 255)),
}

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
        if fg_area < 100:  # real ~14x14 dot has 120-180 fg pixels
            continue
        dots.append((int(x), int(y)))

    # Deduplicate nearby
    unique = []
    for x, y in dots:
        if not any(abs(x - ux) < 10 and abs(y - uy) < 10 for ux, uy in unique):
            unique.append((x, y))
    return unique[:5]


def track_dots(map_dir: Path, color_lower: tuple, color_upper: tuple) -> dict:
    """Per-frame detection + greedy nearest-neighbor matching for smooth tracks."""
    paths = sorted(map_dir.glob("frame_*.jpg"))
    bg = build_background(map_dir)

    active = {}   # track_id -> list of (x, y, fn)
    lasts = {}    # track_id -> (x, y)
    next_id = 0

    for fi, fp in enumerate(paths):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        fn = int(fp.stem.replace("frame_", ""))
        dots = detect_dots(frame, bg, color_lower, color_upper)
        used = set()

        # Match existing tracks to nearest dot (max 30px, longer tracks first)
        for tid in sorted(active.keys(), key=lambda t: -len(active[t])):
            if tid not in lasts:
                continue
            px, py = lasts[tid]
            best_d, best_j = float('inf'), -1
            for j, (dx, dy) in enumerate(dots):
                if j in used:
                    continue
                d = np.sqrt((dx-px)**2 + (dy-py)**2)
                if d < best_d and d < 30:
                    best_d, best_j = d, j
            if best_j >= 0:
                used.add(best_j)
                active[tid].append((dots[best_j][0], dots[best_j][1], fn))
                lasts[tid] = (dots[best_j][0], dots[best_j][1])

        # Create new tracks for unmatched dots
        for j, (dx, dy) in enumerate(dots):
            if j not in used:
                active[next_id] = [(dx, dy, fn)]
                lasts[next_id] = (dx, dy)
                next_id += 1

    trajectories = {}
    for tid in list(active.keys()):
        trajectories[tid] = [(x, y, fn) for x, y, fn in active[tid]]

    # Filter noise tracks first, then identify players
    filtered = {}
    for tid, pts in trajectories.items():
        if len(pts) < 100:
            continue
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        if np.sqrt(dx*dx + dy*dy) > 5:
            filtered[tid] = pts
    trajectories = filtered

    # Attempt OCR-based player identification
    player_ids = _identify_players(map_dir, trajectories)

    result = {}
    for tid, pts in trajectories.items():
        pid = player_ids.get(tid)
        if pid and pid in result:
            pid = f"{pid}_{tid}"
        if pid is None:
            pid = f"P{tid}"
        result[pid] = pts
    return result


def _fuzzy_match(text, target):
    """Check if target appears in text with <=1 char error, first char must match."""
    if target in text:
        return True
    for i in range(len(text) - len(target) + 1):
        window = text[i:i+len(target)]
        if window[0] != target[0]:
            continue  # first char must match
        err = sum(1 for a, b in zip(window, target) if a != b)
        if err <= 1:
            return True
    return False


def _identify_players(map_dir, trajectories):
    """OCR player names from labels at mid-view frames to identify each track."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except ImportError:
        return {}

    KNOWN_PLAYERS = {"BabyB", "Kilzal", "tooEzze"}
    player_ids = {}

    for tid, pts in trajectories.items():
        for frac in [0.5, 0.25, 0.75]:
            idx = int(len(pts) * frac)
            mx, my, mfn = pts[idx]
            fn_str = f"frame_{int(mfn):06d}.jpg"
            fp = map_dir / fn_str
            if not fp.exists():
                continue
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            lx1, lx2 = int(mx) - 49, int(mx) + 49
            ly1, ly2 = int(my) - 20 - 20, int(my) - 20
            if lx1 < 0 or ly1 < 0 or lx2 >= frame.shape[1] or ly2 >= frame.shape[0]:
                continue
            label = frame[ly1:ly2, lx1:lx2]
            if label.shape[0] < 15 or label.shape[1] < 50:
                continue
            results = reader.readtext(label, detail=0)
            found = False
            for text in results:
                clean = ''.join(c.lower() for c in text if c.isalnum())
                for name in KNOWN_PLAYERS:
                    if _fuzzy_match(clean, name.lower()):
                        player_ids[tid] = name
                        found = True
                        break
                if found:
                    break
            if found:
                logger.info(f"  Track {tid} → {player_ids[tid]}")
                break
    # Elimination: if 2 of 3 identified, the 3rd is the remaining player
    identified = set(player_ids.values())
    missing = KNOWN_PLAYERS - identified
    if len(missing) == 1 and len(player_ids) == 2:
        for tid in trajectories:
            if tid not in player_ids:
                player_ids[tid] = missing.pop()
                logger.info(f"  Track {tid} → {player_ids[tid]} (elimination)")
                break

    return player_ids


def track_team(map_dir: Path, team_color: str, team_players: list[str],
               team_name: str = "", map_region: str = None) -> dict:
    """Track one team's dots and identify players.

    Args:
        map_dir: path to map frame directory
        team_color: 'white', 'yellow', 'green', 'red', 'blue', or 'purple'
        team_players: list of 3 player names (e.g. ['BabyB','Kilzal','tooEzze'])
        map_region: path to map_region.png for drawing

    Returns dict with player-named trajectories.
    """
    color_key = team_color if team_color != "red" else "red"
    lo, hi = TEAM_COLORS_HSV[color_key]

    traj = track_dots(map_dir, lo, hi)

    # For red, also merge red2 range results
    if team_color == "red":
        lo2, hi2 = TEAM_COLORS_HSV["red2"]
        traj2 = track_dots(map_dir, lo2, hi2)
        # Merge trajectories (prefer longer ones)
        for tid, pts in traj2.items():
            if tid not in traj or len(pts) > len(traj[tid]):
                traj[tid] = pts

    # Keep only top 3 longest tracks (filter false positives)
    sorted_tracks = sorted(traj.items(), key=lambda x: -len(x[1]))
    traj = dict(sorted_tracks[:3])

    # OCR identification with team-specific names
    player_ids = _identify_players_for_team(map_dir, traj, set(team_players))
    named = {}
    for tid, pts in traj.items():
        pid = player_ids.get(tid)
        if pid and pid in named:
            pid = f"{pid}_{tid}"
        if pid is None:
            pid = f"{team_color}_{tid}"
        named[pid] = pts

    # Draw individual trajectories
    if map_region:
        bgg = cv2.imread(map_region)
        if bgg is not None:
            bgg = cv2.resize(bgg, (854, 852))
            colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]
            out_dir = Path(map_dir).parent / "trajectory"
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, (pid, pts) in enumerate(named.items()):
                c = colors[i % 3]
                cvs = bgg.copy()
                for j in range(1, len(pts)):
                    cv2.line(cvs, (int(pts[j-1][0]), int(pts[j-1][1])),
                             (int(pts[j][0]), int(pts[j][1])), c, 2)
                for x, y, _ in pts[::20]:
                    cv2.circle(cvs, (int(x), int(y)), 7, c, 1)
                cv2.imwrite(str(out_dir / f"track_{team_name}_{pid}.jpg"), cvs)

    return named


def _identify_players_for_team(map_dir, trajectories, player_names):
    """OCR player names from labels, using team-specific known names."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except ImportError:
        return {}

    player_ids = {}
    for tid, pts in trajectories.items():
        for frac in [0.5, 0.25, 0.75]:
            idx = int(len(pts) * frac)
            mx, my, mfn = pts[idx]
            fp = map_dir / f"frame_{int(mfn):06d}.jpg"
            if not fp.exists():
                continue
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            lx1, lx2 = int(mx) - 49, int(mx) + 49
            ly1, ly2 = int(my) - 20 - 20, int(my) - 20
            if lx1 < 0 or ly1 < 0 or lx2 >= frame.shape[1] or ly2 >= frame.shape[0]:
                continue
            label = frame[ly1:ly2, lx1:lx2]
            if label.shape[0] < 15 or label.shape[1] < 50:
                continue
            results = reader.readtext(label, detail=0)
            found = False
            for text in results:
                clean = ''.join(c.lower() for c in text if c.isalnum())
                for name in player_names:
                    if _fuzzy_match(clean, name.lower()):
                        player_ids[tid] = name
                        found = True
                        break
                if found:
                    break
            if found:
                break

    # Elimination fallback
    identified = set(player_ids.values())
    missing = player_names - identified
    if len(missing) == 1 and len(player_ids) == len(player_names) - 1:
        for tid in trajectories:
            if tid not in player_ids:
                player_ids[tid] = missing.pop()
                break

    return player_ids


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
