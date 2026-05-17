"""Kill feed detection triggered by player status events. Post-processes game_analysis.json."""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
import re

logger = logging.getLogger(__name__)

KF_X, KF_Y = 546, 44
KF_W = 447
LINE_H = 18
LINE_GAP = 2
KF_H = LINE_H * 3 + LINE_GAP * 2  # detect up to 3 lines for OCR
KF_H_SAVE = LINE_H                 # save only 1st line for report

KNOWN_NAMES = set()
TEAM_COLORS = {}
PLAYER_TEAMS = {}
_knock_icons = {}  # loaded templates


def load_teams(config_path: str = "config/teams.json"):
    global KNOWN_NAMES, TEAM_COLORS, PLAYER_TEAMS, _knock_icons
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    for t in data["teams"]:
        TEAM_COLORS[t["name"]] = t["color"]
        KNOWN_NAMES.add(t["name"])
        for p in t["players"]:
            KNOWN_NAMES.add(p)
            PLAYER_TEAMS[p] = t["name"]

    # Load knock icons for template matching
    for name in ["knock", "headshot_knock"]:
        path = f"assets/knock_icon/{name}.png"
        icon = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if icon is not None:
            _knock_icons[name] = icon
    logger.info(f"Loaded {len(_knock_icons)} knock icon templates")


def _has_knock_icon(roi_gray: np.ndarray) -> bool:
    """Check if a knock icon is present in the kill feed ROI."""
    for name, tmpl in _knock_icons.items():
        result = cv2.matchTemplate(roi_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > 0.50:
            return True
    return False


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def fuzzy_match(token: str) -> str | None:
    t = token.lower()
    best, best_d = None, 99
    for name in KNOWN_NAMES:
        d = _levenshtein(t, name.lower())
        limit = min(3, max(1, len(name) // 3))
        if d <= limit and d < best_d:
            best_d, best = d, name
    return best


def _ocr_killfeed(roi: np.ndarray) -> list[str]:
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return reader.readtext(roi, detail=0)
    except ImportError:
        pass
    try:
        import pytesseract
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if b.mean() > 127: b = 255 - b
        s = cv2.resize(b, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        t = pytesseract.image_to_string(s, lang='eng', config='--psm 6').strip()
        return t.split('\n') if t else []
    except ImportError:
        return []


def parse_event(lines: list[str], roi: np.ndarray = None) -> dict | None:
    """Parse OCR output + knock icon detection into kill event dict."""
    if not lines:
        return None
    full = ' '.join(lines)
    tokens = re.findall(r'[A-Za-z0-9]{2,}', full)
    matched = []
    for tok in tokens:
        m = fuzzy_match(tok)
        if m: matched.append(m)

    if len(matched) < 2:
        return None

    found_teams, found_players = [], []
    for m in matched:
        if m in TEAM_COLORS:
            found_teams.append(m)
        if m in PLAYER_TEAMS:
            found_players.append(m)

    if len(found_players) < 2:
        return None

    p1, p2 = found_players[0], found_players[1]
    t1 = PLAYER_TEAMS.get(p1, found_teams[0] if found_teams else '?')
    t2 = PLAYER_TEAMS.get(p2, found_teams[1] if len(found_teams) > 1 else '?')

    # Determine event type
    same_player = (p1 == p2)
    same_team = (t1 == t2)

    if same_player:
        etype = "invalid"
    elif same_team:
        etype = "rescue"
    elif "wipeout" in full.lower():
        etype = "wipeout"
    elif roi is not None and _has_knock_icon(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)):
        etype = "knock"
    else:
        etype = "kill"

    return {
        "type": etype,
        "team1": t1, "player1": p1,
        "team2": t2, "player2": p2,
        "raw_ocr": full,
    }


def scan_view(view_dir: str) -> list[dict]:
    """Post-process a view to find kill feed events at player status change frames."""
    analysis_path = Path(view_dir) / "game_analysis.json"
    if not analysis_path.exists():
        logger.warning(f"No game_analysis.json in {view_dir}")
        return []

    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)

    game_dir = Path(view_dir) / "game"
    paths = sorted(game_dir.glob("frame_*.jpg"))
    if not paths:
        return []

    # Build frame name -> index map
    name_to_idx = {p.stem: i for i, p in enumerate(paths)}

    player_events = [e for e in data.get("events", []) if e["type"] == "player_status"]
    if not player_events:
        logger.info(f"No player events in {view_dir}")
        return []

    logger.info(f"Scanning {len(player_events)} player events for kill feed in {view_dir}")

    kill_events = []
    VIEW_START_SKIP = 30  # ignore kill feed in first 30 frames (may be stale)

    for pe in player_events:
        ev_idx = pe.get("frame_idx")
        if ev_idx is not None and ev_idx < VIEW_START_SKIP:
            continue  # skip stale kill feed at view start
        if ev_idx is None:
            fn = pe["frame"]
            ev_idx = name_to_idx.get(fn, -1)
            if ev_idx < 0:
                continue

        # Check ±3 frames around event
        found = False
        for offset in range(-3, 4):
            check_idx = ev_idx + offset
            if check_idx < 0 or check_idx >= len(paths):
                continue

            frame = cv2.imread(str(paths[check_idx]))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if w < KF_X + KF_W or h < KF_Y + KF_H:
                continue

            # Frame diff with previous to detect sudden kill feed appearance
            if check_idx < 1:
                continue
            prev_frame = cv2.imread(str(paths[check_idx - 1]))
            if prev_frame is None:
                continue

            curr_roi = frame[KF_Y:KF_Y + KF_H, KF_X:KF_X + KF_W]
            prev_roi = prev_frame[KF_Y:KF_Y + KF_H, KF_X:KF_X + KF_W]
            diff = cv2.absdiff(curr_roi, prev_roi)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            diff_mean = diff_gray.mean()
            high_change = (diff_gray > 50).sum()

            # Kill feed signature: sudden high change at this ROI
            if diff_mean > 8 and high_change > 100:
                ocr_lines = _ocr_killfeed(curr_roi)
                parsed = parse_event(ocr_lines, curr_roi)

                ke = {
                    "frame": paths[check_idx].stem,
                    "frame_idx": check_idx,
                    "trigger_event_frame": pe["frame"],
                    "diff_mean": round(float(diff_mean), 1),
                    "high_change_px": int(high_change),
                    "ocr_raw": ocr_lines,
                    "event": parsed,
                }
                kill_events.append(ke)

                if parsed:
                    logger.info(f"  {paths[check_idx].stem}: {parsed['type']} "
                                f"{parsed['player1']}({parsed['team1']}) "
                                f"-> {parsed['player2']}({parsed['team2']})")
                found = True
                break

    logger.info(f"  Found {len(kill_events)} kill feed events")
    return kill_events


def scan_all_views(output_dir: str = "output",
                   teams_config: str = "config/teams.json") -> dict:
    load_teams(teams_config)
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
                vp = str(view_dir)
                events = scan_view(vp)
                with open(view_dir / "killfeed_events.json", "w", encoding="utf-8") as f:
                    json.dump(events, f, indent=2, ensure_ascii=False, default=str)
                results[f"{map_dir.name}/{game_dir.name}/{view_dir.name}"] = events
    return results
