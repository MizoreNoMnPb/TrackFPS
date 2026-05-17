# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrackFPS extracts and analyzes player trajectories from FPS game minimap footage. The pipeline: scan video for map-view segments → crop game/map viewports → match map to reference → detect player dots → track trajectories → detect game events.

**Input**: 1080p60fps video (`input/Final/{MapName}_Game{N}.mp4`)
**Output**: `output/{MapName}/Game{N}/View{M}/` with `map/`, `game/`, `map_region.png`, `metadata.json`, `game_analysis.json`

## Running

```bash
conda activate trackfps
# Extract map segments + frames from video:
python -c "from src.map_extractor import run; run('input/Final/Brakkesh_Game2.mp4', scan_step=300, min_segment_frames=120)"

# Analyze game UI (team table, player status, timer):
python -c "from src.game_analyzer import GameAnalyzer; import json; a=GameAnalyzer(teams_config='config/teams.json'); r=a.analyze_view('output/Brakkesh/Game2/View0'); json.dump(r, open('output/Brakkesh/Game2/View0/game_analysis.json','w'), indent=2, ensure_ascii=False, default=str)"

# Track map dots (INK team):
python -c "from src.dot_tracker import track_dots, draw_trajectories; from pathlib import Path; traj=track_dots(Path('output/Brakkesh/Game2/View2/map'), (0,0,180), (180,40,255)); draw_trajectories('output/Brakkesh/Game2/View2/map_region.png', traj, 'output/temp/ink.jpg')"

# Kill report scanning:
python -c "from src.killreport import load_teams, scan_view; load_teams('config/teams.json'); events=scan_view('output/Brakkesh/Game2/View2')"
```

## Key UI Layout (game-view pixel coords, 998×559)

All coords relative to game view frame (cropped from full frame using `map_extractor.VIEWPORT`).

```
Timer bracket:     (454, 8) 92×36       Green `[]` enclosing Match{N} + countdown
Team table:        (8, 208) 164×168     6 rows, 18px header, 23px data rows, 2px gap
  - Color bar:     x=16, 4px wide       Within each row
  - Team name:     x=25, 30px wide      OCR region
  - Player icons:  x=[93, 102, 110]     7×9px each, 3 per row
  - Mandel icon:   x=174                
Kill feed:         (546, 44) 447×18      Right-aligned, 18px/line, 2px line gap, semi-transparent black bg
Mandel Brick:      (551, 9) 33×33        Orange icon right of timer
```

## Known Teams & Players (`config/teams.json`)

| Color | Name | Players |
|-------|------|---------|
| white | INK | BabyB, Kilzal, tooEzze |
| yellow | ESG | FUJIANG, DT, LIUYUN |
| green | Q9 | K1De, GuoR, Mo |
| red | MG | eNvyx, Onigiri, Bonnieeeee |
| blue | FN | Flarich, nhrir, usbishka |
| purple | MBA | skiteyyy, Bulldo, yuvel11r |

3 players per team. Dots on map are 14×14 colored circles. Labels 20px above dot center (98×20 rect, semi-transparent team color, `TeamName | PlayerName`).

## Viewport Boundaries (full-frame coords)

Set in `map_extractor.VIEWPORT`:
- Map: (1008, 112, 854, 852)
- Game: (40, 258, 998, 559)

## Current Module State

### `src/map_extractor.py` — Frame extraction (DONE)
- Scans video for map viewport segments (detects map left border at x=1008)
- Binary search refines start/end boundaries
- Timer check during extraction skips loading/inventory frames (green bracket must be visible)
- Extracts cropped map + game frames + matches to reference map (ORB + homography)
- Output: `output/{Map}/{Game}/{View}/` with `map/`, `game/`, `map_region.png`, `metadata.json`

### `src/game_analyzer.py` — Game UI analysis (DONE, events unreliable)
- Reads team table colors (hue histogram peak, temporal smoothing)
- OCR team names (EasyOCR, English uppercase+digits only)
- Player status via template matching (`assets/player_state/*.png`, 7×8 → 7×9)
- Timer detection + warmup skipping (skips frames without green timer bracket)
- **Known issue**: Player status events from table are noisy (detection oscillates). Kill feed events are more reliable.
- Use `teams_config='config/teams.json'` to skip OCR and use known names

### `src/dot_tracker.py` — Map dot tracking (IN PROGRESS)
- `detect_dots()`: HoughCircles on color mask + 14×14 patch verification + dedup
- `DotTracker`: greedy nearest-neighbor matching, linear extrapolation prediction, max_gap/max_dist params
- `track_dots()`: full pipeline, filters tracks by length (>50) and displacement (>10px)
- **Working for INK (white)**: 3 tracks, high coverage on View 2
- **Not yet extended to other 5 teams** — just change color_lower/color_upper params
- `draw_trajectories()`: renders on map_region.png background

### `src/killreport.py` — Kill feed OCR (DONE, OCR quality poor)
- Detects kill feed by frame differencing at ROI (546, 44) 447×58
- OCR via EasyOCR + fuzzy name matching (Levenshtein)
- Knock/kill distinction via template matching (`assets/knock_icon/`)
- **Known issue**: High OCR error rate on small text. Team colors could help narrow matches.

### `src/map_matcher.py` — Map matching (DONE)
- ORB feature matching viewport → reference map (`assets/map/{MapName}_unlabeled.png`)
- Returns homography matrix for pixel → map coordinate transform

## Gotchas

1. **OpenCV KalmanFilter.predict() bug**: In OpenCV 4.13, `predict()` returns zeros and corrupts statePost. Use manual matrix math instead (see dot_tracker.py's DotTracker class which avoids Kalman entirely).

2. **Map viewport is STATIC** (no scrolling). Median background subtraction works perfectly for dot detection.

3. **Game timer is global countdown** (not per-round). It ticks faster than real time (62-164 real seconds per game-minute).

4. **Kill feed is very transient** — appears for only 1-2 extracted frames. Frame differencing at the ROI is the most reliable detection method.

5. **Video is 59.94fps, frame_skip=2** — each extracted frame represents 2 video frames (~33ms).

6. **Invalid frames**: View starts have loading transitions, mid-view gaps have inventory screens covering UI. Timer bracket detection filters these. Map_extractor now skips them during extraction.

7. **Team table header**: First 18px is a header row, not team data. `DATA_START_Y = TABLE_Y + HEADER_H`.

8. **Watermark**: Row 0 player IDs are occluded by a constant-position video watermark. Manual annotation needed.

9. **Tesseract is installed** (eng+chi_sim) but **EasyOCR gives better results** for small text.

10. **conda environment**: `trackfps`, Python 3.12. Use `/home/wsl_val/miniconda3/envs/trackfps/bin/python` if conda isn't activated.

## Input Directory Structure

```
input/Final/        # Final tournament (6 games)
  Brakkesh_Game2.mp4, Brakkesh_Game3.mp4, Dum_Game1.mp4,
  Space City_Game4.mp4, Space City_Game5.mp4, Space City_Game6.mp4
input/Semi-Final1/  # Semi-final bracket 1
input/Semi-Final2/  # Semi-final bracket 2
```
