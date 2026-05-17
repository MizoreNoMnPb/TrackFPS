# TrackFPS — FPS Game Minimap Player Trajectory Extraction

[中文文档](doc/README_CN.md)

Extracts player trajectories from FPS game minimap footage (1080p60fps). Built on EasyOCR + OpenCV-Python. No deep learning methods due to VRAM and time constraints.

> **Task results**: Example output for Brakkesh_Game2 View 2 is available in [`/TaskResult`](TaskResult/).
> For best detection results, use the high-bitrate video source from Bilibili: [BaiduPan](https://pan.baidu.com/s/19zYqO1Wo7dG0KjOlXZJzag?pwd=TOTK) passcode: TOTK.

## Notes

This project uses map assets from the game wiki and icons captured from the video for detection assistance. All resources can be found under `./assets/`.

## Pipeline

```
Video → Viewport Detection → Frame Extraction → Map Dot Tracking → Event Detection → Output
```

### 1. Viewport Detection (`map_extractor.py`)

Scans the video for segments where the minimap overlay is visible, then crops the game view and map view separately. Results are saved to `output/{MapName}/Game{N}/View{M}/`.

### 2. Map Dot Tracking (`dot_tracker.py`)

Extracts trajectory and heading information from the map view. Method:

Median background subtraction (using the map region corresponding to the current viewport) + HoughCircles circle detection + color filtering. Player dots on the minimap are ~14×14 pixel circles. Player identification uses OCR + fuzzy matching on the 98×20px name label positioned 20px above each dot's center. Greedy nearest-neighbor matching across frames ensures smooth trajectories and filters false positives.

Output in `output/{MapName}/Game{N}/View{M}/trajectory/`: speed heatmaps, trajectory maps (with turn and stop annotations), and per-player stats in `stats_{Team}.json`.

### 3. Game UI Analysis (`game_analyzer.py`)

Processes the game view UI and the top-right kill feed to extract match events.

Game UI: reads the left-side team table (6 rows × 3 players). Team color bars use hue histogram peak detection. Player status uses 7×9 template matching (alive/knocked/defeated/eliminated). Center timer OCR (Tesseract PSM 11) provides global game clock alignment.

Kill feed: detects top-right kill notifications at (546,44) 447×18px. Frame differencing near status change frames + EasyOCR with edit-distance fuzzy matching against known team and player names.

## Key Challenges

- **Map noise**: Roads and labels resemble player dots. Median background subtraction + foreground area filtering.
- **Color classification**: Semi-transparent team color bars suffer from game background bleed. 15-frame temporal smoothing + majority voting + RANSAC outlier rejection.
- **Player identification**: 98×20px name label OCR. Fuzzy edit-distance matching + elimination fallback for unidentified tracks.

## Unsolved Issues

- **Visual-only matching has inherent limits**. For example, dot tracking works well only for the white team (INK) in the given segment. When many players cluster in one area or the map viewport is too large, trajectories become unreliable.
- **Video resolution limits template matching**. At 1080p split-screen, player status icons in the UI are only 7×9px, and kill feed text is only 18px tall — extremely difficult for OCR or template matching. The current approach uses UI + kill feed dual validation to compensate.
- **Semi-transparent UI causes information loss**. Many UI elements are semi-transparent and overlap directly (e.g., name labels above map dots). When multiple teams fight at close range, overlapping labels make player identification impossible. I chose to provide correct team data via `config/teams.json`; an OCR-based approach is available as an alternative.

## Environment

Requires Python 3.12, conda, and tesseract OCR.

**Linux (Ubuntu/Debian):**
```bash
# System package for OCR
sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# Python environment
conda create -n trackfps python=3.12 -y
conda activate trackfps
pip install -r requirements.txt
```

**Windows:**
```bash
# 1. Install tesseract from https://github.com/UB-Mannheim/tesseract/wiki
#    (check "Chinese Simplified" during installation)
# 2. Add to PATH or set in code:
#    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Python environment
conda create -n trackfps python=3.12 -y
conda activate trackfps
pip install -r requirements.txt
```

## Usage

```bash
conda activate trackfps

# First run
python main.py

# Subsequent runs
python main.py --extract-only                     # extraction only
python main.py --track-only                       # skip extraction, analyze + track
python main.py -c config/custom.yaml              # custom config
```

## Config Reference (`config/default.yaml`)

```yaml
input:
  video_path: "input/Final/Brakkesh_Game2.mp4"

map_dir: "assets/map"                  # reference map directory
teams_config: "config/teams.json"      # team color, name, player list

output:
  dir: "output"

pipeline:
  skip_extraction: false               # true if frames already extracted
  analyze_ui: true                     # generate events.csv
  track_teams: "INK"                   # "all", "INK", or ["INK","ESG"]

scanning:
  scan_step: 300                       # check every N frames for map view
  min_segment_frames: 120              # minimum frames per valid segment

trajectory:
  turn_threshold: 90                   # degrees for marking turns
  stop_speed: 2                        # px/s threshold for stopped
  stop_min_duration: 0.5               # seconds minimum for stop annotation
```

## Output Structure

```
output/{MapName}/Game{N}/View{M}/
├── map/                    # Map crops
├── game/                   # Game view crops
├── map_region.png          # Reference map region
├── metadata.json           # Viewport coords, homography, frame range
├── game_analysis.json      # Per-frame UI state + event timeline
├── events.csv              # Human-readable event table
└── trajectory/
    ├── track_{Team}_{Player}.jpg     # Individual trajectory map
    ├── heatmap_{Team}_{Player}.jpg   # Speed heatmap
    └── stats_{Team}.json             # Speed, turns, distance stats
```

## Events CSV Format

`events.csv` columns:

| Column | Description |
|--------|-------------|
| `frame` | Extracted frame number |
| `video_time` | Video timestamp (M:SS) |
| `game_time` | In-game clock countdown (M:SS) |
| `type` | `player_status` or `ranking_change` |

**player_status rows:**

| Column | Description |
|--------|-------------|
| `team` | Team name (color) |
| `player` | Player name from teams.json |
| `from_status` | Previous state: alive / knocked / defeated / eliminated |
| `to_status` | New state |

**ranking_change rows:**

| Column | Description |
|--------|-------------|
| `team` | Team moving to new rank position |
| `player` | Rank #N (1-indexed) |
| `from_status` | Previous team at this rank |
| `to_status` | New team at this rank |

Valid player state transitions: `alive→knocked`, `knocked→alive`, `knocked→defeated`, `defeated→alive`, `*→eliminated`.
