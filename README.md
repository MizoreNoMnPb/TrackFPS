# TrackFPS — FPS Game Minimap Player Trajectory Extraction

[中文文档](doc/README_CN.md)

Extracts player trajectories from FPS game minimap footage (1080p60fps). Built on EasyOCR + OpenCV-Python. No deep learning due to VRAM/time constraints.

> For best results, use the high-bitrate video source from Bilibili: [BaiduPan](https://pan.baidu.com/s/19zYqO1Wo7dG0KjOlXZJzag?pwd=TOTK) code: TOTK.
> View 2 of Brakkesh_Game2 corresponds to the segment referenced in the original task document.

## Setup

Requires conda and tesseract:

```bash
# System packages
sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# Python environment
conda create -n trackfps python=3.12 -y
conda activate trackfps
pip install -r requirements.txt
```

Place video files in `input/Final/{MapName}_Game{N}.mp4` (e.g. `input/Final/Brakkesh_Game2.mp4`).

## Usage

```bash
conda activate trackfps

# First run — extract frames + analyze + track
python main.py

# Subsequent runs — skip extraction
python main.py --track-only

# Custom config or video
python main.py -c config/custom.yaml -v input/Final/Dum_Game1.mp4

# Extract only (no analysis)
python main.py --extract-only
```

## Pipeline

```
Video → Viewport Detection → Frame Extraction → Dot Tracking + UI Analysis → Output
```

### 1. Viewport Detection (`map_extractor.py`)

Scans the video for minimap overlay segments (right side, map border at x=1008). Crops map (854×852) and game view (998×559). ORB feature matching + homography aligns the viewport with the reference map. Skips frames without visible timer (loading/inventory screens).

### 2. Map Dot Tracking (`dot_tracker.py`)

Median background subtraction (static map) + HoughCircles + color filter. Dots are ~14×14 circles. Greedy nearest-neighbor matching (max 30px/frame) for smooth trajectories. Player identification via OCR of 98×20px name labels above each dot (EasyOCR + fuzzy Levenshtein matching).

Output in `output/{MapName}/Game{N}/View{M}/trajectory/`: individual trajectory maps (with turn/stop annotations), speed heatmaps, and `stats_{Team}.json` (speed, distance, turns).

### 3. Game UI Analysis (`game_analyzer.py`)

Reads the left-side team table (6 rows × 3 players) using hue histogram peak detection for team colors and 7×9 template matching for player status (alive/knocked/defeated/eliminated). Center timer OCR (Tesseract PSM 11) provides global game clock alignment via RANSAC-fit across all views.

Kill feed detection at (546,44) 447×18px — frame differencing near status change frames, EasyOCR + fuzzy matching against known team/player names.

## Key Challenges

- **Map noise**: Roads/labels mimic player dots. Median background subtraction + foreground area filter (>100px).
- **Color classification**: Semi-transparent team color bars bleed game background. 15-frame temporal smoothing + majority voting + RANSAC outlier rejection.
- **Player identification**: 98×20px name label OCR. Fuzzy edit-distance matching + elimination fallback.
- **UI scale**: Icons are only 7×9px and kill feed text is 18px tall at 1080p split-screen. Dual-validation (UI table + kill feed) compensates for OCR/template limitations.
- **UI transparency**: Team-colored overlays merge when players cluster. Config files with known team data bypass OCR where needed.

## Config

`config/teams.json` — team colors, names, and player lists (3/team × 6 teams).

`config/default.yaml`:

```yaml
input:
  video_path: "input/Final/Brakkesh_Game2.mp4"
map_dir: "assets/map"                  # reference maps
teams_config: "config/teams.json"      # team data
output:
  dir: "output"
pipeline:
  skip_extraction: false               # true if frames already extracted
  analyze_ui: true                     # generate events.csv
  track_teams: "INK"                   # "all", "INK", or ["INK","ESG"]
scanning:
  scan_step: 300                       # check every N frames
  min_segment_frames: 120              # min frames per view segment
trajectory:
  turn_threshold: 90                   # degrees for turn markers
  stop_speed: 2                        # px/s below this = stopped
  stop_min_duration: 0.5               # seconds minimum stop
```

## Output Structure

```
output/{MapName}/Game{N}/View{M}/
├── map/                    # Map crops (854×852)
├── game/                   # Game view crops (998×559)
├── map_region.png          # Reference map crop
├── metadata.json           # Viewport coords, homography, frame range
├── game_analysis.json      # Per-frame UI state + event timeline
├── events.csv              # Human-readable event table
└── trajectory/
    ├── track_{Team}_{Player}.jpg     # Individual trajectory
    ├── heatmap_{Team}_{Player}.jpg   # Speed heatmap
    └── stats_{Team}.json             # Speed, turns, distance
```
