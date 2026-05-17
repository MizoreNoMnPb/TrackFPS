# TrackFPS — FPS Game Minimap Player Trajectory Extraction

[中文文档](doc/README_CN.md)

Extracts and analyzes player movement trajectories from FPS game minimap footage (1080p60fps). Tracks 3-player squads across 6 teams, detects game events (knocks, kills, rescues, eliminations), and generates annotated trajectory maps with speed heatmaps.

## Pipeline

```
Video → Map Viewport Detection → Frame Extraction → Dot Tracking → Event Detection → Output
```

1. **Viewport Detection** (`map_extractor.py`): Scans video for minimap overlay (right side of screen). Detects map border at x=1008, crops map (854x852) and game view (998x559) using annotated boundaries. Uses ORB feature matching + homography to align viewport with reference map. Skips frames without visible timer (loading/inventory screens).

2. **Player Dot Tracking** (`dot_tracker.py`): Per-team dot detection using median background subtraction + HoughCircles + color filter. 14-pixel circular dots (BGR=(230,235,234) for white team). Greedy nearest-neighbor matching (max 30px inter-frame displacement) for smooth trajectories. OCR-based player identification from name labels above dots. Outputs individual trajectory maps with speed/heatmap overlays.

3. **Game UI Analysis** (`game_analyzer.py`): Reads team table (6 rows x 3 players) using hue histogram peak detection. Player status via 7x9 template matching (alive/knocked/defeated/eliminated). Timer OCR (Tesseract PSM 11) for game clock alignment across views.

4. **Kill Feed** (`killreport.py`): Detects right-aligned kill notifications at (546,44) 447x18px. EasyOCR + fuzzy name matching against known team/player names.

## Key Challenges

- **Dot detection noise**: Map features (roads, labels) mimic player dots. Fixed with median background subtraction (static map, no scrolling) + area filter (>100px foreground area).
- **Color classification**: Semi-transparent team color bars bleed game background. Solved with temporal smoothing (15-frame window) + majority voting + global timer-based outlier rejection (RANSAC).
- **Player identification**: OCR on 98x20px name labels at 20px above dot center. Fuzzy Levenshtein matching with elimination fallback for unidentified tracks.
- **Timer OCR drift**: PSM 6 misread "19" as "13" causing 6-minute error. Switched to PSM 11 + RANSAC outlier rejection for stable global timer mapping across all views.

## Usage

```bash
conda activate trackfps
python -c "from src.map_extractor import run; run('input/Final/Brakkesh_Game2.mp4')"
python -c "from src.game_analyzer import GameAnalyzer; a=GameAnalyzer(teams_config='config/teams.json'); a.analyze_view('output/.../View0')"
python -c "from src.dot_tracker import track_team; track_team(map_dir, 'white', players, 'INK', map_region='...')"
```

## Output Structure

```
output/{MapName}/Game{N}/View{M}/
├── map/                  # Cropped minimap frames (854x852)
├── game/                 # Cropped game UI frames (998x559)
├── map_region.png        # Reference map crop
├── map_region_labeled.png
├── metadata.json         # Viewport coords, homography, frame range
├── game_analysis.json    # Per-frame UI state + event timeline
├── events.csv            # Human-readable event table
└── trajectory/
    ├── track_{Team}_{Player}.jpg   # Individual trajectory maps
    ├── heatmap_{Team}_{Player}.jpg # Speed heatmaps
    └── stats_{Team}.json           # Speed, turns, distance
```
