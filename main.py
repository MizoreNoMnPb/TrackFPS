"""TrackFPS — FPS Game Minimap Player Trajectory Extraction.

Usage:
    python main.py                                    # default config
    python main.py --config config/custom.yaml         # custom config
    python main.py --video input/Final/Brakkesh_Game2.mp4

Per-view analysis is run via individual module scripts (see README).
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.map_extractor import run as extract
from src.map_matcher import MapMatcher


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="TrackFPS")
    parser.add_argument("--config", "-c", default="config/default.yaml")
    parser.add_argument("--video", "-v", default=None)
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract frames, skip analysis")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    video_path = args.video or config.get("input", {}).get("video_path")
    if not video_path or not Path(video_path).exists():
        logging.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Parse map name from filename: {MapName}_Game{N}.mp4
    stem = Path(video_path).stem
    parts = stem.split("_")
    map_name = "_".join(parts[:-1]) if parts[-1].startswith("Game") else stem

    ref_map = Path(config.get("map_dir", "assets/map")) / f"{map_name}_unlabeled.png"
    if not ref_map.exists():
        logging.error(f"Reference map not found: {ref_map}")
        sys.exit(1)

    # Extract frames
    output_dir = config.get("output", {}).get("dir", "output")
    extract(video_path, str(ref_map.parent), output_dir,
            scan_step=config.get("scanning", {}).get("scan_step", 300),
            min_segment_frames=config.get("scanning", {}).get("min_segment_frames", 120))

    logging.info("Extraction complete. Run game_analyzer and dot_tracker for analysis.")


if __name__ == "__main__":
    main()
