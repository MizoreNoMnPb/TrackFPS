"""TrackFPS - FPS Game Player Trajectory Extraction & Behavior Analysis.

Usage:
    python main.py                                    # default config
    python main.py --config config/custom.yaml         # custom config
    python main.py --video input/Brakkesh_Game2.mp4    # specify video directly
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.map_identifier import parse_filename, get_reference_map_path
from src.viewport_detector import ViewportDetector
from src.map_matcher import MapMatcher
from src.pipeline import ViewProcessor


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="TrackFPS - FPS Player Trajectory Extraction"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        help="Override video path from config",
    )
    args = parser.parse_args()

    setup_logging()

    # Load config
    if not Path(args.config).exists():
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Determine video path
    video_path = args.video or config.get("input", {}).get("video_path")
    if not video_path:
        logging.error("No video path specified. Use --video or set input.video_path in config.")
        sys.exit(1)

    if not Path(video_path).exists():
        logging.error(f"Video file not found: {video_path}")
        sys.exit(1)

    # ---- Step 1: Identify map from filename ----
    info = parse_filename(video_path)
    map_name = info["map_name"]
    game_num = info["game_num"]
    logging.info(f"Map: {map_name}, Game: {game_num}")

    # ---- Step 2: Find reference map ----
    ref_map_path = get_reference_map_path(map_name, config.get("map_dir", "map"))
    if ref_map_path is None:
        logging.error(f"Reference map not found for {map_name}. "
                      f"Place it at map/{map_name}_unlabel.png")
        sys.exit(1)

    map_matcher = MapMatcher(ref_map_path)

    # ---- Step 3: Scan video for map viewport segments ----
    detector = ViewportDetector(
        scan_step=config.get("scanning", {}).get("scan_step", 30),
        min_segment_frames=config.get("scanning", {}).get("min_segment_frames", 60),
        edge_density_threshold=config.get("scanning", {}).get("edge_density_threshold", 10.0),
        border_grad_threshold=config.get("scanning", {}).get("border_grad_threshold", 30.0),
    )
    segments = detector.scan(video_path)

    if not segments:
        logging.warning("No map view segments found in video!")
        sys.exit(0)

    # ---- Step 4: Process each view segment ----
    processor = ViewProcessor(config)
    all_results = []

    for seg in segments:
        # Refine segment boundaries
        seg = detector.refine_segment(video_path, seg)

        result = processor.process(video_path, map_name, game_num, seg, map_matcher)
        all_results.append(result)

    # ---- Step 5: Summary ----
    total_players = sum(len(r["tracks"]) for r in all_results)
    total_events = sum(len(r["events"]) for r in all_results)
    logging.info("=" * 50)
    logging.info("Pipeline Complete!")
    logging.info(f"  Views processed: {len(all_results)}")
    logging.info(f"  Total players tracked: {total_players}")
    logging.info(f"  Total events detected: {total_events}")
    logging.info(f"  Output: {config['output']['dir']}/{map_name}/Game{game_num}/")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
