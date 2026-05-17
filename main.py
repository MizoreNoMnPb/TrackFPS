"""TrackFPS — Full pipeline: extract, analyze, track."""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.map_extractor import run as extract_frames
from src.game_analyzer import GameAnalyzer
from src.dot_tracker import track_team


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="TrackFPS")
    parser.add_argument("--config", "-c", default="config/default.yaml")
    parser.add_argument("--video", "-v", default=None)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--track-only", action="store_true")
    args = parser.parse_args()

    setup_logging()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Video path
    video_path = args.video or cfg["input"]["video_path"]
    if not Path(video_path).exists():
        logging.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Parse metadata from filename
    stem = Path(video_path).stem
    parts = stem.split("_")
    map_name = "_".join(parts[:-1]) if parts[-1].startswith("Game") else stem
    game_num = parts[-1].replace("Game", "") if parts[-1].startswith("Game") else "0"
    logging.info(f"Map: {map_name}, Game: {game_num}")

    # Load team data
    teams_cfg = cfg.get("teams_config", "config/teams.json")
    with open(teams_cfg, encoding="utf-8") as f:
        teams = json.load(f)["teams"]

    # Output root
    output_root = Path(cfg["output"]["dir"]) / map_name / f"Game{game_num}"

    # ── Step 1: Extract frames ──
    pipeline = cfg.get("pipeline", {})
    if not pipeline.get("skip_extraction") and not args.track_only:
        logging.info("=== Step 1: Frame Extraction ===")
        map_dir = str(cfg.get("map_dir", "assets/map"))
        scan_cfg = cfg.get("scanning", {})
        extract_frames(
            video_path,
            map_dir=map_dir,
            output_dir=cfg["output"]["dir"],
            scan_step=scan_cfg.get("scan_step", 300),
            min_segment_frames=scan_cfg.get("min_segment_frames", 120),
        )
        if args.extract_only:
            return

    # Find all views
    views = sorted(output_root.glob("View*"))
    if not views:
        logging.error(f"No views found in {output_root}")
        sys.exit(1)
    logging.info(f"Processing {len(views)} views: {[v.name for v in views]}")

    # ── Step 2: Game UI analysis ──
    if pipeline.get("analyze_ui", True):
        logging.info("=== Step 2: Game UI Analysis ===")
        # Reset global timer for this game
        GameAnalyzer._global_timer_slope = None
        GameAnalyzer._global_timer_intercept = None

        for view_dir in views:
            logging.info(f"  Analyzing {view_dir.name}")
            analyzer = GameAnalyzer(teams_config=teams_cfg)
            result = analyzer.analyze_view(str(view_dir))
            s = result["summary"]
            logging.info(f"    {s['total_events']} events, "
                         f"{s.get('ranking_changes', 0)} ranking, "
                         f"{s.get('player_status_events', 0)} player")

    # ── Step 3: Map dot tracking ──
    track_teams = pipeline.get("track_teams", "all")
    if track_teams:
        logging.info("=== Step 3: Map Dot Tracking ===")
        if isinstance(track_teams, str) and track_teams != "all":
            track_teams = [track_teams]

        traj_cfg = cfg.get("trajectory", {})

        for view_dir in views:
            map_dir = view_dir / "map"
            if not map_dir.exists():
                continue
            map_region = str(view_dir / "map_region.png")

            for t in teams:
                if track_teams != "all" and t["name"] not in track_teams:
                    continue
                logging.info(f"  Tracking {t['name']} ({t['color']}) in {view_dir.name}")
                try:
                    track_team(
                        map_dir,
                        t["color"], t["players"], t["name"],
                        map_region=map_region,
                    )
                except Exception as e:
                    logging.warning(f"    Failed: {e}")

    logging.info("=== Done ===")
    logging.info(f"Output: {output_root}/")


if __name__ == "__main__":
    main()
