"""Speed heatmap visualization from trajectory data."""

import cv2
import json
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_heatmap(trajectories: dict, map_path: str, output_path: str,
                     map_w: int = 854, map_h: int = 852,
                     blur_radius: int = 25, speed_multiplier: float = 1.0):
    """Generate a speed heatmap overlaid on the map.

    Args:
        trajectories: {player_id: [(x, y, fn), ...]}
        map_path: path to map_region_labeled.png or map_region.png
        output_path: where to save the heatmap
        map_w, map_h: output dimensions
        blur_radius: Gaussian blur radius for heatmap smoothing
        speed_multiplier: scale speed values (for visibility)
    """
    # Load background
    bg = cv2.imread(map_path)
    if bg is not None:
        bg = cv2.resize(bg, (map_w, map_h))
    else:
        bg = np.zeros((map_h, map_w, 3), dtype=np.uint8)

    # Accumulate speed onto a float canvas
    heat = np.zeros((map_h, map_w), dtype=np.float32)
    weight = np.zeros((map_h, map_w), dtype=np.float32)
    dt = 2.0 / 59.94  # seconds per extracted frame

    for pid, pts in trajectories.items():
        for i in range(1, len(pts)):
            x1, y1 = int(pts[i-1][0]), int(pts[i-1][1])
            x2, y2 = int(pts[i][0]), int(pts[i][1])

            if not (0 <= x1 < map_w and 0 <= y1 < map_h):
                continue
            if not (0 <= x2 < map_w and 0 <= y2 < map_h):
                continue

            dx, dy = x2 - x1, y2 - y1
            speed = np.sqrt(dx*dx + dy*dy) / dt * speed_multiplier

            # Draw speed onto heatmap (thick line = more visible)
            cv2.line(heat, (x1, y1), (x2, y2), speed, thickness=6)
            cv2.line(weight, (x1, y1), (x2, y2), 1.0, thickness=6)

    # Normalize: divide by weight where weight > 0
    mask = weight > 0
    heat[mask] /= weight[mask]

    # Gaussian blur for smooth heatmap
    if blur_radius > 0:
        heat = cv2.GaussianBlur(heat, (blur_radius | 1, blur_radius | 1), 0)

    # Normalize to 0-255
    if heat.max() > 0:
        heat_norm = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat_norm = np.zeros_like(heat, dtype=np.uint8)

    # Apply color map (JET: blue=slow, red=fast)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

    # Blend with background
    heat_mask = heat_norm > 5
    bg[heat_mask] = cv2.addWeighted(bg[heat_mask], 0.3, heat_color[heat_mask], 0.7, 0)

    # Color bar on the right
    bar_w = 20
    bar = np.zeros((map_h, bar_w, 3), dtype=np.uint8)
    for y in range(map_h):
        val = int((1 - y / map_h) * 255)
        bar[y, :] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
    result = cv2.hconcat([bg, bar])

    # Speed labels on color bar
    if heat.max() > 0:
        max_speed = heat.max()
        cv2.putText(result, f'{max_speed:.0f} px/s', (map_w + 2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(result, '0 px/s', (map_w + 2, map_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.imwrite(output_path, result)
    logger.info(f"Heatmap saved to {output_path}")


def generate_per_player_heatmaps(view_dir: str, team_name: str):
    """Generate speed heatmaps for each player in a view."""
    traj_dir = Path(view_dir) / "trajectory"
    stats_path = traj_dir / f"stats_{team_name}.json"
    map_path = str(Path(view_dir) / "map_region_labeled.png")
    if not Path(map_path).exists():
        map_path = str(Path(view_dir) / "map_region.png")

    if not stats_path.exists():
        logger.warning(f"No stats for {team_name} in {view_dir}")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    for pid, data in stats.items():
        pts = [(p["x"], p["y"], p["frame"]) for p in data["points"]]
        traj = {pid: pts}
        out = str(traj_dir / f"heatmap_{team_name}_{pid}.jpg")
        generate_heatmap(traj, map_path, out)
