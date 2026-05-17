"""Extract map viewport regions from video by matching to reference map."""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MapExtractor:
    """Scan video for map viewport segments, match to reference map,
    and extract the shown map region."""

    def __init__(self, video_path: str, map_dir: str = "assets/map",
                 output_dir: str = "output"):
        self.video_path = Path(video_path)
        self.map_dir = Path(map_dir)
        self.output_dir = Path(output_dir)

        # Parse filename: {MapName}_{GameNum}.mp4
        stem = self.video_path.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].startswith("Game"):
            self.map_name = "_".join(parts[:-1])
            self.game_num = parts[-1].replace("Game", "")
        else:
            raise ValueError(f"Cannot parse map/game from filename: {stem}")

        # Load reference map
        # Try to find reference map (handle naming variations)
        ref_path = self.map_dir / f"{self.map_name}_unlabeled.png"
        if not ref_path.exists():
            # Try with underscores replacing spaces
            alt_name = self.map_name.replace(" ", "_")
            ref_path = self.map_dir / f"{alt_name}_unlabeled.png"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference map not found for '{self.map_name}'. "
                f"Tried: {self.map_dir}/{self.map_name}_unlabeled.png"
            )
        self.ref_img = cv2.imread(str(ref_path))
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)

        # Pre-compute reference features for matching
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.ref_enhanced = clahe.apply(self.ref_gray)
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref_enhanced, None)

        logger.info(f"MapExtractor: {self.map_name} Game {self.game_num}")
        logger.info(f"  Reference map: {self.ref_img.shape[1]}x{self.ref_img.shape[0]}")
        logger.info(f"  Reference keypoints: {len(self.kp_ref)}")

    def scan(self, scan_step: int = 30, min_segment_frames: int = 60):
        """Scan video and find map viewport segments.

        Yields ViewSegment info dicts as they are found.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"  Scanning {total_frames} frames @ {fps:.0f}fps, step={scan_step}")

        # Coarse scan
        presence = []  # [(frame_num, is_map, viewport_rect)]
        for fn in range(0, total_frames, scan_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                continue
            is_map, rect = self._detect_viewport(frame)
            presence.append((fn, is_map, rect))

        cap.release()

        # Group into segments
        segments = self._group_segments(presence, total_frames, min_segment_frames)

        # Refine boundaries with binary search
        segments = self._refine_all_boundaries(segments, scan_step)

        for seg in segments:
            seg["fps"] = fps  # carry actual fps for duration calculation

        logger.info(f"  Found {len(segments)} map view segments")
        for seg in segments:
            dur = (seg["end_frame"] - seg["start_frame"]) / fps
            r = seg["viewport_rects"]
            logger.info(f"    View {seg['view_num']}: "
                        f"frames {seg['start_frame']}-{seg['end_frame']} "
                        f"({dur:.1f}s), map={r['map'][2]}x{r['map'][3]}, "
                        f"game={r['game'][2]}x{r['game'][3]}")

        return segments

    # Manually annotated viewport boundaries (full-frame pixel coords)
    VIEWPORT = {
        "map":  (1008, 112, 854, 852),   # (x, y, w, h)
        "game": (40, 258, 998, 559),
    }

    def _detect_viewport(self, frame):
        """Detect if map viewport is visible.

        Returns (is_map, viewport_rects) where viewport_rects is
        {"map": (x,y,w,h), "game": (x,y,w,h)} or None.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mx, my, mw, mh = self.VIEWPORT["map"]
        gx, gy, gw, gh = self.VIEWPORT["game"]

        # Check for sharp map left border (the main signature of map visibility)
        col_out = gray[my:my+mh, mx - 8:mx].mean()
        col_in = gray[my:my+mh, mx:mx + 8].mean()
        if abs(col_out - col_in) < 30:
            return False, None

        # Check edge density in map region
        map_region = gray[my:my+mh, mx:mx+mw]
        edges = cv2.Canny(map_region, 50, 150)
        if edges.mean() < 10:
            return False, None

        rects = {"map": (mx, my, mw, mh), "game": (gx, gy, gw, gh)}
        return True, rects

    @staticmethod
    def _group_segments(presence, total_frames, min_frames):
        segments = []
        in_seg = False
        seg_start = None
        seg_rects = []

        for fn, is_map, rects in presence:
            if is_map and not in_seg:
                in_seg = True
                seg_start = fn
                seg_rects = [rects] if rects else []
            elif is_map and in_seg:
                if rects:
                    seg_rects.append(rects)
            elif not is_map and in_seg:
                in_seg = False
                end = fn
                if end - seg_start >= min_frames:
                    avg_rects = MapExtractor._average_rects(seg_rects)
                    segments.append({
                        "view_num": len(segments),
                        "start_frame": seg_start,
                        "end_frame": end,
                        "viewport_rects": avg_rects,
                    })
                seg_rects = []

        if in_seg:
            end = total_frames - 1
            if end - seg_start >= min_frames:
                avg_rects = MapExtractor._average_rects(seg_rects)
                segments.append({
                    "view_num": len(segments),
                    "start_frame": seg_start,
                    "end_frame": end,
                    "viewport_rects": avg_rects,
                })

        return segments

    @staticmethod
    def _average_rects(rects_list):
        """Average viewport rects (dicts of {map:tuple, game:tuple}) for stability."""
        if not rects_list:
            return {"map": (1008, 112, 854, 852), "game": (40, 258, 998, 559)}
        result = {}
        for key in ["map", "game"]:
            vals = [(r[key][0], r[key][1], r[key][2], r[key][3]) for r in rects_list if r and key in r]
            if vals:
                result[key] = (int(np.mean([v[0] for v in vals])),
                               int(np.mean([v[1] for v in vals])),
                               int(np.mean([v[2] for v in vals])),
                               int(np.mean([v[3] for v in vals])))
            else:
                result[key] = (1008, 112, 854, 852) if key == "map" else (40, 258, 998, 559)
        return result

    def _refine_all_boundaries(self, segments, scan_step):
        """Refine segment start/end boundaries with binary search."""
        cap = cv2.VideoCapture(str(self.video_path))

        for seg in segments:
            # Refine start
            lo = max(0, seg["start_frame"] - scan_step)
            hi = min(seg["start_frame"] + scan_step, seg["end_frame"])
            for _ in range(8):
                mid = (lo + hi) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()
                if not ret:
                    break
                is_map, _ = self._detect_viewport(frame)
                if is_map:
                    hi = mid
                else:
                    lo = mid
            seg["start_frame"] = hi

            # Refine end
            lo = max(seg["start_frame"], seg["end_frame"] - scan_step)
            hi = seg["end_frame"] + scan_step
            for _ in range(8):
                mid = (lo + hi) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()
                if not ret:
                    break
                is_map, _ = self._detect_viewport(frame)
                if is_map:
                    lo = mid
                else:
                    hi = mid
            seg["end_frame"] = hi

        cap.release()
        return segments

    def match_and_extract(self, segment, frame_skip: int = 2, max_frames: int = None):
        """Match the viewport to reference map, extract frames, and save map region.

        Args:
            segment: dict with viewport_rects, start_frame, end_frame, view_num
            frame_skip: save every Nth frame
            max_frames: max number of frames to save (None = all)

        Returns:
            dict with map_path, num_frames, homography, or None on failure
        """
        view_num = segment["view_num"]
        rects = segment["viewport_rects"]
        mx, my, mw, mh = rects["map"]
        gx, gy, gw, gh = rects["game"]
        logger.info(f"  View {view_num}: map={mw}x{mh} game={gw}x{gh}")

        # Get a middle frame from the segment for map matching
        mid_frame = (segment["start_frame"] + segment["end_frame"]) // 2
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.warning(f"    Cannot read frame {mid_frame}")
            return None

        # Crop to map viewport
        map_viewport = frame[my:my + mh, mx:mx + mw]

        # Match viewport to reference map
        H = self._match_viewport(map_viewport)
        if H is None:
            logger.warning(f"    View {view_num}: matching failed")
            return None

        # Map viewport corners to reference map coordinates
        vp_h, vp_w = map_viewport.shape[:2]
        corners = np.float32([[0, 0], [vp_w, 0], [vp_w, vp_h], [0, vp_h]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        logger.info(f"    Viewport maps to ref: ({mapped[0,0]:.0f},{mapped[0,1]:.0f}) "
                    f"to ({mapped[2,0]:.0f},{mapped[2,1]:.0f})")

        # Crop the matching region from the reference map
        x1 = int(max(0, min(mapped[:, 0])))
        y1 = int(max(0, min(mapped[:, 1])))
        x2 = int(min(self.ref_img.shape[1], max(mapped[:, 0])))
        y2 = int(min(self.ref_img.shape[0], max(mapped[:, 1])))

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"    Invalid crop region: ({x1},{y1})-({x2},{y2})")
            return None

        map_crop = self.ref_img[y1:y2, x1:x2].copy()

        # Draw viewport outline on the crop for visualization
        local_corners = mapped.copy()
        local_corners[:, 0] -= x1
        local_corners[:, 1] -= y1
        cv2.polylines(map_crop, [local_corners.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 0), 2)

        # Output directories: output/{MapName}/{GameNum}/{ViewNum}/
        out_dir = (self.output_dir / self.map_name /
                   f"Game{self.game_num}" / f"View{view_num}")
        map_frames_dir = out_dir / "map"
        game_frames_dir = out_dir / "game"
        map_frames_dir.mkdir(parents=True, exist_ok=True)
        game_frames_dir.mkdir(parents=True, exist_ok=True)

        # Save map_region.png directly in View/
        map_region_path = out_dir / "map_region.png"
        cv2.imwrite(str(map_region_path), map_crop)
        logger.info(f"    Saved map_region to {map_region_path}")

        # Extract both map and game frames
        num_map, num_game, skipped = self._extract_frames(
            segment, map_frames_dir, game_frames_dir, frame_skip, max_frames)
        logger.info(f"    Saved {num_map} map + {num_game} game frames"
                    + (f" ({skipped} skipped)" if skipped else ""))

        # Format duration using actual video FPS
        fps = segment.get("fps", 30.0)
        duration_sec = (segment["end_frame"] - segment["start_frame"]) / fps
        duration_str = self._format_duration(duration_sec)

        # Save metadata
        import json
        meta = {
            "map_name": self.map_name,
            "game_num": self.game_num,
            "view_num": view_num,
            "frame_range": [segment["start_frame"], segment["end_frame"]],
            "duration": duration_str,
            "duration_seconds": round(duration_sec, 1),
            "viewport": {
                "map": {"x": mx, "y": my, "w": mw, "h": mh},
                "game": {"x": gx, "y": gy, "w": gw, "h": gh},
            },
            "ref_crop_region": {"x": int(x1), "y": int(y1),
                               "w": int(x2 - x1), "h": int(y2 - y1)},
            "homography": H.tolist(),
            "frames_saved": {"map": num_map, "game": num_game},
            "frame_skip": frame_skip,
            "fps": fps,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return {"map_path": str(map_region_path),
                "num_frames": num_map + num_game,
                "homography": H}

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds as HH:MM:SS, omitting HH if < 1 hour."""
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        else:
            return f"{m}:{s:02d}"

    def _extract_frames(self, segment, map_dir, game_dir, frame_skip, max_frames):
        """Extract map and game cropped frames from the video segment.

        Skips frames where the game UI timer is not visible (loading/inventory).

        Returns (map_count, game_count, skipped_count).
        """
        rects = segment["viewport_rects"]
        mx, my, mw, mh = rects["map"]
        gx, gy, gw, gh = rects["game"]

        cap = cv2.VideoCapture(str(self.video_path))
        map_count = 0
        game_count = 0
        skipped = 0

        for fn in range(segment["start_frame"], segment["end_frame"], frame_skip):
            if max_frames and max(map_count, game_count) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                break

            fh, fw = frame.shape[:2]

            # Check if game UI timer is visible (green bracket)
            # Timer in game view: (454, 8). In full frame: (gx+454, gy+8)
            timer_x, timer_y = gx + 454, gy + 8
            timer_w, timer_h = 92, 36
            if (timer_x + timer_w <= fw and timer_y + timer_h <= fh
                    and timer_x >= 0 and timer_y >= 0):
                timer_roi = frame[timer_y:timer_y+timer_h, timer_x:timer_x+timer_w]
                hsv = cv2.cvtColor(timer_roi, cv2.COLOR_BGR2HSV)
                green = cv2.inRange(hsv, (40, 80, 40), (80, 255, 255))
                if (green > 0).sum() / green.size <= 0.02:
                    skipped += 1
                    continue  # UI not loaded / covered

            # Map crop
            m_xs = max(0, min(mx, fw - 1))
            m_ys = max(0, min(my, fh - 1))
            m_ws = min(mw, fw - m_xs)
            m_hs = min(mh, fh - m_ys)
            if m_ws >= 100 and m_hs >= 100:
                map_crop = frame[m_ys:m_ys + m_hs, m_xs:m_xs + m_ws]
                cv2.imwrite(str(map_dir / f"frame_{fn:06d}.jpg"), map_crop)
                map_count += 1

            # Game crop
            g_xs = max(0, min(gx, fw - 1))
            g_ys = max(0, min(gy, fh - 1))
            g_ws = min(gw, fw - g_xs)
            g_hs = min(gh, fh - g_ys)
            if g_ws >= 100 and g_hs >= 100:
                game_crop = frame[g_ys:g_ys + g_hs, g_xs:g_xs + g_ws]
                cv2.imwrite(str(game_dir / f"frame_{fn:06d}.jpg"), game_crop)
                game_count += 1

        cap.release()
        return map_count, game_count, skipped

    def _match_viewport(self, viewport):
        """Match viewport image to reference map. Returns homography matrix."""
        vp_gray = cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        vp_enhanced = clahe.apply(vp_gray)

        kp_vp, des_vp = self.orb.detectAndCompute(vp_enhanced, None)
        if des_vp is None or len(kp_vp) < 20:
            logger.warning(f"    Only {len(kp_vp) if kp_vp else 0} viewport keypoints")
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.des_ref, des_vp)
        matches = sorted(matches, key=lambda x: x.distance)

        good = [m for m in matches if m.distance < 50]
        if len(good) < 10:
            logger.warning(f"    Only {len(good)} good matches")
            return None

        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp_vp[m.trainIdx].pt for m in good]).reshape(-1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        inliers = int(np.sum(mask))
        logger.info(f"    {len(good)} matches, {inliers} inliers")
        return H


def run(video_path: str, map_dir: str = "map",
        output_dir: str = "output", scan_step: int = 300,
        min_segment_frames: int = 60, frame_skip: int = 2,
        max_frames: int = None):
    """Main entry point: scan, match, extract all view segments."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    extractor = MapExtractor(video_path, map_dir, output_dir)
    segments = extractor.scan(scan_step, min_segment_frames)

    results = []
    for seg in segments:
        result = extractor.match_and_extract(seg, frame_skip, max_frames)
        results.append({"segment": seg, "result": result})

    return results


