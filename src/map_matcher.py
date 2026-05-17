"""Match cropped map viewport to reference map using feature matching."""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MapMatcher:
    """Match viewport frames to reference map using ORB features + homography."""

    def __init__(self, reference_map_path: str):
        self.ref_path = reference_map_path
        self.ref_img = cv2.imread(reference_map_path)
        if self.ref_img is None:
            raise FileNotFoundError(f"Cannot load reference map: {reference_map_path}")

        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        self.ref_h, self.ref_w = self.ref_gray.shape

        # Enhance contrast for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.ref_enhanced = clahe.apply(self.ref_gray)

        # Pre-compute reference features
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref_enhanced, None)
        logger.info(f"Reference map: {self.ref_w}x{self.ref_h}, "
                    f"{len(self.kp_ref)} keypoints")

    def match(self, viewport: np.ndarray) -> np.ndarray:
        """Match viewport to reference map and return homography matrix.

        Args:
            viewport: BGR image of the cropped map viewport

        Returns:
            3x3 homography matrix mapping viewport pixels → reference map pixels,
            or None if matching fails
        """
        vp_gray = cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        vp_enhanced = clahe.apply(vp_gray)

        kp_vp, des_vp = self.orb.detectAndCompute(vp_enhanced, None)

        if des_vp is None or len(kp_vp) < 10:
            logger.warning("Not enough keypoints in viewport")
            return None

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.des_ref, des_vp)
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter good matches
        good = [m for m in matches if m.distance < 50]
        if len(good) < 10:
            logger.warning(f"Only {len(good)} good matches (need >=10)")
            return None

        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp_vp[m.trainIdx].pt for m in good]).reshape(-1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            logger.warning("Homography computation failed")
            return None

        inliers = int(np.sum(mask))
        logger.info(f"Map matched: {len(good)} good matches, {inliers} inliers")

        return H

    def viewport_to_map(self, px: float, py: float, H: np.ndarray) -> tuple[float, float]:
        """Convert viewport pixel coordinates to reference map coordinates."""
        pt = np.array([[px, py]], dtype=np.float32).reshape(1, 1, 2)
        mapped = cv2.perspectiveTransform(pt, H)
        return (float(mapped[0][0][0]), float(mapped[0][0][1]))

    def viewport_corners_to_map(self, H: np.ndarray, vp_w: int, vp_h: int) -> np.ndarray:
        """Get the 4 corners of the viewport in reference map coordinates."""
        corners = np.float32([[0, 0], [vp_w, 0], [vp_w, vp_h], [0, vp_h]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(corners, H).reshape(-1, 2)
