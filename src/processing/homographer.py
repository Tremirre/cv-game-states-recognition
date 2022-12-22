import cv2
import numpy as np


def coords_from_keypoints_matches(
    keypoints: list[cv2.KeyPoint], indexes: list[int]
) -> np.ndarray:
    return np.float32([keypoints[index].pt for index in indexes]).reshape(-1, 1, 2)


class Homographer:
    def __init__(
        self,
        reference_keypoints: list[cv2.KeyPoint],
        matched_keypoints: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
    ):
        self.kp_ref = reference_keypoints
        self.kp_match = matched_keypoints
        self.matches = matches
        query_indexes = [match.queryIdx for match in self.matches]
        train_indexes = [match.trainIdx for match in self.matches]
        self.src_pts = coords_from_keypoints_matches(self.kp_ref, query_indexes)
        self.dst_pts = coords_from_keypoints_matches(self.kp_match, train_indexes)

    def get_homography(
        self, method: int = cv2.RANSAC, threshold: float = 5.0
    ) -> tuple[np.ndarray, np.ndarray]:
        return cv2.findHomography(self.src_pts, self.dst_pts, method, threshold)

    def get_inverse_homography(
        self, method: int = cv2.RANSAC, threshold: float = 5.0
    ) -> tuple[np.ndarray, np.ndarray]:
        return cv2.findHomography(self.dst_pts, self.src_pts, method, threshold)
