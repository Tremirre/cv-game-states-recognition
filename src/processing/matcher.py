import cv2
import numpy as np


class ImageMatcher:
    def __init__(self, reference_img: np.ndarray, matched_img: np.ndarray):
        self.reference_img = reference_img
        self.matched_img = matched_img
        self.reference_img_gray = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2GRAY)
        self.matched_img_gray = cv2.cvtColor(self.matched_img, cv2.COLOR_BGR2GRAY)
        self.sift = cv2.SIFT_create()
        self.kp_ref, self.desc_ref = self.sift.detectAndCompute(
            self.reference_img_gray, None
        )
        self.kp_match, self.desc_match = self.sift.detectAndCompute(
            self.matched_img_gray, None
        )
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def get_matches(self) -> list[cv2.DMatch]:
        matches = sorted(
            self.bf.match(self.desc_ref, self.desc_match), key=lambda x: x.distance
        )
        return matches

    def get_image_with_matches(self, matches: list[cv2.DMatch]) -> np.ndarray:
        return cv2.drawMatches(
            self.reference_img,
            self.kp_ref,
            self.matched_img,
            self.kp_match,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
