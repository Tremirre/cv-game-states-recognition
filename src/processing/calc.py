import cv2
import numpy as np


def calculate_similarity(frame_norm_1: np.ndarray, frame_norm_2: np.ndarray) -> float:
    return np.sum(frame_norm_1 * frame_norm_2)


def get_normalized_image(img: np.ndarray) -> np.ndarray:
    img_gs_norm = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255
    return img_gs_norm / np.sqrt(np.sum(img_gs_norm**2))


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32), homography
    )


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_inside_bounding_rect(
    point: tuple[float, float], rect: tuple[float, float, float, float]
) -> bool:
    x, y, w, h = rect
    return x <= point[0] <= x + w and y <= point[1] <= y + h
