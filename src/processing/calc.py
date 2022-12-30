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


def is_min_area_rect_in_bounds(
    min_area_rect: tuple[tuple[float, float], tuple[float, float], float],
    area_low: int,
    area_high: int,
) -> bool:
    width = min_area_rect[1][0]
    height = min_area_rect[1][1]
    area = width * height
    return area_low < area < area_high


def is_min_area_in_bounds_and_good_edges_ratio(
    contour: np.ndarray, area_low: int, area_high: int, ratio: float
) -> bool:
    min_rect = cv2.minAreaRect(contour)
    width = min_rect[1][0]
    height = min_rect[1][1]
    if width == 0 or height == 0:
        return False
    actual_ratio = max(width, height) / min(width, height)
    return (
        is_min_area_rect_in_bounds(min_rect, area_low, area_high)
        and abs(actual_ratio - ratio) < 0.15
    )


def get_mid_point_from_rect_points(points: np.ndarray) -> tuple[int, int]:
    mid_x = int(np.max(points[:, 0]) + np.min(points[:, 0])) // 2
    mid_y = int(np.max(points[:, 1]) + np.min(points[:, 1])) // 2
    return mid_x, mid_y
