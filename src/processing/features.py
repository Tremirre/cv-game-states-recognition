import cv2
import numpy as np


def get_clear_edges(image: np.ndarray) -> np.ndarray:
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gs_blur = cv2.GaussianBlur(image_gs, (5, 5), 0)
    image_edges = cv2.Canny(image_gs_blur, 70, 250)
    return cv2.dilate(image_edges, np.ones((3, 3), np.uint8), iterations=1)


def get_contours(edge_image: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def get_bounding_rects_with_area_from_image(
    image: np.ndarray, area_min: float, area_max: float
) -> list[tuple[int, int, int, int]]:
    edge_image = get_clear_edges(image)
    contours = get_contours(edge_image)
    contour_bounding_rects = [cv2.boundingRect(c) for c in contours]
    contour_bounding_rects = [
        r for r in contour_bounding_rects if area_min < r[2] * r[3] < area_max
    ]
    return contour_bounding_rects


def get_image_fragment_by_rect(
    image: np.ndarray, rect: tuple[int, int, int, int]
) -> np.ndarray:
    x, y, w, h = rect
    return image[y : y + h, x : x + w]


def find_dice_throwing_area(image: np.ndarray) -> tuple[int, int, int, int] | None:
    edge_image = get_clear_edges(image)
    contours = get_contours(edge_image)
    contour_bounding_rects = [cv2.boundingRect(c) for c in contours]
    contour_bounding_rects = [
        r for r in contour_bounding_rects if 100000 < r[2] * r[3] < 200000
    ]
    sorted_bounding_rects = sorted(contour_bounding_rects, key=lambda r: get_image_fragment_by_rect(image, r).std())
    if not sorted_bounding_rects:
        return None
    return sorted_bounding_rects[0]
