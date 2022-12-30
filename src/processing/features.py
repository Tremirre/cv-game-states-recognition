import cv2
import numpy as np


def get_clear_edges(
    image: np.ndarray, cannny_threshold_1: float = 70, cannny_threshold_2: float = 250
) -> np.ndarray:
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gs_blur = cv2.GaussianBlur(image_gs, (5, 5), 0)
    image_edges = cv2.Canny(image_gs_blur, cannny_threshold_1, cannny_threshold_2)
    return cv2.dilate(image_edges, np.ones((3, 3), np.uint8), iterations=1)


def get_clear_inverted_edges(
    image: np.ndarray,
    cannny_threshold_1: float = 60,
    cannny_threshold_2: float = 100,
    erosion_count: int = 2,
) -> np.ndarray:
    edges = cv2.Canny(image, cannny_threshold_1, cannny_threshold_2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.bitwise_not(edges)
    return cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=erosion_count)


def get_threshold_edges(img: np.ndarray) -> np.ndarray:
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gs_blur = cv2.GaussianBlur(img_gs, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        img_gs_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    thresholded = cv2.erode(thresholded, kernel=np.ones((3, 3), np.uint8), iterations=1)
    return thresholded


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


def find_dice_throwing_rect(image: np.ndarray) -> tuple[int, int, int, int] | None:
    edge_image = get_clear_edges(image)
    contours = get_contours(edge_image)
    contour_bounding_rects = [cv2.boundingRect(c) for c in contours]
    contour_bounding_rects = [
        r for r in contour_bounding_rects if 100000 < r[2] * r[3] < 200000
    ]
    sorted_bounding_rects = sorted(
        contour_bounding_rects, key=lambda r: get_image_fragment_by_rect(image, r).std()
    )
    if not sorted_bounding_rects:
        return None
    return sorted_bounding_rects[0]


def get_board_rect_from_homography(
    board_to_image_homography: np.ndarray, board_image: np.ndarray
) -> tuple[int, int, int, int]:
    full_rect = cv2.boundingRect(cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY))
    box_points = np.array(
        [
            [full_rect[0], full_rect[1]],
            [full_rect[0] + full_rect[2], full_rect[1]],
            [full_rect[0] + full_rect[2], full_rect[1] + full_rect[3]],
            [full_rect[0], full_rect[1] + full_rect[3]],
        ]
    )
    transformed = cv2.perspectiveTransform(
        box_points.reshape(-1, 1, 2).astype(np.float32), board_to_image_homography
    )
    return cv2.boundingRect(transformed)


def get_board_min_rect_from_homography(
    board_to_image_homography: np.ndarray, board_image: np.ndarray
) -> tuple[int, int, int, int]:
    full_rect = cv2.boundingRect(cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY))
    box_points = np.array(
        [
            [full_rect[0], full_rect[1]],
            [full_rect[0] + full_rect[2], full_rect[1]],
            [full_rect[0] + full_rect[2], full_rect[1] + full_rect[3]],
            [full_rect[0], full_rect[1] + full_rect[3]],
        ]
    )
    transformed = cv2.perspectiveTransform(
        box_points.reshape(-1, 1, 2).astype(np.float32), board_to_image_homography
    )
    return cv2.minAreaRect(transformed)


def get_rect_around_point(
    point: tuple[int, int], image: np.ndarray, width: int, height: int
) -> np.ndarray:
    x, y = point
    rect = (int(x) - width // 2, int(y) - height // 2, width, height)
    return get_image_fragment_by_rect(image, rect)
