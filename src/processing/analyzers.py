import abc
import threading

import cv2
import numpy as np

from typing import Callable

from .calc import (
    calculate_similarity,
    get_normalized_image,
    is_inside_bounding_rect,
    get_mid_point_from_rect_points,
)
from .features import get_rect_around_point, get_clear_inverted_edges


class Analyzer(abc.ABC):
    @abc.abstractmethod
    def analyze_raw(self, frame: np.ndarray, **kwargs) -> None:
        ...

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def get_context(self) -> dict:
        return {}


class ThreadableAnalyzer(Analyzer):
    def __init__(self, threaded: bool = False) -> None:
        self.threaded = threaded
        self.running = False

    @abc.abstractmethod
    def analyze_job(self, frame: np.ndarray, **kwargs) -> None:
        ...

    def _task(self, *args, **kwargs) -> None:
        self.analyze_job(*args, **kwargs)
        self.running = False

    def analyze_raw(self, frame: np.ndarray, **kwargs) -> None:
        if not self.threaded:
            return self.analyze_job(frame, **kwargs)
        if not self.running:
            self.running = True
            threading.Thread(target=self._task, args=(frame,), kwargs=kwargs).start()

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def get_context(self) -> dict:
        return {}


class SimilarityAnalyzer(Analyzer):
    def __init__(self, threshold: float, history_size: int) -> None:
        self.threshold = threshold
        self.prev_frame_norm = None
        self.history = [1.0] * history_size
        self.high_activity_on_frame = False

    def analyze_raw(self, frame: np.ndarray, **kwargs) -> None:
        frame_gs_norm = get_normalized_image(frame)
        similarity = 1.0
        if self.prev_frame_norm is not None:
            similarity = calculate_similarity(self.prev_frame_norm, frame_gs_norm)
        self.history.append(similarity)
        self.history.pop(0)

        average_similarity = np.mean(self.history)
        self.high_activity_on_frame = average_similarity < self.threshold

        self.prev_frame_norm = frame_gs_norm

    def get_context(self) -> dict:
        return {"high_activity": self.high_activity_on_frame}

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        text = "LOW ACTIVITY"
        color = (0, 255, 0)
        if self.high_activity_on_frame:
            text = "HIGH ACTIVITY"
            color = (0, 0, 255)
        return cv2.putText(
            frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA
        )


class DotsAnalyzer(ThreadableAnalyzer):
    def __init__(
        self,
        blob_detector: cv2.SimpleBlobDetector,
        img_transformer=lambda x: x,
        threaded: bool = False,
    ) -> None:
        self.dice_keypoints = []
        self.card_keypoints = []
        self.blob_detector = blob_detector
        self.img_transformer = img_transformer
        super().__init__(threaded)

    def analyze_job(
        self,
        frame: np.ndarray,
        dice_area_rect: tuple[int, int, int, int] | None = None,
        board_points: np.ndarray | None = None,
        **kwargs
    ) -> None:
        dice_keypoints = []
        card_keypoints = []
        transformed_frame = self.img_transformer(frame)
        keypoints = self.blob_detector.detect(transformed_frame)
        for kp in keypoints:
            if dice_area_rect is not None:
                if is_inside_bounding_rect(kp.pt, dice_area_rect):
                    dice_keypoints.append(kp)
                else:
                    card_keypoints.append(kp)
            else:
                card_keypoints.append(kp)
        if board_points is not None:
            card_keypoints = [
                kp
                for kp in card_keypoints
                if cv2.pointPolygonTest(board_points, kp.pt, False) < 0
            ]
        self.dice_keypoints = dice_keypoints
        self.card_keypoints = card_keypoints

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.drawKeypoints(
            frame,
            self.dice_keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        frame = cv2.drawKeypoints(
            frame,
            self.card_keypoints,
            np.array([]),
            (20, 255, 150),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return frame

    def get_context(self) -> dict:
        return {
            "dice_dots": self.dice_keypoints,
            "card_dots": self.card_keypoints,
        }


class CardsAnalyzer(ThreadableAnalyzer):
    def __init__(
        self,
        edge_detector: Callable[[np.ndarray], np.ndarray],
        contour_filter: Callable[[list[np.ndarray]], list[np.ndarray]],
        threaded: bool = False,
    ) -> None:
        self.bounding_rects = []
        self.edge_detector = edge_detector
        self.contour_filter = contour_filter
        super().__init__(threaded)

    def analyze_job(
        self, frame: np.ndarray, board_points: np.ndarray | None = None, **kwargs
    ) -> None:
        edges = self.edge_detector(frame)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = self.contour_filter(contours)
        self.bounding_rects = [cv2.boundingRect(c) for c in contours]
        if board_points is not None:
            self.bounding_rects = [
                rect
                for rect in self.bounding_rects
                if cv2.pointPolygonTest(
                    board_points,
                    (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2),
                    False,
                )
                < 0
            ]

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        for rect in self.bounding_rects:
            cv2.rectangle(frame, rect, (0, 255, 0), 3)
        return frame

    def get_context(self) -> dict:
        return {"cards": len(self.bounding_rects)}


class BlackPieceAnalyzerCircleMethod(ThreadableAnalyzer):
    def __init__(self, threaded: bool = False) -> None:
        super().__init__(threaded=threaded)
        self.circles = []
        self.black_piece_pos = None

    def analyze_job(
        self, frame: np.ndarray, board_points: np.ndarray | None = None, **kwargs
    ) -> None:
        frame_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            frame_gs,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=40,
            param2=40,
            minRadius=27,
            maxRadius=50,
        )
        circles = np.uint16(np.around(circles))[0].tolist()
        if board_points is not None:
            circles = [
                circle
                for circle in circles
                if cv2.pointPolygonTest(board_points, circle[:2], False) > 0
            ]
        circles_with_color = [
            (circle, get_rect_around_point(circle[:2], frame_gs, 40, 40).mean())
            for circle in circles
        ]
        circles_with_color = [
            (circle, color) for circle, color in circles_with_color if 10 < color < 80
        ]
        if not circles_with_color:
            return
        circles_with_color = sorted(
            circles_with_color, key=lambda circle_with_color: circle_with_color[1]
        )
        circles = list(zip(*circles_with_color))[0]
        self.circles = circles
        self.black_piece_pos = (int(circles[0][0]), int(circles[0][1]))

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.black_piece_pos is not None:
            frame = cv2.drawMarker(
                frame, self.black_piece_pos, (255, 0, 0), cv2.MARKER_CROSS, 20, 2
            )
        return frame

    def get_context(self) -> dict:
        return {
            "black_pos": self.black_piece_pos,
        }


class BlackPieceAnalyzer(ThreadableAnalyzer):
    def __init__(
        self,
        piece_detector: cv2.SimpleBlobDetector,
        binarizer: Callable[[np.ndarray], np.ndarray],
        threaded: bool = False,
    ) -> None:
        super().__init__(threaded=threaded)
        self.binarizer = binarizer
        self.piece_detector = piece_detector
        self.black_piece_pos = None

    def analyze_job(
        self, frame: np.ndarray, board_points: np.ndarray | None = None, **kwargs
    ) -> None:
        frame_bin = self.binarizer(frame)
        keypoints = self.piece_detector.detect(frame_bin)
        if board_points is not None:
            keypoints = [
                kp
                for kp in keypoints
                if cv2.pointPolygonTest(board_points, kp.pt, False) > 0
            ]
            midpoint = get_mid_point_from_rect_points(board_points)
            scaled = np.int32((board_points - midpoint) * 0.66 + midpoint)
            keypoints = [
                kp for kp in keypoints if cv2.pointPolygonTest(scaled, kp.pt, False) < 0
            ]
        self.all_keypoints = keypoints
        keypoints_with_color = [
            (kp, get_rect_around_point(kp.pt, frame, 60, 60).mean()) for kp in keypoints
        ]
        keypoints_with_color = [kp for kp in keypoints_with_color if 20 < kp[1] < 70]
        if not keypoints_with_color:
            return
        keypoints_with_color = sorted(keypoints_with_color, key=lambda kp: kp[1])
        keypoints = list(zip(*keypoints_with_color))[0]
        self.black_piece_pos = (int(keypoints[0].pt[0]), int(keypoints[0].pt[1]))

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.black_piece_pos is not None:
            frame = cv2.drawMarker(
                frame, self.black_piece_pos, (255, 0, 0), cv2.MARKER_CROSS, 20, 2
            )
        return frame

    def get_context(self) -> dict:
        return {
            "black_pos": self.black_piece_pos,
        }


class WhitePieceAnalyzer(ThreadableAnalyzer):
    def __init__(
        self, piece_detector: cv2.SimpleBlobDetector, binarizer: Callable[[np.ndarray], np.ndarray], threaded: bool = False
    ) -> None:
        super().__init__(threaded=threaded)
        self.piece_detector = piece_detector
        self.binarizer = binarizer
        self.white_piece_pos = None

    def analyze_job(
        self, frame: np.ndarray, board_points: np.ndarray | None = None, **kwargs
    ) -> None:
        frame_bin = self.binarizer(frame)
        keypoints = self.piece_detector.detect(frame_bin)
        if board_points is not None:
            keypoints = [
                kp
                for kp in keypoints
                if cv2.pointPolygonTest(board_points, kp.pt, False) > 0
            ]
        if not keypoints:
            return
        keypoints_with_color = [
            (kp, get_rect_around_point(kp.pt, frame, 60, 60).mean()) for kp in keypoints
        ]
        keypoints_with_color = [kp for kp in keypoints_with_color if kp[1] > 180]
        if not keypoints_with_color:
            return
        keypoints_with_color = sorted(
            keypoints_with_color, key=lambda kp: kp[1], reverse=True
        )
        keypoints = list(zip(*keypoints_with_color))[0]
        self.white_piece_pos = (int(keypoints[0].pt[0]), int(keypoints[0].pt[1]))

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.white_piece_pos is not None:
            frame = cv2.drawMarker(
                frame, self.white_piece_pos, (0, 0, 255), cv2.MARKER_CROSS, 20, 2
            )
        return frame

    def get_context(self) -> dict:
        return {
            "white_pos": self.white_piece_pos,
        }
