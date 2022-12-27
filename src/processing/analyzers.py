import abc
import threading

import cv2
import numpy as np

from typing import Callable

from .calc import calculate_similarity, get_normalized_image


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


class DiceAnalyzer(ThreadableAnalyzer):
    def __init__(
        self,
        blob_detector: cv2.SimpleBlobDetector,
        img_transformer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        threaded: bool = False,
    ) -> None:
        self.keypoints = []
        self.blob_detector = blob_detector
        self.img_transformer = img_transformer
        super().__init__(threaded)

    def analyze_job(self, frame: np.ndarray, **kwargs) -> None:
        transformed_frame = self.img_transformer(frame)
        self.keypoints = self.blob_detector.detect(transformed_frame)

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.drawKeypoints(
            frame,
            self.keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    def get_context(self) -> dict:
        return {"dice_dots": len(self.keypoints)}


class CardsAnalyzer(ThreadableAnalyzer):
    def __init__(
        self,
        edge_detector: Callable[[np.ndarray], np.ndarray],
        contour_filter: Callable[[list[np.ndarray]], list[np.ndarray]],
        threaded: bool = False,
    ) -> None:
        self.contours = []
        self.edge_detector = edge_detector
        self.contour_filter = contour_filter
        super().__init__(threaded)

    def analyze_job(self, frame: np.ndarray, **kwargs) -> None:
        edges = self.edge_detector(frame)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = self.contour_filter(contours)

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.drawContours(frame, self.contours, -1, (0, 255, 0), 3)

    def get_context(self) -> dict:
        return {"cards": len(self.contours)}
