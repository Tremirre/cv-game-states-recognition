import cv2
import numpy as np

from typing import Protocol


class FrameProcessor(Protocol):
    def on_start(self) -> None:
        ...

    def process_frame(self, frame: np.ndarray) -> bool:
        ...

    def on_finish(self) -> None:
        ...


class FrameAnalyzer(Protocol):
    def analyze_raw(self, frame: np.ndarray, **kwargs) -> None:
        ...

    def mutate_frame(self, frame: np.ndarray) -> np.ndarray:
        ...

    def get_context(self) -> dict:
        ...


class VideoHandler:
    def __init__(self, path_to_video: str) -> None:
        self.path_to_video = path_to_video
        self.capture = cv2.VideoCapture(path_to_video)
        if not self.capture.isOpened():
            raise RuntimeError(f"Can't open video file [{path_to_video}]")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_in_sec = self.frame_count / self.fps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.capture.release()

    def get_frame(self, frame_number: int) -> np.ndarray:
        if frame_number < 0:
            raise ValueError("Frame number must be positive")
        if frame_number > self.frame_count:
            raise ValueError("Frame number must be less than frame count")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = self.capture.read()
        return frame

    def get_frame_by_time(self, time_in_sec: float) -> np.ndarray:
        frame_number = int(time_in_sec * self.fps)
        return self.get_frame(frame_number)

    def go_through_video(
        self, frame_processor: FrameProcessor, frame_analyzers: list[FrameAnalyzer]
    ) -> None:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_processor.on_start()
        context = {"frame_number": 0}
        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break
                for analyzer in frame_analyzers:
                    analyzer.analyze_raw(frame, **context)
                    context.update(analyzer.get_context())
                for analyzer in frame_analyzers:
                    frame = analyzer.mutate_frame(frame)
                continue_processing = frame_processor.process_frame(frame)
                if not continue_processing:
                    break
                context["frame_number"] += 1
        finally:
            frame_processor.on_finish()
