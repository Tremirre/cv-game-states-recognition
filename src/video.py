import cv2
import numpy as np

from time import perf_counter
from typing import Protocol


class FrameProcessor(Protocol):
    def on_start(self) -> None:
        ...

    def process_frame(self, frame: np.ndarray, context: dict) -> bool:
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

    @property
    def video_name(self) -> str:
        return self.path_to_video.split("/")[-1].split(".")[0]

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
        self,
        frame_processor: FrameProcessor,
        frame_analyzers: list[FrameAnalyzer],
        additional_context: dict | None = None,
    ) -> None:
        if additional_context is None:
            additional_context = {}
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_processor.on_start()
        context = {
            "video_name": self.video_name,
            "frame_number": 0,
            **additional_context,
        }
        try:
            while True:
                time_start = perf_counter()
                ret, frame = self.capture.read()
                if not ret:
                    break
                for analyzer in frame_analyzers:
                    analyzer.analyze_raw(frame, **context)
                    context.update(analyzer.get_context())
                for analyzer in frame_analyzers:
                    frame = analyzer.mutate_frame(frame)
                continue_processing = frame_processor.process_frame(frame, context)
                print(context)
                if not continue_processing:
                    break
                context["frame_number"] += 1
                time_end = perf_counter()
                time_delta = time_end - time_start
                context["fps"] = 1 / time_delta
        finally:
            frame_processor.on_finish()
