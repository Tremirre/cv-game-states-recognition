import cv2
import numpy as np


class LiveFrameProcessor:
    def __init__(self, window_name: str, width: int, height: int) -> None:
        self.window_name = window_name
        self.width = width
        self.height = height

    def on_start(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

    def process_frame(self, frame: np.ndarray, context: dict) -> bool:
        if not cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE):
            return False
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(3)
        if key == ord("q"):
            return False
        if key == ord("s"):
            video_name = context["video_name"]
            frame_number = context["frame_number"]
            cv2.imwrite(f"{video_name}_{frame_number}.jpg", frame)
        return True

    def on_finish(self) -> None:
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE):
            cv2.destroyWindow(self.window_name)


class SaverFrameProcessor:
    def __init__(
        self, path_to_video: str, fourcc, fps: int, shape: tuple[int, int]
    ) -> None:
        self.path_to_video = path_to_video
        self.writer = None
        self.fourcc = fourcc
        self.fps = fps
        self.shape = shape

    def on_start(self) -> None:
        self.writer = cv2.VideoWriter(
            self.path_to_video,
            self.fourcc,
            self.fps,
            self.shape,
        )

    def process_frame(self, frame: np.ndarray) -> bool:
        if self.writer is None:
            raise RuntimeError("Writer is not initialized")
        self.writer.write(frame)
        return True

    def on_finish(self) -> None:
        if self.writer is not None:
            self.writer.release()
