import cv2
import numpy as np


class BoardMapper:
    def __init__(
        self,
        homography: np.ndarray,
        board_length: int,
        side_section_size: int = 9,
        field_width_to_height: float = 0.63225,
    ):
        self.homography = homography
        self.board_length = board_length
        self.side_section_size = side_section_size
        self.field_width_to_height = field_width_to_height
        self.tile_height = int(board_length / self.board_length_in_field_heights)
        self.tile_width = int(self.tile_height * self.field_width_to_height)

    @property
    def board_length_in_field_heights(self) -> float:
        return 2 + self.field_width_to_height * self.side_section_size

    def map_point_to_field_position(self, x: int, y: int) -> tuple[str, int]:
        x, y = cv2.perspectiveTransform(
            np.expand_dims([x, y], (0, 1)).astype(np.float32), self.homography
        ).flatten()
        if x < 0 or y < 0 or x > self.board_length or y > self.board_length:
            return "outside", 0
        if (
            self.tile_height < x < self.board_length - self.tile_height
            and self.tile_width < y < self.board_length - self.tile_width
        ):
            return "inside", 0
        field_x = int((x - self.tile_height) / self.tile_width)
        field_y = int((y - self.tile_width) / self.tile_height)

        if x < self.tile_height:
            if y < self.tile_width:
                return "top_left", 0
            if y > self.board_length - self.tile_width:
                return "bottom_left", 0
            return "left", field_y

        if x > self.board_length - self.tile_width:
            if y < self.tile_width:
                return "top_right", 0
            if y > self.board_length - self.tile_width:
                return "bottom_right", 0
            return "right", field_y

        if y < self.tile_width:
            return "top", field_x
        return "bottom", field_x
