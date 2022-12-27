import cv2
import numpy as np


def calculate_similarity(frame_norm_1: np.ndarray, frame_norm_2: np.ndarray) -> float:
    return np.sum(frame_norm_1 * frame_norm_2)


def get_normalized_image(img: np.ndarray) -> np.ndarray:
    img_gs_norm = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255
    return img_gs_norm / np.sqrt(np.sum(img_gs_norm**2))
