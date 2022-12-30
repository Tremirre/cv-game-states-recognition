import cv2


def get_dots_detector() -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 40
    params.maxArea = 120
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.8
    return cv2.SimpleBlobDetector_create(params)


def get_dice_detector() -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 120
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByColor = True
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.7
    return cv2.SimpleBlobDetector_create(params)


def get_white_piece_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1200
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = True
    params.minConvexity = 0.6
    params.filterByInertia = False
    params.filterByColor = False
    return cv2.SimpleBlobDetector_create(params)


def get_black_piece_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1000
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = True
    params.minConvexity = 0.2
    params.filterByInertia = False
    params.filterByColor = False
    return cv2.SimpleBlobDetector_create(params)
