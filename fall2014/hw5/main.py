__author__ = 'nikita_kartashov'

import cv2
import numpy as np

from copy import copy

VIDEO = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw5/resource/sequence.mpg'
INITIAL_HARRIS_POINTS = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw5/resource/initial_harris_keypoints.bmp'
INITIAL_FAST_POINTS = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw5/resource/initial_fast_keypoints.bmp'
RESULT_HARRIS = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw5/resource/result_harris.avi'
RESULT_FAST = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw5/resource/result_fast.avi'


def graify(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def write_features(result_path, image, features, colors):
    image = copy(image)
    for i, feature in enumerate(features):
        a, b = feature[0]
        a, b = map(int, [a, b])
        cv2.circle(image, (a, b), 3, colors[i].tolist(), -1)
    cv2.imwrite(result_path, image)


def open_video_sources_outputs(input_path, output_paths):
    source_video = cv2.VideoCapture(input_path)
    width = int(source_video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(source_video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(source_video.get(cv2.cv.CV_CAP_PROP_FPS))
    fourcc = int(source_video.get(cv2.cv.CV_CAP_PROP_FOURCC))
    return source_video, [cv2.VideoWriter(path, fourcc, fps, (width, height)) for path in output_paths]


MAX_POINTS = 200
SHOW_TRACKING_LINES = True


def detect_features_harris(image):
    feature_params = dict(useHarrisDetector=True,
                          maxCorners=MAX_POINTS,
                          qualityLevel=0.15,
                          minDistance=7,
                          blockSize=7)

    return cv2.goodFeaturesToTrack(image, mask=None, **feature_params)


def detect_features_fast(image):
    fast = cv2.FastFeatureDetector()
    best_features = list(sorted(fast.detect(image, None), key=lambda f: f.response, reverse=True))[:MAX_POINTS]
    features = [[kp.pt] for kp in best_features]
    return features

def perform_feature_tracking(algorithm_stacks):
    source_video, result_videos = open_video_sources_outputs(VIDEO, RESULT_HARRIS)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    colors = np.random.randint(0, 255, (MAX_POINTS, 3))


if __name__ == '__main__':
    source_video, result_videos = open_video_sources_outputs(VIDEO, RESULT_HARRIS)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    colors = np.random.randint(0, 255, (MAX_POINTS, 3))

    ret, old_frame = source_video.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_features_harris = detect_features_harris(old_gray)
    old_features_fast = detect_features_fast(old_gray)
    write_features(INITIAL_HARRIS_POINTS, old_frame, old_features_harris, colors)
    write_features(INITIAL_FAST_POINTS, old_frame, old_features_fast, colors)
    old_features_fast = detect_features_fast(old_gray)

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = source_video.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_features, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features_harris, None, **lk_params)

        good_new = new_features[st == 1]
        good_old = old_features_harris[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = map(int, [a, b, c, d])
            if SHOW_TRACKING_LINES:
                cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 1)
            cv2.circle(frame, (a, b), 3, colors[i].tolist(), -1)
        result_frame = cv2.add(frame, mask)
        result_videos.write(result_frame)

        old_gray = frame_gray.copy()
        old_features_harris = good_new.reshape(-1, 1, 2)

    source_video.release()
    result_videos.release()