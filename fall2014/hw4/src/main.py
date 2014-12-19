__author__ = 'nikita_kartashov'

import cv2
import numpy as np

from utils.cv_utils import read_image_grayscale, draw_matches


IMAGE = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw3/resource/mandril.bmp'
RESULT = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw4/resource/result.bmp'
KEYPOINTS = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw4/resource/keypoints.bmp'
NEW_KEYPOINTS = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw4/resource/new_keypoints.bmp'


def transform_image(image):
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
    return cv2.warpAffine(image, rotation_matrix, image.shape)


def transform_keypoints(points, cols, rows):
    transformation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
    return np.array([np.dot(transformation_matrix, point) for point in points])


NFEATURES = 1000
EPSILON = 20

if __name__ == '__main__':
    image = read_image_grayscale(IMAGE)
    sift = cv2.SIFT(NFEATURES)
    initial_keypoints, initial_descriptors = sift.detectAndCompute(image, None)
    old_points = np.array(map(lambda x: np.array([x.pt[0], x.pt[1], 1], np.float32), initial_keypoints))
    transformed_points = transform_keypoints(old_points, image.shape[0], image.shape[1])
    keypoint_image = cv2.drawKeypoints(image, initial_keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    transformed_image = transform_image(image)

    new_keypoints, new_descriptors = sift.detectAndCompute(transformed_image, None)
    new_keypoint_image = cv2.drawKeypoints(transformed_image, new_keypoints,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(initial_descriptors, new_descriptors, k=2)

    really_matched_points = []
    for i, (m, n) in enumerate(matches):
        distance = np.linalg.norm(transformed_points[m.queryIdx] - new_keypoints[m.trainIdx].pt)
        really_matched_points.append(1 if distance < EPSILON else 0)

    fraction_matched = sum(really_matched_points) * 1.0 / (NFEATURES if NFEATURES else len(initial_keypoints))
    print('{0} fraction of points of interest were matched with epsilon = {1}'.format(fraction_matched, EPSILON))

    result_image = draw_matches(image, initial_keypoints, transformed_image, new_keypoints, matches)

    cv2.imwrite(NEW_KEYPOINTS, new_keypoint_image)
    cv2.imwrite(KEYPOINTS, keypoint_image)
    cv2.imwrite(RESULT, result_image)