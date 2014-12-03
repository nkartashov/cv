__author__ = 'nikita_kartashov'

from math import sqrt

import cv2
import numpy as np

from utils.cv_utils import read_image_grayscale, draw_matches


IMAGE = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw3/resource/mandril.bmp'
RESULT = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw4/resource/result.bmp'
KEYPOINTS = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw4/resource/keypoints.bmp'
NEW_KEYPOINTS = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw4/resource/new_keypoints.bmp'


def rotate45(image):
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    return cv2.warpAffine(image, rotation_matrix, image.shape)


def resize_half(image):
    return cv2.resize(image, (image.shape[0] / 2, image.shape[1] / 2))


def manually_transform_keypoints(points):
    rotation_matrix = np.empty((2, 2), np.float32)
    rotation_matrix.fill(sqrt(2) / 2)
    rotation_matrix[0, 1] *= -1
    rotation_matrix[1, 0] *= 1

    resize_matrix = np.zeros((2, 2), np.float32)

    resize_matrix[0, 0] = 0.5
    resize_matrix[1, 1] = 0.5
    transformation_matrix = np.dot(resize_matrix, rotation_matrix)
    return [np.dot(transformation_matrix, point) for point in points]


NFEATURES = 100
EPSILON = 50

if __name__ == '__main__':
    image = read_image_grayscale(IMAGE)
    sift = cv2.SIFT(NFEATURES)
    initial_keypoints, initial_descriptors = sift.detectAndCompute(image, None)
    old_points = np.array(map(lambda x: np.array(x.pt, np.float32), initial_keypoints))
    transformed_points = manually_transform_keypoints(old_points)
    keypoint_image = cv2.drawKeypoints(image, initial_keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    rotated_image = rotate45(image)
    resized_image = resize_half(rotated_image)

    new_keypoints, new_descriptors = sift.detectAndCompute(resized_image, None)
    new_keypoint_image = cv2.drawKeypoints(resized_image, new_keypoints,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(initial_descriptors, new_descriptors, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    really_matched_points = []
    for i, (m, n) in enumerate(matches):
        distance = np.linalg.norm(transformed_points[i] - new_keypoints[m.trainIdx].pt)
        really_matched_points.append(1 if distance < EPSILON else 0)

    fraction_matched = sum(really_matched_points) * 1.0 / (NFEATURES if NFEATURES else len(initial_keypoints))
    print('{0} fraction of points of interest were matched with epsilon = {1}'.format(fraction_matched, EPSILON))

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    result_image = draw_matches(image, initial_keypoints, resized_image, new_keypoints, matches)

    cv2.imwrite(NEW_KEYPOINTS, new_keypoint_image)
    cv2.imwrite(KEYPOINTS, keypoint_image)
    cv2.imwrite(RESULT, result_image)