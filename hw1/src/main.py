__author__ = 'nikita_kartashov'

import cv2
import numpy as np

IMAGE = '/Users/nikita_kartashov/Documents/Work/cv/hw1/resource/Text.bmp'
RESULT = '/Users/nikita_kartashov/Documents/Work/cv/hw1/resource/result.bmp'

def make_square_kernel(n):
    return np.ones((n, n), np.uint8)


if __name__ == '__main__':
    image = cv2.imread(IMAGE, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_32F)
    # laplacian = cv2.convertScaleAbs(laplacian)
    threshed_image = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(RESULT, threshed_image)