__author__ = 'nikita_kartashov'

import cv2
from copy import copy

from utils.cv_utils import make_kernel

IMAGE = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw1/resource/result.bmp'
RESULT = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw2/resource/result.bmp'

BLUE = (255, 255, 0)

if __name__ == '__main__':
    image = cv2.imread(IMAGE, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # dilated_image = cv2.dilate(image, kernel=make_kernel(5, 1))
    # dilated_image = cv2.dilate(dilated_image, kernel=make_kernel(3, 3))
    # eroded_image = cv2.erode(dilated_image, kernel=make_kernel(3, 3))
    # eroded_image = cv2.erode(eroded_image, kernel=make_kernel(5, 1))
    eroded_image = image
    image_for_contours = copy(eroded_image)
    contours, hierarchy = cv2.findContours(image_for_contours, mode=cv2.RETR_LIST, method=cv2.RETR_LIST)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(eroded_image, (x, y), (x + w, y + h), color=BLUE, thickness=1)
    resulting_image = eroded_image

    cv2.imwrite(RESULT, resulting_image)