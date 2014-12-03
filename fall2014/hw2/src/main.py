__author__ = 'nikita_kartashov'

import cv2
import numpy as np

from utils.cv_utils import make_kernel, read_image_grayscale_inverted


IMAGE = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw1/resource/Text.bmp'
INTERMEDIATE_IMAGE = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw2/resource/interm.bmp'
RESULT = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw2/resource/result.bmp'

BLUE = (255, 255, 0)


def generate_binary_groups(image):
    height, width = image.shape
    # Two pixels wider & higher
    image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    mask = np.zeros(shape=(height + 2, width + 2), dtype=np.uint8)
    resulting_rectangles = []
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if image[i, j] < 255:
                continue
            result_flag, bounding_rectangle_coords = \
                cv2.floodFill(image, mask, seedPoint=(j, i), newVal=0, flags=cv2.FLOODFILL_MASK_ONLY)
            if result_flag:
                resulting_rectangles.append(bounding_rectangle_coords)
    return resulting_rectangles


def draw_rectangles(image, rectangles):
    for rectangle in rectangles:
        x, y, w, h = rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color=BLUE, thickness=1)


if __name__ == '__main__':
    image = read_image_grayscale_inverted(IMAGE)
    dilated_image = image
    dilated_image = cv2.dilate(image, kernel=make_kernel(1, 5))
    image = dilated_image
    eroded_image = image

    eroded_image = cv2.erode(eroded_image, kernel=make_kernel(1, 3))
    image = eroded_image

    dilated_image = cv2.dilate(image, kernel=make_kernel(3, 1))
    image = dilated_image
    eroded_image = image

    eroded_image = cv2.erode(eroded_image, kernel=make_kernel(3, 1))
    image = eroded_image

    cv2.imwrite(INTERMEDIATE_IMAGE, image)

    rectangles = generate_binary_groups(image)
    color_image = cv2.imread(IMAGE, cv2.CV_LOAD_IMAGE_COLOR)
    draw_rectangles(color_image, rectangles)
    cv2.imwrite(RESULT, color_image)