__author__ = 'nikita_kartashov'

import cv2
import numpy as np

IMAGE = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw3/resource/mandril.bmp'
RESULT_FOURIER = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw3/resource/result_fourier.bmp'
RESULT_LAPLASIAN = '/Users/nikita_kartashov/Documents/Work/cv/fall2014/hw3/resource/result_laplasian.bmp'

FREQUENCY_BORDER = 30

if __name__ == '__main__':
    image = cv2.imread(IMAGE, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    after_dft_image = cv2.dft(np.float32(cv2.bitwise_not(image)))
    after_dft_shifted = np.fft.fftshift(after_dft_image)

    rows, columns = after_dft_image.shape
    center_row, center_column = rows / 2, columns / 2
    mask = np.ones(after_dft_image.shape, np.uint8)
    mask[center_row - FREQUENCY_BORDER:center_row + FREQUENCY_BORDER,
         center_column - FREQUENCY_BORDER:center_column + FREQUENCY_BORDER] = 0

    after_dft_shifted_masked = after_dft_shifted * mask
    after_dft_masked = np.fft.ifftshift(after_dft_shifted_masked)
    result_image = cv2.idft(after_dft_masked)

    laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=9)
    cv2.imwrite(RESULT_FOURIER, result_image)
    cv2.imwrite(RESULT_LAPLASIAN, laplacian)