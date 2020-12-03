import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


def Kernel_convolution(image, kernel_filter):
    image_h = image.size[0]
    image_w = image.size[1]
    kernel_h = kernel_filter.shape[0]
    kernel_w = kernel_filter.shape[1]

    h = kernel_h
    w = kernel_w
    new_image = np.zeros(image.size)
    for i in range(h, image_h - h):
        for j in range(w, image_w - w):
            sum = 0
            for n in range(kernel_h):
                for m in range(kernel_w):
                    sum = sum + kernel_filter[m][n] * image[i - h + m][j - w + n]
            new_image[i][j] = sum
    return new_image


def openImage(image):
    imageRGB = Image.open(image).convert('L')
    imageArray = np.array(image)
    print("image opend")
    return imageArray


def saveImage(name, image):
    plt.imsave(name, image)


def Sharpening_filters(image):
    kernel = (np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]]))
    filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    return filtered


def Line_and_Edge_Enhancing_filters(image):
    kernel = (np.array([[-1, -1, -1],
                        [2, 2.9, 2],
                        [-1, -1, -1]]))
    filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    return filtered


def Laplacian_filters(image):
    kernel = (np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]]))
    filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    return filtered


def Gaussian_filters(image):
    kernel = (np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]])) / 16
    filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    return filtered


def Mean_filters(image):
    kernel = np.ones((3, 3)) / 9
    filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    return filtered


def UnsharpeMasking_filters(image):
    kernel = (np.array([[1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, -476, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1]])) * -1 / 256
    filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    return filtered


def Sobel_filters(image):
    kernel1 = (np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]]))
    kernel2 = (np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]]))

    filtered1 = cv2.filter2D(src=image, kernel=kernel1, ddepth=-1)
    filtered = cv2.filter2D(src=filtered1, kernel=kernel2, ddepth=-1)
    return filtered


def Laplacian_of_Gaussian_filter(image):
    karnel = (np.array([[0, 1, 1, 2, 2, 2, 1, 1, 0],
                        [1, 2, 4, 5, 5, 5, 4, 2, 1],
                        [1, 4, 5, 3, 0, 3, 5, 4, 1],
                        [2, 5, 3, -12, -24, -12, 3, 2, 1],
                        [2, 5, 0, -24, -40, -24, 0, 5, 2],
                        [2, 5, 3, -12, -24, -12, 3, 2, 1],
                        [1, 4, 5, 3, 0, 3, 5, 4, 1],
                        [1, 2, 4, 5, 5, 5, 4, 2, 1],
                        [0, 1, 1, 2, 2, 2, 1, 1, 0]]))
    filtered = cv2.filter2D(src=image, kernel=karnel, ddepth=-1)

    return filtered


def pipline(image):
    # Im = openImage(image)
    step1 = Sharpening_filters(image)
    step2 = Gaussian_filters(step1)
    step3 = UnsharpeMasking_filters(step2)
    step4 = Mean_filters(step3)

    cv2.imshow('pipline', step4)
    cv2.imwrite("pip.jpg", step4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread("kk.jpg")

    filter = Sharpening_filters(image)
    cv2.imshow('sharpening', filter)
    cv2.imwrite("pip.jpg", filter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
