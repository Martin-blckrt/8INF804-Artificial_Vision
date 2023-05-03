import sys
import skimage.io
import cv2
from skimage.restoration import denoise_bilateral
from skimage.color import rgb2gray
import algos
from scipy import ndimage as ndi
from skimage import filters
import numpy as np
import pandas as pd
import skimage.segmentation
import os


def pretreatment(img):
    # convert to grayscale
    image = rgb2gray(img)

    # threshold the image
    _, thresh = cv2.threshold(image, 0.15, 255, cv2.THRESH_TOZERO)

    # use bilateral denoise to "smooth" the grains
    denoise = denoise_bilateral(thresh, sigma_color=0.1, sigma_spatial=9, channel_axis=None)

    return denoise


def algo_pretreatment(img):
    # convert to grayscale
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold the image
    thresh = filters.threshold_triangle(image)

    # use bilateral denoise to "smooth" the grains
    denoise = denoise_bilateral(thresh, sigma_color=0.1, sigma_spatial=9, channel_axis=None)
    binary = image > denoise
    return binary


def posttreatment(og_img, treated_img):
    list = []
    # label the segments
    labeled, _ = ndi.label(treated_img)

    # find all contours
    grains = ndi.find_objects(labeled)

    for grain in grains:
        list.append(og_image[grain])

    return list


def color_df(frag_list):
    avg_list = []

    for frag in frag_list:
        avg_list.append("%.3f" % elem for elem in np.average(frag, axis=(0, 1)))  # all 3 channels at once

    df = pd.DataFrame(avg_list, columns=['avgB', 'avgG', 'avgR'], dtype=float)

    return df


if __name__ == "__main__":
    folder_dir = sys.argv[1]
    big_df = pd.DataFrame()

    for i, img in enumerate(os.listdir(folder_dir)):
        # load an image
        og_image = skimage.io.imread(folder_dir + '/' + img)

        pretreated_img = algo_pretreatment(og_image)
        treated_img = algos.algorithm(og_image, pretreated_img)
        grain_list = posttreatment(og_image, treated_img)
        dataframe = color_df(grain_list)
        big_df = pd.concat([big_df, dataframe])

    big_df = big_df.reset_index()
    big_df.rename(lambda x: 'grain_' + str(x + 1), axis=0, inplace=True)
    print(big_df)
