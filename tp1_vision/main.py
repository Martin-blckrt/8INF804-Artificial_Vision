import random
import cv2
import numpy as np
import sys
import transfo
import utils
from matplotlib import pyplot as plt

scaling_factor = 0.25
fig = plt.figure(figsize=(15, 7))


def process(image_folder):

    # load all images in a dict
    imgs, imgs_ref = utils.load_imgs(image_folder)

    # read all reference images
    ref_chambre = cv2.imread(image_folder + '/Chambre/' + imgs_ref['Chambre'][0])
    ref_cuisine = cv2.imread(image_folder + '/Cuisine/' + imgs_ref['Cuisine'][0])
    ref_salon = cv2.imread(image_folder + '/Salon/' + imgs_ref['Salon'][0])

    # for each location, select a random image to analyse
    img_chambre = cv2.imread(image_folder + '/Chambre/' + random.choice(imgs['Chambre']))
    img_cuisine = cv2.imread(image_folder + '/Cuisine/' + random.choice(imgs['Cuisine']))
    img_salon = cv2.imread(image_folder + '/Salon/' + random.choice(imgs['Salon']))

    original_images = [ref_chambre, img_chambre, ref_cuisine, img_cuisine, ref_salon, img_salon]
    original_test_images = [img_chambre, img_cuisine, img_salon]

    # resize images
    test_images = transfo.resize_images(original_test_images, scaling_factor)
    images = transfo.resize_images(original_images, scaling_factor)

    # equalize histograms of each image
    eq_images = transfo.equalize_images(images)

    # apply masks to the images
    masked_images = transfo.mask_images(eq_images)

    # convert images to grey
    gray_images = transfo.gray_images(masked_images)

    # apply blur to the images
    blur_images = transfo.blur_images(gray_images, [10, 10])

    # absolute difference of the test and reference image
    diff_images = transfo.abs_diff_images(blur_images)

    # apply threshold to images
    thresh_images = transfo.threshold_images(diff_images)

    # apply dilation to images
    dilated_images = transfo.erode_images(thresh_images, np.ones((5, 5), np.uint8))

    # get the boundings
    bounded_images = transfo.bound_images(dilated_images, test_images)

    # show the results
    titles = []
    imgs_to_show = []

    for location in ['Chambre', 'Cuisine', 'Salon']:
        for step in ['référence', 'égalisée', 'thresholdée + dilatée', 'résultat final']:
            titles.append(location + ' : ' + step)

    j = 0
    for i in range(0, len(images), 2):
        imgs_to_show.append(images[i])
        imgs_to_show.append(eq_images[i + 1])
        imgs_to_show.append(dilated_images[j])
        imgs_to_show.append(bounded_images[j])
        j += 1

    utils.show_all_with_matplotlib(imgs_to_show, titles)
    plt.show()


if __name__ == "__main__":
    process(sys.argv[1])
    # C://Users/marti/PycharmProjects/tp1_vision/Images
