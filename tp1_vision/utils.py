from matplotlib import pyplot as plt
import cv2
import numpy as np
import os


def mask_image(img, location):
    mask = np.full(img.shape[:2], 255, dtype="uint8")

    if location == 'Chambre':
        bed_pts = np.array([[750, 0], [750, 190], [1500, 400], [1500, 0]])
        mask = cv2.rectangle(mask, (600, 0), (100, 350), 0, -1)
        mask = cv2.fillPoly(mask, np.int32([bed_pts]), 0)
        mask = cv2.rectangle(mask, (0, 600), (550, 1000), 0, -1)
    elif location == 'Cuisine':
        bg_pts = np.array([[0, 470], [0, 1000], [210, 1000], [450, 470]])
        mask = cv2.rectangle(mask, (0, 0), (1500, 470), 0, -1)
        mask = cv2.rectangle(mask, (1030, 0), (1500, 630), 0, -1)
        mask = cv2.fillPoly(mask, np.int32([bg_pts]), 0)
    elif location == 'Salon':
        bg_pts = np.array([[0, 0], [0, 600], [1000, 400], [1000, 0]])
        mask = cv2.fillPoly(mask, np.int32([bg_pts]), 0)

    return cv2.bitwise_and(img, img, mask=mask)


def show_all_with_matplotlib(imgs, titles):
    for i, (img, title) in enumerate(zip(imgs, titles)):

        # Convert BGR image to RGB if the image is in color :
        if len(img.shape) == 3:
            img = img[:, :, ::-1]

        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis('off')


def load_imgs(path):
    imgs = {}
    imgs_ref = {}

    for location in os.listdir(path):
        imgs["{0}".format(location)] = []
        imgs_ref["{0}".format(location)] = []
        for images in os.listdir(path + '/' + location):
            if images.split('.')[0] == 'Reference':
                imgs_ref["{0}".format(location)].append(images)
            else:
                imgs["{0}".format(location)].append(images)

    return imgs, imgs_ref


def merge_overlapping_rectangles(rectangles):
    # create a copy of the list to avoid modifying it while iterating
    rects = rectangles.copy()

    # iterate over each rectangle in the list
    for i, rect1 in enumerate(rectangles[:-1]):
        # check if this rectangle overlaps with any of the remaining rectangles
        for j, rect2 in enumerate(rectangles[i+1:-1], start=i+1):
            if (rect1[0] < rect2[0] + rect2[2] and
                    rect1[0] + rect1[2] > rect2[0] and
                    rect1[1] < rect2[1] + rect2[3] and
                    rect1[1] + rect1[3] > rect2[1]):
                # merge the rectangles into a larger one
                new_x = min(rect1[0], rect2[0])
                new_y = min(rect1[1], rect2[1])
                new_w = max(rect1[0] + rect1[2], rect2[0] + rect2[2]) - new_x
                new_h = max(rect1[1] + rect1[3], rect2[1] + rect2[3]) - new_y
                merged_rect = (new_x, new_y, new_w, new_h)

                # replace the overlapping rectangles with the merged one
                rects[i] = merged_rect
                rects.pop(j)
                break

    return rects
