import cv2
import imutils
import utils
import numpy as np


def resize_images(inputs, scaling_factor):
    outputs = []
    for img in inputs:
        outputs.append(cv2.resize(img, (0, 0), fx=scaling_factor, fy=scaling_factor))
    return outputs


def mask_images(inputs):
    outputs = []
    for index, img in enumerate(inputs):
        if index < 2:
            outputs.append(utils.mask_image(img, 'Chambre'))
        elif index > 3:
            outputs.append(utils.mask_image(img, 'Salon'))
        else:
            outputs.append(utils.mask_image(img, 'Cuisine'))
    return outputs


def blur_images(inputs, ksize):
    outputs = []
    for img in inputs:
        outputs.append(cv2.blur(img, ksize))
    return outputs


def clahe_equalize_images(inputs, clip):
    outputs = []
    for img in inputs:
        cla = cv2.createCLAHE(clipLimit=clip)
        channels = cv2.split(img)
        eq_channels = []
        for ch in channels:
            eq_channels.append(cla.apply(ch))

        outputs.append(cv2.merge(eq_channels))
    return outputs


def equalize_images(inputs):
    outputs = []
    for img in inputs:
        colorimage_b = cv2.equalizeHist(img[:, :, 0])
        colorimage_g = cv2.equalizeHist(img[:, :, 1])
        colorimage_r = cv2.equalizeHist(img[:, :, 2])
        colorimage_e = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
        outputs.append(colorimage_e)
    return outputs


def equalizeHist_images(inputs):
    outputs = []
    for img in inputs:
        outputs.append(cv2.equalizeHist(img))
    return outputs


def abs_diff_images(inputs):
    outputs = []
    for i in range(0, len(inputs), 2):
        outputs.append(cv2.absdiff(inputs[i], inputs[i + 1]))
    return outputs


def gray_images(inputs):
    outputs = []
    for img in inputs:
        outputs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return outputs


def threshold_images(inputs):
    outputs = []
    for img in inputs:
        outputs.append(cv2.threshold(img, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    return outputs


def erode_images(inputs, kernel):
    outputs = []
    for img in inputs:
        outputs.append(cv2.erode(img, kernel, iterations=1))
    return outputs


def dilate_images(inputs, kernel):
    outputs = []
    for img in inputs:
        outputs.append(cv2.dilate(img, kernel, iterations=3))
    return outputs


def canny_images(inputs, lower, upper):
    outputs = []
    for img in inputs:
        outputs.append(cv2.Canny(img, lower, upper))
    return outputs


def bound_images(thresh_inputs, test_inputs):
    outputs = []
    for (thresh_img, test_img) in zip(thresh_inputs, test_inputs):
        t_img = test_img.copy()
        contours = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        big_contour = max(contours, key=cv2.contourArea)

        # create a list to hold all the bounding rectangles
        rectangles = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = (x, y, w, h)

            # check if this rectangle is inside any existing rectangles
            contained = False
            for idx, r in enumerate(rectangles):
                if x >= r[0] and y >= r[1] and x + w <= r[0] + r[2] and y + h <= r[1] + r[3]:
                    # this rectangle is fully contained in an existing rectangle
                    contained = True
                    break
                elif x <= r[0] and y <= r[1] and x + w >= r[0] + r[2] and y + h >= r[1] + r[3]:
                    # this rectangle fully contains an existing rectangle
                    # replace the existing rectangle with this one
                    rectangles[idx] = rect
                    contained = True
                    break
                elif x < r[0] + r[2] and x + w > r[0] and y < r[1] + r[3] and y + h > r[1]:
                    # this rectangle overlaps with an existing rectangle
                    # merge the rectangles into a larger one
                    new_x = min(x, r[0])
                    new_y = min(y, r[1])
                    new_w = max(x + w, r[0] + r[2]) - new_x
                    new_h = max(y + h, r[1] + r[3]) - new_y
                    rectangles[idx] = (new_x, new_y, new_w, new_h)
                    contained = True
                    break

            # if this rectangle is not inside any existing rectangles
            # add it to the list of rectangles
            if not contained:
                rectangles.append(rect)

        # filter the merged rectangles by area
        filtered_rectangles = [r for r in rectangles if (r[2] * r[3]) > cv2.contourArea(big_contour) * 0.06]

        filtered_rectangles = utils.merge_overlapping_rectangles(filtered_rectangles)

        # draw the filtered bounding rectangles
        for rect in filtered_rectangles:
            x, y, w, h = rect
            cv2.rectangle(t_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        outputs.append(t_img)

    return outputs
