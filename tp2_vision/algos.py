from scipy import ndimage as ndi
from skimage.filters import sobel
import numpy as np
from skimage import morphology
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import rank
from skimage.morphology import disk, h_maxima


def algo1(og_img, treated_img):
    treated_img = (treated_img * 255).astype(int)

    # sobel elevation
    elevate = sobel(treated_img)

    # markers
    marker = np.zeros_like(rgb2gray(og_img))
    marker[rgb2gray(og_img) < 0.16] = 1
    marker[rgb2gray(og_img) > 0.28] = 2

    coords = peak_local_max(elevate, footprint=morphology.diamond(3), labels=treated_img)

    mask = np.zeros(elevate.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    # watershed
    water = watershed(elevate, markers, mask=treated_img, watershed_line=True)

    # individual segmentation
    segmentation = ndi.binary_fill_holes(water - 1)

    return segmentation


def algo2(og_img, treated_img):
    # canny edge detection
    edges = canny(treated_img)

    # filling
    fill = ndi.binary_fill_holes(edges)

    # removing small objects
    cleaned = morphology.remove_small_objects(fill, 30)

    return cleaned


def algo3(og_img, treated_img):
    segments_fz = felzenszwalb(treated_img, scale=110, sigma=8, min_size=40)

    return segments_fz


def algo4(og_img, treated_img):
    segments_slic = slic(treated_img, n_segments=40, compactness=0.2, sigma=0.5, channel_axis=None)

    return segments_slic


def algo5(og_img, treated_img):
    segments_quick = quickshift(treated_img, kernel_size=15, max_dist=40, ratio=1.7)

    return segments_quick


def algo6(og_img, treated_img):
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(treated_img, disk(5)) < 10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) to keep edges thin
    gradient = rank.gradient(treated_img, disk(2))

    # process the watershed
    labels = watershed(gradient, markers)

    # individual segmentation
    segmentation = ndi.binary_fill_holes(labels - 1)

    return segmentation


def algorithm(og_img, treated_img):
    distance = ndi.distance_transform_edt(treated_img)
    coords = h_maxima(distance, np.amax(distance) * 0.1)
    np.zeros(distance.shape, dtype=bool)
    markers, _ = ndi.label(coords)

    labels = watershed(-distance, markers, mask=treated_img, watershed_line=True)

    return labels
