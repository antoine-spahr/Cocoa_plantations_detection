import numpy as np
import skimage
from skimage.morphology import disk
from skimage.feature import local_binary_pattern

#<*> adjust histogram
def stretchlim(img, values):
    """ returns the CDF value (define in list val) of the image img
        INPUT : img (2D np.array) -> the image to get the limit on
                values (list of floats) -> the values of the cdf for which the corresponding pixel values are wanted
        OUTPUT : (list of floats) -> the pixel values associated with the provided cdf values
    """
    if type(values) != list:
        values= [values]

    cdf, cdf_center = skimage.exposure.cumulative_distribution(img, nbins=256)
    cdf = (cdf - cdf.min())/(cdf.max()-cdf.min())

    return [cdf_center[np.where(cdf >= val)[0][0]] for val in values]

def adjust_intensity(img, values=[0.01,0.99]):
    """ Adjust the intensity of the images in the list by saturating the upper and lower 1%
        INPUT : img (3D np.array) -> the image to adjust with dimension (H x W x Bands)
                values (list of 2 float) -> the quantile at which the image will be saturated (default [0.01, 0.99])
        OUTPUT : img_adj (np.array) -> the adjusted image
    """
    img_adj = [skimage.exposure.rescale_intensity(img[:,:,i], in_range=tuple(stretchlim(img[:,:,i], [0.01, 0.99])), out_range=(0,1)) for i in range(img.shape[2])]
    return np.stack(img_adj, axis=2)
#<*!>

# NDVI
def NDVI(R, NIR):
    """ Compute the NDVI
        INPUT : R (np.array) -> the Red band images as a numpy array of float
                NIR (np.array) -> the Near Infrared images as a numpy array of float
        OUTPUT : NDVI (np.array) -> the NDVI
    """
    NDVI = (NIR - R) / (NIR + R + 1e-12)
    return NDVI

def preprocessing(img, img_to_match=None):
    """  """
    new_bands = []
    # keep only band some band : 10m R, G, B, NIR (1,2,3,7) ; 20m (4,5,6,10,11,12)
    img = img[:,:,[1,2,3,4,5,6,7,10,11,12]]

    # Histogram Matching or Histogram streching
    # if not img_to_match is None:
    #     img = skimage.transform.match_histograms(img, img_to_match[:,:,:img.shape[2]], multichannel=True)
    # else:
    #     img = np.stack([skimage.exposure.rescale_intensity(img[:,:,i], in_range=tuple(np.percentile(img[:,:,i], (0.5, 99.5))), out_range=(0,1)) for i in range(img.shape[2])], axis=2)

    # compute NDVI
    ndvi = NDVI(img[:,:,2], img[:,:,6])
    new_bands.append(ndvi)

    # compute green NDVI (G, R instead of R NIR)
    new_bands.append(NDVI(img[:,:,1], img[:,:,2]))
    # define images to further process : R, NIR, NDVI
    img_to_process = [img[:,:,2], img[:,:,6], ndvi]
    # entropy on R, NIR and NDVI
    for im in img_to_process:
        new_bands.append(skimage.filters.rank.entropy(im, selem=disk(2)))

    # morphological opening/closing reconstruction on R, NIR and NDVI
    se = skimage.morphology.disk(5)
    for im in img_to_process:
        new_bands.append(skimage.morphology.reconstruction(skimage.morphology.erosion(im), im, method='dilation', selem=se))
        new_bands.append(skimage.morphology.reconstruction(skimage.morphology.dilation(im), im, method='erosion', selem=se))

    # LBP on R, NIR, NDVI
    for im in img_to_process:
        new_bands.append(local_binary_pattern(im, 8, 3, method='uniform'))

    return np.append(img, np.stack(new_bands, axis=2), axis=2)
