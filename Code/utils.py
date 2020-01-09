import rasterio
import numpy as np
import pandas as pd
import skimage
import fiona
import pyproj
import glob

def to_single_geotiff(path_in, path_out):
    """ Convert a folder of multiple geotiff images as a single multiband geotiff
        INPUT : path_in (str) -> path to the input folder as (path_to_folder/../*.tiff)
                path_out (str) -> path to where the geotiff will be saved (path_to_folder/../image_name.tiff)
        OUTPUT : None
    """
    file_path = [f for f in sorted(glob.glob(path_in))]
    # Read metadata of first file
    with rasterio.open(file_path[0]) as src0:
        meta = src0.meta
    # Update meta to reflect the number of layers
    meta.update(count=len(file_path))
    # Read each layer and write it to stack
    with rasterio.open(path_out, 'w', **meta) as dst:
        for id, path in enumerate(file_path, start=1):
            with rasterio.open(path) as src1:
                dst.write_band(id, src1.read(1))

def load_geotiff(path, window=None, in_range='uint16', out_range=(0,1)):
    """ Load the geotiff as a list of numpy array.
        INPUT : path (str) -> the path to the geotiff
                window (raterio.windows.Window) -> the window to use when loading the image
                in_range (str or tuple) -> the in_range to use for the skimage.exposure.rescale_intensity (default is 'uint16')
                out_range (str or tuple) -> the out_range to use for the skimage.exposure.rescale_intensity (default is (0,1))
        OUTPUT : band (list of numpy array) -> the different bands as float scalled to 0:1
                 meta (dictionnary) -> the metadata associated with the geotiff
    """
    with rasterio.open(path) as f:
        #band = [skimage.exposure.rescale_intensity(f.read(i+1, window=window).astype('float64'), in_range=in_range, out_range=out_range) for i in range(f.count)]
        band = [skimage.img_as_float(f.read(i+1, window=window)) for i in range(f.count)]
        meta = f.meta
        if window != None:
            meta['height'] = window.height
            meta['width'] = window.width
            meta['transform'] = f.window_transform(window)

    return band, meta

def load_shapefile(path, projection):
    """ Load the shapefile as a list of numpy array of coordinates
        INPUT : path (str) -> the path to the shapefile
                proj (pyproj.Proj) -> the projection to use to convert the lat/lon into UTM
        OUTPUT : poly (list of np.array) -> list of polygons (as numpy.array of coordinates)
    """
    with fiona.open(path) as shapefile:
        features = [feature["geometry"] for feature in shapefile]
        proj_in = pyproj.Proj(shapefile.crs)

    poly = [np.array([pyproj.transform(proj_in, projection, coord[0], coord[1]) for coord in features[i]['coordinates'][0]]) for i in range(len(features))]

    return poly

def load_target_shp(path, transform=None, projection=None):
    """ Load the shapefile as a list of numpy array of coordinates
        INPUT : path (str) -> the path to the shapefile
                transform (rasterio.Affine) -> the affine transformation to get the polygon in row;col format from UTM.
        OUTPUT : poly (list of np.array) -> list of polygons (as numpy.array of coordinates)
                 poly_rc (list of np.array) -> list of polygon in row-col format if a transform is given
    """
    with fiona.open(path) as shapefile:
        proj_in = pyproj.Proj(shapefile.crs)
        class_type = [feature['properties']['id']+1 for feature in shapefile]
        features = [feature["geometry"] for feature in shapefile]
    # reproject polygons if necessary
    if projection is None:
        poly = [np.array([(coord[0], coord[1]) for coord in features[i]['coordinates'][0]]) for i in range(len(features))]
    else:
        poly = [np.array([pyproj.transform(proj_in, projection, coord[0], coord[1]) for coord in features[i]['coordinates'][0]]) for i in range(len(features))]
    poly_rc = None

    # transform in row-col if a transform is given
    if not transform is None:
        poly_rc = [np.array([rasterio.transform.rowcol(transform, coord[0], coord[1])[::-1] for coord in p]) for p in poly]

    return poly, poly_rc, class_type

def compute_mask(polygon_list, img_w, img_h, val_list):
    """ Get mask of class of a polygon list
        INPUT : polygon_list (list od polygon in coordinates (x, y)) -> the polygons in row;col format
                img_w (int) -> the image width
                img_h (int) -> the image height
                val_list(list of int) -> the class associated with each polygon
        OUTPUT : img (np.array 2D) -> the mask in which the pixel value reflect it's class (zero being the absence of class)
    """
    img = np.zeros((img_h, img_w), dtype=np.uint8) #skimage : row,col --> h,w
    for polygon, val in zip(polygon_list, val_list):
        rr, cc = skimage.draw.polygon(polygon[:,1], polygon[:,0], img.shape)
        img[rr, cc] = val

    return img

def scores_summary(scores, score_list):
    """ return a score summary of the CV results in a pandas dataframe
        INPUT : scores (dictionnary of scores) -> the sklearn CV results
                score_list (list of string) -> the name of the scores used
        OUTPUT : df (pandas.DataFrame) -> The summary scores in a DataFrame
    """
    tmp_df = []
    for name, score in scores.items():
        scores_df = pd.DataFrame(score).drop(columns=['fit_time', 'score_time'])
        scores_df.columns = pd.MultiIndex.from_arrays([list(sum(zip(score_list, score_list), ())),['test','train']*len(score_list)])
        s_agg = scores_df.agg(['mean','std'], axis=0).transpose()
        s_agg.columns = pd.MultiIndex.from_arrays([[name,name],['mean','std']])
        tmp_df.append(s_agg)

    return pd.concat(tmp_df, axis=1)

def human_format(num, pos=None):
    """ Format large number using a human interpretable unit (kilo, mega, ...)
        INPUT : num (int) -> the number to reformat
        OUTPUT : num (str) -> the reformated number
    """
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def contrast_adjust(band, val):
    """ Adjust the contrast of the image by stretching the histogram in order to saturate the upper and
        lower quantile defined by val.
        INPUT : band (list of np.array) -> the bands of the goetiff to stretch.
                val (tuple) -> the upper and lower quantile to saturate (ex: (1,99))
        OUTPUT : band_adj (3D np.array) -> the stacked adjusted bands
    """
    band_adj = [skimage.exposure.rescale_intensity(img, in_range=tuple(np.percentile(img, val)), out_range=(0,1)) for img in band]
    return np.stack(band_adj, axis=2)
