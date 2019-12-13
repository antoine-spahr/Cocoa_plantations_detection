import rasterio
import rasterio.plot as rioplot
from rasterio.merge import merge
import numpy as np
import glob

def merge_tiles(path_list, out_path, bounds=None):
    """ Merge the raster images given in the path list and save the results on disk.
        INPUT : path_list (list of string) -> the path to all the images to merge
                out_path (str) -> the path and file name to which the merge is saved
                bounds (tuple) -> (left, bottom, right, top) the boundaries to extract from (in UTM).
        OUTPUT : None
    """
    # open all tiles
    src_file_mosaic = []
    for fpath in path_list:
        src = rasterio.open(fpath)
        src_file_mosaic.append(src)
    # merge the files into a single mosaic
    mosaic, out_trans = merge(src_file_mosaic, bounds=bounds)
    # update
    out_meta = src.meta.copy()
    out_meta.update({'driver': 'GTiff', 'height': mosaic.shape[1], 'width': mosaic.shape[2], 'transform': out_trans})
    # save the merged
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mosaic)

# %%
bounds=(794000, 550000, 805000, 558500)
img_path_MS = [path for path in glob.glob('dataset_20181227/*MS.tif')]
img_path_MS_SR = [path for path in glob.glob('dataset_20181227/*MS_SR.tif')]

merge_tiles(img_path_MS, 'plantation_27122018_MS.tif', bounds)
merge_tiles(img_path_MS_SR, 'plantation_27122018_MS_SR.tif', bounds)
