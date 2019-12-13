import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import rasterio.plot as rioplot

# plot img and shapefile
def show_image(img, transform, polygons=[], ax=None, band_idx=[0], shp_dict={'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'black'}, cmap='Greys'):
    """  """
    if len(img.shape) == 3:
        rioplot.show(rioplot.reshape_as_raster(img[:,:,band_idx]), transform=transform, ax=ax, cmap=cmap)
    elif len(img.shape) == 2:
        rioplot.show(img, transform=transform, ax=ax, cmap=cmap)
    else:
        raise 'Wrong image dimension. Must be HxW or HxWxB'
    for p in polygons:
        ax.add_patch(matplotlib.patches.Polygon(p, **shp_dict))
