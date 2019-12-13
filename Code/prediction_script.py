import rasterio
import rasterio.features
import rasterio.plot as rioplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage
from skimage.morphology import disk
import pyproj
import pickle
import fiona

from utils import load_geotiff, load_shapefile, load_target_shp, human_format, contrast_adjust
from processing import preprocessing
from visualization import show_image

# paths
path_geotiff = '../Data/images/geotiffs/'
path_shp = '../Data/shapefiles/'
image_name_test = 'geotiffs/2018_12_29_Tai_bigger.tiff'

# %% -------------------------------------------------------------------------------------------------------------
# load whole park image for visualization
band, meta_whole = load_geotiff(path_geotiff+'2018_12_29_Tai_whole.tiff', window=rasterio.windows.Window(30, 30, 1080, 2050))
img_whole = contrast_adjust(band, (0.1,99.9))

# %% -------------------------------------------------------------------------------------------------------------
# load part sample from large images
h, w = 1000, 1000
windows = [rasterio.windows.Window(1000, 2000, w, h), \
           rasterio.windows.Window(1000, 1000, w, h), \
           rasterio.windows.Window(1200, 100, w, h), \
           rasterio.windows.Window(1000, 2500, w, h), \
           rasterio.windows.Window(1000, 1100, w, h), \
           rasterio.windows.Window(600, 2000, w, h), \
           rasterio.windows.Window(1100, 600, w, h), \
           rasterio.windows.Window(300, 1000, w, h), \
           rasterio.windows.Window(500, 900, w, h), \
           rasterio.windows.Window(1500, 2500, w, h)]

filenames = ['2018_12_29_Tai_bigger.tiff', \
             '2018_12_29_Tai_bigger.tiff', \
             '2018_12_29_Tai_bigger.tiff', \
             '2018_12_29_Tai_distant_plantations.tiff', \
             '2018_12_29_Tai_NE.tiff', \
             '2018_12_29_Tai_NE.tiff', \
             '2018_12_29_Tai_SE.tiff', \
             '2018_12_29_Tai_S.tiff', \
             '2018_12_29_Tai_SW.tiff', \
             '2018_12_29_Tai_SW.tiff']

img = []
img_adj = []
metas = []
bounds = []

for window, filename in zip(windows, filenames):
    print(f'>>> loading {window} from {filename}')
    band, meta = load_geotiff(path_geotiff+filename, window=window)
    img_adj.append(contrast_adjust(band, (0.5, 99.5)))
    img.append(np.stack(band, axis=2))
    metas.append(meta)
    bounds.append(rasterio.transform.array_bounds(meta['height'], meta['width'], meta['transform']))

# %% -------------------------------------------------------------------------------------------------------------
# preprocess images
img = [preprocessing(im) for im in img]

# %% -------------------------------------------------------------------------------------------------------------
# reshape in 2D
X = [im.reshape(-1,im.shape[2]) for im in img]

# %% -------------------------------------------------------------------------------------------------------------
# Load the fitted KNN models
with open('../models/KNN_trained.pickle', 'rb') as src:
    fitted_model = pickle.load(src)

# %% -------------------------------------------------------------------------------------------------------------
# Predict all sample and post-process them
preds = []
for i, Xi in enumerate(X):
    print(f'>>> Detecting cocoa plantations on image {i+1}')
    p = fitted_model.predict(Xi).reshape((h,w))
    p = skimage.morphology.binary_closing(p, selem=disk(2))
    preds.append(skimage.morphology.binary_opening(p, selem=disk(2)))

# %% -------------------------------------------------------------------------------------------------------------
# Polygonize predictions
pred_poly = []
for i, (pred, meta) in enumerate(zip(preds, metas)):
    print(f'>>> Polygonizing the prediction {i+1}')
    shapes = rasterio.features.shapes(pred.astype('int16'), transform=meta['transform'])
    pred_poly.append([shape['coordinates'][0] for shape, val in shapes if val == 1])

# %% -------------------------------------------------------------------------------------------------------------
# Visualize results
fig = plt.figure(figsize=(12,16))
gs = plt.GridSpec(4, 3, wspace=0.2, hspace=0.2)
title_fs = 11
pred_fc = 'crimson'
pred_ec = None
pred_alpha = 1

# overview img
ax_overview = fig.add_subplot(gs[0:2,0])
ax_overview.set_title('Taï National Park', fontsize=title_fs)
show_image(img_whole, meta_whole['transform'], ax=ax_overview, band_idx=[2,1,0])
for i, bound in enumerate(bounds):
    ax_overview.add_patch(matplotlib.patches.Rectangle(bound[0:2], \
                                                       width=np.abs(bound[0] - bound[2]), \
                                                       height=np.abs(bound[1] - bound[3]), \
                                                       linewidth=1, facecolor=(0,0,0,0), edgecolor='darkgray'))
    ax_overview.text(bound[2]-6*w, bound[3]-6*h, str(i+1), fontsize=9, color='crimson', fontweight='bold')

# create axes for predictions
axs = []
for i in range(3*4):
    if not i in [0,3]: axs.append(fig.add_subplot(gs[i]))

# add image + polygon predictions
for i, (im, meta, poly, ax) in enumerate(zip(img_adj, metas, pred_poly, axs)):
    ax.set_title(f'Prediction {i+1}', fontsize=title_fs)
    show_image(im, meta['transform'], ax=ax, band_idx=[3,2,1])
    for p in poly:
        ax.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':1, 'facecolor':pred_fc, 'edgecolor':pred_ec, 'alpha':pred_alpha}))

# pimp axis labels
for i, ax in enumerate([ax_overview]+axs):
    if i in [8,9,10]: ax.set_xlabel('Easting [m]', fontsize=8)
    if i in [0,5,8]: ax.set_ylabel('Northing [m]', fontsize=8)
    ax.tick_params(axis='y', which='major', labelsize=7.5, colors='gray')
    ax.tick_params(axis='x', which='major', labelsize=7.5, colors='gray')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))

# legend
handles = [matplotlib.patches.Patch(linewidth=0.75, facecolor=pred_fc, edgecolor=pred_ec, alpha=pred_alpha)]
labels = ['KNN Prediction']
lgd = fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.07), bbox_transform=fig.transFigure, ncol=1, fontsize=9)

fig.savefig('../Figures/predictions.png', dpi=100, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
