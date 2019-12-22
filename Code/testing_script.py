import rasterio
import rasterio.plot as rioplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import skimage
from skimage.morphology import disk
import pyproj
import pickle

from utils import load_geotiff, load_shapefile, load_target_shp, compute_mask, scores_summary, human_format
from processing import preprocessing
from visualization import show_image

def detection_rate(mask, pred):
    """
    Compute the fraction of the polygon on mask present on the pred.
    INPUT : mask (2D numpy array) -> the mask of ground truth
            pred (2D numpy array) -> the mask of prediction
    OUTPUT : rate [float] -> the detection rate

    """
    polygon_mask = (mask == 1).ravel()
    t = mask.ravel()[polygon_mask]
    p = pred.ravel()[polygon_mask]

    return p.sum()/t.sum()

# %%
path_geotiff = '../Data/images/'
path_shp = '../Data/shapefiles/'
image_name_test = 'geotiffs/2018_12_29_Tai_bigger.tiff'
label_name_test = 'labels/control/segmentation_control.shp'
label_path_train = 'labels/Tai/segmentation.shp'
image_name_test_dist = 'geotiffs/2018_12_29_Tai_distant_plantations.tiff'
label_name_test_dist = 'labels/distant/Segmentation_distant2.shp'

# ---------------------------------------------------------------------------------------------------------------
# %% load plantation images
# Nearby window --> rasterio.windows.Window(1300, 3600, 1000, 1000)
band, meta_test = load_geotiff(path_geotiff+image_name_test, window=rasterio.windows.Window(1300, 3600, 1000, 1000))
img_test = np.stack(band, axis=2)

# Distant window --> rasterio.windows.Window(1800, 1500, 1000, 1000)
band, meta_test_dist = load_geotiff(path_geotiff+image_name_test_dist, window=rasterio.windows.Window(1800, 1500, 1000, 1000))
img_test_dist = np.stack(band, axis=2)

# ---------------------------------------------------------------------------------------------------------------
# %% preprocessing
img_test = preprocessing(img_test)
img_test_dist = preprocessing(img_test_dist)

# ---------------------------------------------------------------------------------------------------------------
# %% load training label polygons
_, meta_train = load_geotiff(path_geotiff+'geotiffs/2018_12_29_Tai_bigger.tiff', window=rasterio.windows.Window(1490, 4020, 530, 350))
polygons_train, polygons_rc_train, class_list = load_target_shp(path_shp+label_path_train, transform=meta_train['transform'], projection=pyproj.Proj(meta_train['crs']))

# %% get the test labels polygons
# Nearby
polygons_test, polygons_rc_test, class_list_test = load_target_shp(path_shp+label_name_test, transform=meta_test['transform'], projection=pyproj.Proj(meta_test['crs']))
mask_test = compute_mask(polygons_rc_test, meta_test['width'], meta_test['height'], np.ones(len(class_list)))

# distant
polygons_test_dist, polygons_rc_test_dist, class_list_test_dist = load_target_shp(path_shp+label_name_test_dist, transform=meta_test_dist['transform'], projection=pyproj.Proj(meta_test_dist['crs']))
mask_test_dist = compute_mask(polygons_rc_test_dist, meta_test_dist['width'], meta_test_dist['height'], np.ones(len(class_list)))

# %% get the park polygon
polygons_park = load_shapefile(path_shp+'Tai_boundaries/WDPA_Oct2019_protected_area_721-shapefile-polygons.shp', projection=pyproj.Proj(meta_test['crs']))
polygons_park += load_shapefile(path_shp+'NZo/WDPA_Jan2020_protected_area_2293-shapefile-polygons.shp', projection=pyproj.Proj(meta_nearby['crs']))

# ---------------------------------------------------------------------------------------------------------------
# %% Load the fitted models
fitted_model = []
names = ['KNN', 'SVM', 'RF', 'MLP']
for name in names:
    with open('../models/'+name+'_trained.pickle', 'rb') as src:
        fitted_model.append(pickle.load(src))

# ---------------------------------------------------------------------------------------------------------------
# %% get the matrix
# nearby
X_test = img_test.reshape(-1,img_test.shape[2])
y_test = mask_test.reshape(-1)

# distant
X_test_dist = img_test_dist.reshape(-1,img_test_dist.shape[2])
y_test_dist = mask_test_dist.reshape(-1)

# ---------------------------------------------------------------------------------------------------------------
# %% predict the image with each model
# nearby
pred = [model.predict(X_test) for model in fitted_model]
pred = [p.reshape(mask_test.shape) for p in pred]

# distant
pred_dist = [model.predict(X_test_dist) for model in fitted_model]
pred_dist = [p.reshape(mask_test_dist.shape) for p in pred_dist]

# ---------------------------------------------------------------------------------------------------------------
# %% Post processing : morphological treatment :
# nearby
pred_smoothed_int = [skimage.morphology.binary_closing(pred_mask, selem=disk(2)).astype(int) for pred_mask in pred]
pred_smoothed = [skimage.morphology.binary_opening(pred_mask, selem=disk(2)).astype(int) for pred_mask in pred_smoothed_int]

# distant
pred_smoothed_dist = [skimage.morphology.binary_closing(pred_mask, selem=disk(2)).astype(int) for pred_mask in pred_dist]
pred_smoothed_dist = [skimage.morphology.binary_opening(pred_mask, selem=disk(2)).astype(int) for pred_mask in pred_smoothed_dist]

# ---------------------------------------------------------------------------------------------------------------
# %% Compute the detection rate
mask_test = compute_mask(polygons_rc_test, meta_test['width'], meta_test['height'], np.ones(len(class_list_test)))
mask_test_dist = compute_mask(polygons_rc_test_dist, meta_test_dist['width'], meta_test_dist['height'], np.ones(len(class_list_test_dist)))

detection_rate_near = np.array([detection_rate(mask_test, pred) for pred in pred_smoothed])
detection_rate_dist = np.array([detection_rate(mask_test_dist, pred) for pred in pred_smoothed_dist])

# ---------------------------------------------------------------------------------------------------------------
# %% Post-processing figure
fig, axs = plt.subplots(1,3,figsize=(14,7))
titles = ['Raw KNN prediciton', 'Closing of prediction', 'Openning of closed prediction']
imgs = [pred[0], pred_smoothed_int[0], pred_smoothed[0]]

for ax, title, img, i in zip(axs.reshape(-1), titles, imgs, range(3)):
    show_image(img, meta_test['transform'], ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Easting [m]')
    if i == 0:
        ax.set_ylabel('Northing [m]')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
    else:
        ax.set_yticklabels([])
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))

fig.tight_layout()
fig.savefig('../Figures/PostProcessing_example.png', dpi=100, bbox_inches='tight')
plt.show()

# %% Prediction results
fig, axs = plt.subplots(4,3,figsize=(15,20), gridspec_kw={'hspace':0.25})

# NEARBY
plt.annotate('Nearby predictions', xy=(0.06, 0.735), xytext=(0.02, 0.735), xycoords='figure fraction',
            fontsize=14, ha='center', va='center', rotation=90,
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=19.5, lengthB=0.25', lw=2.0))

# True color with polygons
img_stretch = np.stack([skimage.exposure.rescale_intensity(img_test[:,:,i], in_range=tuple(np.percentile(img_test[:,:,i], (0.5, 99.5))), out_range=(0,1)) for i in range(img_test.shape[2])], axis=2)
show_image(img_stretch, meta_test['transform'], polygons=polygons_test, ax=axs[0][0], band_idx=[2,1,0], shp_dict={'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'deepskyblue'})
axs[0][0].set_title('True color')

for i, ax in enumerate(axs[:2,1:].reshape(-1)):
    show_image(pred_smoothed[i], meta_test['transform'], ax=ax, polygons=polygons_test, shp_dict={'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'deepskyblue'})
    ax.set_title(names[i] + ': Predicted plantations')
for ax in axs[:2,:].reshape(-1):
    for p in polygons_train:
        ax.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'orangered'}))
    for p in polygons_park:
        ax.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'orange'}))

# table
DR_near_str = [[f'{val:.3%}'] for val in detection_rate_near]
table = axs[1][0].table(cellText=DR_near_str, rowLabels=names, cellLoc='center', colLabels=['Nearby Detection Rate [%]'], \
                                     colColours=['gainsboro'], loc='center', colWidths=[0.8], edges='horizontal', \
                                     bbox=[0.2, 0.1, 0.6, 0.8])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1,1)

# DISTANT
plt.annotate('Distant predictions', xy=(0.06, 0.265), xytext=(0.02, 0.265), xycoords='figure fraction', \
            fontsize=14, ha='center', va='center', rotation=90, \
            bbox=dict(boxstyle='square', fc='white'), \
            arrowprops=dict(arrowstyle='-[, widthB=19.5, lengthB=0.25', lw=2.0))

# True color with polygons
img_stretch_dist = np.stack([skimage.exposure.rescale_intensity(img_test_dist[:,:,i], in_range=tuple(np.percentile(img_test_dist[:,:,i], (0.5, 99.5))), out_range=(0,1)) for i in range(img_test_dist.shape[2])], axis=2)
show_image(img_stretch_dist, meta_test_dist['transform'], polygons=polygons_test_dist, ax=axs[2][0], band_idx=[2,1,0], shp_dict={'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'deepskyblue'})
axs[2][0].set_title('True color')

for i, ax in enumerate(axs[2:,1:].reshape(-1)):
    show_image(pred_smoothed_dist[i], meta_test_dist['transform'], ax=ax, polygons=polygons_test_dist, shp_dict={'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'deepskyblue'})
    ax.set_title(names[i] + ': Predicted plantations')
for ax in axs[2:,:].reshape(-1):
    for p in polygons_park:
        ax.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'orange'}))

# table
DR_dist_str = [[f'{val:.3%}'] for val in detection_rate_dist]
table = axs[3][0].table(cellText=DR_dist_str, rowLabels=names, cellLoc='center', colLabels=['Distant Detection Rate [%]'], \
                                     colColours=['gainsboro'], loc='center', colWidths=[0.8], edges='horizontal', \
                                     bbox=[0.2, 0.1, 0.6, 0.8])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1,1)

# General
for ax in axs.reshape(-1):
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))

axs[1][0].set_axis_off()
axs[3][0].set_axis_off()

#fig.tight_layout()
fig.savefig('../Figures/Tai_testing_prediction.png', dpi=100, bbox_inches='tight')
plt.show()
