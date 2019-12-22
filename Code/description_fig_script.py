import rasterio
import rasterio.plot as rioplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage
import pyproj

from utils import load_geotiff, load_shapefile, load_target_shp, human_format, contrast_adjust
from visualization import show_image

# %% Load the images
img_path = '../Data/images/geotiffs/'

# whole park
band, meta_whole = load_geotiff(img_path+'2018_12_29_Tai_whole.tiff', window=rasterio.windows.Window(30, 30, 1080, 2050))
img_whole = contrast_adjust(band, (0.1,99.9))

# north-west subpart
band, meta_NW = load_geotiff(img_path+'2018_12_29_Tai_NW.tiff', window=rasterio.windows.Window(10, 10, 1080, 2050))
img_NW = contrast_adjust(band, (0.5,99.5))
nw_bounds = rasterio.transform.array_bounds(meta_NW['height'], meta_NW['width'], meta_NW['transform'])

# nearby subsubpart
band, meta_nearby = load_geotiff(img_path+'2018_12_29_Tai_bigger.tiff', window=rasterio.windows.Window(1300, 3600, 1000, 1000))
img_nearby = contrast_adjust(band, (0.5,99.5))
near_bounds = rasterio.transform.array_bounds(meta_nearby['height'], meta_nearby['width'], meta_nearby['transform'])
# train image :
band, meta_train = load_geotiff(img_path+'2018_12_29_Tai_bigger.tiff', window=rasterio.windows.Window(1490, 4020, 530, 350))
img_train = contrast_adjust(band, (0.5,99.5))
train_bounds = rasterio.transform.array_bounds(meta_train['height'], meta_train['width'], meta_train['transform'])

# distant-subsubpart
band, meta_distant = load_geotiff(img_path+'2018_12_29_Tai_distant_plantations.tiff', window=rasterio.windows.Window(1800, 1500, 1000, 1000))
img_distant = contrast_adjust(band, (0.5,99.5))
dist_bounds = rasterio.transform.array_bounds(meta_distant['height'], meta_distant['width'], meta_distant['transform'])

# %% Load shapefile
shp_path = '../Data/shapefiles/'
polygons_Tai = load_shapefile(shp_path+'Tai_boundaries/WDPA_Oct2019_protected_area_721-shapefile-polygons.shp', projection=pyproj.Proj(meta_nearby['crs']))
polygons_Nzo = load_shapefile(shp_path+'NZo/WDPA_Jan2020_protected_area_2293-shapefile-polygons.shp', projection=pyproj.Proj(meta_nearby['crs']))
polygons_train, _, _ = load_target_shp(shp_path+'labels/Tai/segmentation.shp', transform=meta_nearby['transform'], projection=pyproj.Proj(meta_nearby['crs']))
polygons_test, _, _ = load_target_shp(shp_path+'labels/control/segmentation_control.shp', transform=meta_nearby['transform'], projection=pyproj.Proj(meta_nearby['crs']))
polygons_test_dist, _, _ = load_target_shp(shp_path+'labels/distant/Segmentation_distant2.shp', transform=meta_distant['transform'], projection=pyproj.Proj(meta_distant['crs']))
polygons_Tai += polygons_Nzo

# %% plot images
title_fs = 10
zone_color = 'darkgray'
zone_lw = 1.3
line_color = zone_color
line_lw = zone_lw
park_color = 'orange'
train_color = 'orangered'
test_color = 'deepskyblue'

fig = plt.figure(figsize=(13,7))
gs = fig.add_gridspec(2,3, hspace=0.25, wspace=0.0)

# whole image
ax_whole = fig.add_subplot(gs[:, 0])
ax_whole.set_title('Taï National Park', fontsize=title_fs)
show_image(img_whole, meta_whole['transform'], ax=ax_whole, band_idx=[2,1,0])
for p in polygons_Tai:
    ax_whole.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'orange'}))
ax_whole.add_patch(matplotlib.patches.Rectangle(nw_bounds[0:2], width=np.abs(nw_bounds[0]-nw_bounds[2]), height=np.abs(nw_bounds[1]-nw_bounds[3]), linewidth=zone_lw, facecolor=(0,0,0,0), edgecolor=zone_color))

# NW image
ax_nw = fig.add_subplot(gs[:, 1])
ax_nw.set_title('North-West part of Park', fontsize=title_fs)
show_image(img_NW, meta_NW['transform'], ax=ax_nw, band_idx=[2,1,0])
for p in polygons_Tai:
    ax_nw.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'orange'}))
ax_nw.add_patch(matplotlib.patches.Rectangle(near_bounds[0:2], width=np.abs(near_bounds[0]-near_bounds[2]), height=np.abs(near_bounds[1]-near_bounds[3]), linewidth=zone_lw, facecolor=(0,0,0,0), edgecolor=zone_color))
ax_nw.add_patch(matplotlib.patches.Rectangle(dist_bounds[0:2], width=np.abs(dist_bounds[0]-dist_bounds[2]), height=np.abs(dist_bounds[1]-dist_bounds[3]), linewidth=zone_lw, facecolor=(0,0,0,0), edgecolor=zone_color))

# line between whole and NW
ax_nw.add_artist(matplotlib.patches.ConnectionPatch(xyA=(nw_bounds[0],nw_bounds[1]), xyB=(nw_bounds[2],nw_bounds[1]), \
                                                    coordsA="data", coordsB="data", axesA=ax_nw, axesB=ax_whole, color=line_color, lw=line_lw))
ax_nw.add_artist(matplotlib.patches.ConnectionPatch(xyA=(nw_bounds[0],nw_bounds[3]), xyB=(nw_bounds[2],nw_bounds[3]), \
                                                    coordsA="data", coordsB="data", axesA=ax_nw, axesB=ax_whole, color=line_color, lw=line_lw))

# distant image
ax_dist = fig.add_subplot(gs[0, 2])
ax_dist.set_title('Distant Testing Area', fontsize=title_fs)
show_image(img_distant, meta_distant['transform'], ax=ax_dist, band_idx=[3,2,1])

for p in polygons_test_dist:
    ax_dist.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':0.75, 'facecolor':(0,0,0,0), 'edgecolor':test_color}))
for p in polygons_Tai:
    ax_dist.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':'orange'}))

# line between NW and distant
ax_dist.add_artist(matplotlib.patches.ConnectionPatch(xyA=(dist_bounds[0],dist_bounds[1]), xyB=(dist_bounds[2],dist_bounds[1]), \
                                                    coordsA="data", coordsB="data", axesA=ax_dist, axesB=ax_nw, color=line_color, lw=line_lw))
ax_dist.add_artist(matplotlib.patches.ConnectionPatch(xyA=(dist_bounds[0],dist_bounds[3]), xyB=(dist_bounds[2],dist_bounds[3]), \
                                                    coordsA="data", coordsB="data", axesA=ax_dist, axesB=ax_nw, color=line_color, lw=line_lw))

# nearby image
ax_near = fig.add_subplot(gs[1, 2])
ax_near.set_title('Nearby Testing and Training Area', fontsize=title_fs)
show_image(img_nearby, meta_nearby['transform'], ax=ax_near, band_idx=[3,2,1])
for p in polygons_train:
    ax_near.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':0.75, 'facecolor':(0,0,0,0), 'edgecolor':train_color}))
for p in polygons_test:
    ax_near.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':0.75, 'facecolor':(0,0,0,0), 'edgecolor':test_color}))
for p in polygons_Tai:
    ax_near.add_patch(matplotlib.patches.Polygon(p, **{'linewidth':1, 'facecolor':(0,0,0,0), 'edgecolor':park_color}))
ax_near.add_patch(matplotlib.patches.Rectangle(train_bounds[0:2], width=np.abs(train_bounds[0]-train_bounds[2]), height=np.abs(train_bounds[1]-train_bounds[3]), linewidth=zone_lw, facecolor=(0,0,0,0), edgecolor=zone_color))
ax_near.text(train_bounds[2]-2500, train_bounds[3]+200, 'Training area', fontsize=8, color=zone_color)

# line between NW and near
ax_near.add_artist(matplotlib.patches.ConnectionPatch(xyA=(near_bounds[0],near_bounds[1]), xyB=(near_bounds[2],near_bounds[1]), \
                                                    coordsA="data", coordsB="data", axesA=ax_near, axesB=ax_nw, color=line_color, lw=line_lw))
ax_near.add_artist(matplotlib.patches.ConnectionPatch(xyA=(near_bounds[0],near_bounds[3]), xyB=(near_bounds[2],near_bounds[3]), \
                                                    coordsA="data", coordsB="data", axesA=ax_near, axesB=ax_nw, color=line_color, lw=line_lw))

axs = [ax_whole, ax_nw, ax_dist, ax_near]
for i, ax in enumerate(axs):
    if i!=2 : ax.set_xlabel('Easting [m]', fontsize=8)
    if i==0 : ax.set_ylabel('Northing [m]', fontsize=8)
    ax.tick_params(axis='y', which='major', labelsize=7.5, colors='gray')
    ax.tick_params(axis='x', which='major', labelsize=7.5, colors='gray')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))

# legend
handles = [matplotlib.lines.Line2D([0], [0], color=c, linewidth=2) for c in [park_color, train_color, test_color]]
labels = ["Taï National Park and N'Zo boundaries", "Training cocoa plantations", "Testing cocoa plantations"]
lgd = fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), bbox_transform=fig.transFigure, ncol=4, fontsize=9)

fig.savefig('../Figures/overview.png', dpi=100, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
