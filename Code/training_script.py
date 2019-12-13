import rasterio
import rasterio.plot as rioplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import skimage
import pyproj
import pickle

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from utils import load_geotiff, load_shapefile, load_target_shp, compute_mask, scores_summary, human_format
from processing import preprocessing
from visualization import show_image

# ---------------------------------------------------------------------------------------------------------------
# %% Paths
path_geotiff = '../Data/images/'
path_shp = '../Data/shapefiles/'
image_name_train = 'geotiffs/2018_12_29_Tai_bigger.tiff'
label_path_train = 'labels/Tai/segmentation.shp'

# ---------------------------------------------------------------------------------------------------------------
# %% load plantation image
band, meta_train = load_geotiff(path_geotiff+image_name_train, window=rasterio.windows.Window(1490, 4020, 530, 350))

# ---------------------------------------------------------------------------------------------------------------
# %% Generate new features
img_train = preprocessing(np.stack(band, axis=2))

# %% vizualize new features
Nband_kept = 10-1
fig, axs = plt.subplots(4,4, figsize=(12,10), gridspec_kw={'hspace':0.1, 'wspace':0.1})
show_image(img_train[:,:,Nband_kept+1], meta_train['transform'], ax=axs[0][0], cmap='PiYG')
axs[0][0].set_title('NDVI', fontsize=10)
axs[0][0].set_ylabel('Northing [m]')
axs[0][0].set_xticklabels([])
show_image(img_train[:,:,Nband_kept+2], meta_train['transform'], ax=axs[0][1], cmap='PiYG')
axs[0][1].set_title('green NDVI', fontsize=10)
axs[0][1].set_xticklabels([])
axs[0][1].set_yticklabels([])

titles_part1 = ['entropy: ', 'Opening reconstruction: ', 'closing reconstruction: ', 'LBP: ']
titles_part2 = ['R', 'NIR', 'NDVI']
for i, ax in enumerate(axs[1:,:]):
    for j in range(4):
        show_image(img_train[:,:,(Nband_kept+3)+i+j*3], meta_train['transform'], ax=ax[j], cmap='RdBu')
        ax[j].set_title(titles_part1[j]+titles_part2[i], fontsize=10)
        if j==0:
            ax[j].set_ylabel('Northing [m]')
        else:
            ax[j].set_yticklabels([])
        if i==2:
            ax[j].set_xlabel('Easting [m]')
        else:
            ax[j].set_xticklabels([])

for ax_v, ax_h in zip(axs[:,0].reshape(-1), axs[-1,:].reshape(-1)):
    ax_v.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
    ax_h.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))

axs[0][2].set_axis_off()
axs[0][3].set_axis_off()
fig.savefig('../Figures/Feature_Engineered.png', dpi=100, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------------------------------------------
# %% load label shapefile
polygons_train, polygons_rc_train, class_list = load_target_shp(path_shp+label_path_train, transform=meta_train['transform'], projection=pyproj.Proj(meta_train['crs']))
mask_train = compute_mask(polygons_rc_train, meta_train['width'], meta_train['height'], np.ones(len(class_list)))

# %% get the labels and data in 2D matrix shape (sample x features)
X = img_train.reshape(-1,img_train.shape[2])
y = mask_train.reshape(-1)

# ---------------------------------------------------------------------------------------------------------------
# %% define the models
scaller = MinMaxScaler(feature_range=(0, 1))
pca = PCA(n_components=X.shape[1])
models = [KNeighborsClassifier(n_neighbors=5), \
          LinearSVC(penalty='l2', C=1000), \
          RandomForestClassifier(n_estimators=150, max_depth=20), \
          MLPClassifier(hidden_layer_sizes=(100), activation='relu', solver='adam', max_iter=200)]
names = ['KNN', 'SVM', 'RF', 'MLP']
models_pipelines = [Pipeline(steps=[('scaller',scaller), ('pca',pca), (name, model)]) for name, model in zip(names, models)]

# ---------------------------------------------------------------------------------------------------------------
# %% Cross-validation to get an estimation of the performance of the models (mean +/- std)
kfold = 5
score_list = ['accuracy', 'f1', 'precision', 'recall']
return_train_score = True
scores = {}
for model, name in zip(models_pipelines, names):
     print(f'>>> training {name}')
     scores[name] = cross_validate(model, X, y, scoring=score_list, cv=kfold, return_train_score=return_train_score, n_jobs=-1)

# save model's performance
scores_cv = scores_summary(scores, score_list)
(scores_cv*100).to_csv('../scores/scoresCV.csv', float_format='%.2f%%')
scores_cv.style.format('{:.2%}')

# ---------------------------------------------------------------------------------------------------------------
# %% Build full model with all data
fitted_model = [model.fit(X,y) for model in models_pipelines]

# %% save models
for model, name in zip(fitted_model, names):
    with open('../models/'+name+'_trained.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
