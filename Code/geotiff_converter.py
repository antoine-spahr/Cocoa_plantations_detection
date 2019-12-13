import glob

from utils import to_single_geotiff

# get the folder names
path_to_folders = '../Data/images/originals/*'
folder_list = [f for f in glob.glob(path_to_folders)]

# save to single geotiff
for folder in folder_list:
    folder_name = folder.split('/')[-1]
    path_in = folder+'/*.tiff'
    path_out = '../Data/images/geotiffs/'+folder_name+'.tiff'
    to_single_geotiff(path_in, path_out)
    print(f'Geotiff saved at {path_out}')
