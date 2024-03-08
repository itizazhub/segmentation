from config import config
from pathlib import Path
import numpy as np
from zipfile import ZipFile
import matplotlib.image as mpimg
from tqdm import tqdm
import h5py
import os


def makedir(path:Path):
    if not (os.path.isdir(path)):
        print(f'Creating {path} folder')
        os.mkdir(path)
    else:
        print(path, " is already available")  

def unzip(zipped_file_path:Path, destination_folder_path:Path):
     if not (os.path.exists(zipped_file_path)):
        print(f'The zipped_file_path is not Available : {zipped_file_path}')
        return
     if not (os.path.isdir(destination_folder_path)):
        print(f'The destination_folder_path is not Available : {destination_folder_path}')
        return
     
     with ZipFile(zipped_file_path, 'r') as zipfile:
        print(f'\tExtracting files of {zipped_file_path}')
        zipfile.extractall(destination_folder_path)
        print(f'\tDone with {zipped_file_path}')
     
def convert_matfiles_to_images(mat_files_path:Path, images_destination_path:Path, masks_destination_path:Path):
    mat_files_names = [str(file).split('/')[-1].split('.')[0] for file in mat_files_path.glob('*.mat')] # changes '\\' to '/'
    for filename in tqdm(mat_files_names):
        image_path = images_destination_path.joinpath(filename + '.png')
        mask_path = masks_destination_path.joinpath(filename + '_mask.png')
        mat_file_path = mat_files_path.joinpath(filename + '.mat')
        # print(filename)
        # print(image_path)
        # print(mask_path)
        # print(mat_file_path)
        with h5py.File(mat_file_path, 'r') as mat_file:
            cjdata_group = mat_file['cjdata']
            label = cjdata_group['label'][0][0]
            image = np.array(cjdata_group['image'])
            mask = np.array(cjdata_group['tumorMask'])
            
            if label == 3: # only save pituitary tumor
                mpimg.imsave(image_path, image, cmap='gray', format='png')
                mpimg.imsave(mask_path,mask, cmap='gray', format='png')
            else:
                continue
    images = [i for i in images_destination_path.glob('*.png')]
    print(f"Total of {len(images)} Images are converted from .mat files")


if __name__ == '__main__':
    makedir(config.unzipped_dataset)
    unzip(config.zipped_dataset_path, config.unzipped_dataset)

    file_names = [file_name for file_name in config.unzipped_dataset.glob('*.zip')]
    makedir(config.mat_files_path)
    for zipped_file_path in tqdm(file_names):
        unzip(zipped_file_path, config.mat_files_path)

    makedir(config.images_path)
    makedir(config.masks_path)
    convert_matfiles_to_images(config.mat_files_path, config.images_path, config.masks_path)