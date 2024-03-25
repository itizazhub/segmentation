from config import config
import numpy as np
np.bool = np.bool_
import pandas as pd
from skimage.measure import label, regionprops
from PIL import Image
from skimage.feature import greycomatrix, greycoprops
import warnings
warnings.filterwarnings('ignore')
import math
# %matplotlib inline

def extract_features(mask, result):
    pixel_spacing_x, pixel_spacing_y = (0.5, 0.5)
    
    # Compute area
    label_img = label(mask, connectivity=mask.ndim)
    props = regionprops(label_img)
    area = props[0].area * pixel_spacing_x * pixel_spacing_y

    # Calculate the radius
    radius = round(math.sqrt(area / math.pi), 4)

    # Calculate the diameter
    diameter = round(2 * radius, 4)

    print("Diameter:", diameter, "mm")

    # Compute perimeter
    perimeter = props[0].perimeter * (pixel_spacing_x + pixel_spacing_y) / 2

    # Compute other shape features
    # Compute circularity
    circularity = round((4 * np.pi * area) / (perimeter ** 2) + 1e-100, 4)

    # Compute eccentricity
    eccentricity = round(props[0].eccentricity, 4)

    # Compute GLCM
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(result, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Compute GLCM properties
    try:
        contrast = round(greycoprops(glcm, 'contrast').mean(), 4)
        energy = round(greycoprops(glcm, 'energy').mean(), 4)
        homogeneity = round(greycoprops(glcm, 'homogeneity').mean(), 4)
    except ValueError:
        contrast, energy, homogeneity = np.nan, np.nan, np.nan
    
    # Combine texture features
    texture_features = round(np.mean((contrast, energy, homogeneity)), 4)

    return round(perimeter, 4), round(area, 4) , circularity, eccentricity, texture_features, radius, diameter

if __name__ == "__main__":
    masks_paths = config.masks_path.glob("*.png")
    images_paths = config.images_path.glob("*.png")

    feature_dict = {'perimeter' : [], 'area' : [], 'radius': [], 'diameter':[], 'circularity' : [], 'eccentricity' : [], 'texture_features' : []}
    
    for (image_path, mask_path) in zip(images_paths, masks_paths):
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
            image = np.array(Image.open(image_path).convert('L'))
            result = mask * image

            perimeter, area, circularity,\
            eccentricity, texture_features, radius, diameter = extract_features(mask, result)

            feature_dict['perimeter'].append(perimeter)
            feature_dict['area'].append(area)
            feature_dict['radius'].append(radius)
            feature_dict['diameter'].append(diameter)
            feature_dict['circularity'].append(circularity)
            feature_dict['eccentricity'].append(eccentricity)
            feature_dict['texture_features'].append(texture_features)
        except RuntimeError as e:
            print("something bad happened")
            print(e)

    features_dataset = pd.DataFrame(feature_dict)
    features_dataset.to_csv(config.result_folder_path.joinpath('features_dataset.csv'), index=False)
