from config import config
import numpy as np
np.bool = np.bool_
import pandas as pd
from skimage import measure
from skimage.measure import label, regionprops
from PIL import Image
from skimage.feature.texture import greycomatrix, greycoprops
# %matplotlib inline

def extract_features(mask, result, pixel_spacing):
    # Compute perimeter
    label_img = label(mask, connectivity=mask.ndim)
    props = regionprops(label_img)
    perimeter_ = props[0].perimeter
    # perimeter_ = np.sum(perimeter(mask, neighbourhood = 4))

    # Compute other shape features
    # Compute area
    area = props[0].area

    # Compute convex area
    convex_area = measure.label(mask, background=0)

    # Compute solidity
    solidity = area / convex_area.max()

    # Compute circularity
    circularity = (4 * np.pi * area) / (perimeter_ ** 2) + 0.0000001

    # Compute eccentricity
    props = measure.regionprops(mask.astype(int))
    eccentricity = props[0].eccentricity

    # Calculate tumor area in pixels
    # area = np.sum(mask)

    # Calculate tumor area in square millimeters
    pixel_spacing_x, pixel_spacing_y = pixel_spacing
    tumor_area_mm2 = area * pixel_spacing_x * pixel_spacing_y

    # Calculate equivalent diameter (optional)
    tumor_diameter_mm = np.sqrt(tumor_area_mm2 / np.pi)

        # Compute GLCM
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(result, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Compute GLCM properties
    contrast = greycoprops(glcm, 'contrast').flatten()
    energy = greycoprops(glcm, 'energy').flatten()
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()
    
    # Combine texture features
    texture_features = np.concatenate((contrast, energy, homogeneity))

    # Return computed shape features
    return perimeter_, area, solidity, circularity, eccentricity, tumor_area_mm2, tumor_diameter_mm, texture_features


if __name__ == "__main__":
    images_paths = config.inference_images_path.glob("*.png")
    masks_paths = config.inference_out_images_path.glob("*.png")

    feature_dict = {'perimeter_' : [], 'area' : [], 'solidity' : [], 'circularity' : [], 'eccentricity' : [], 'tumor_area_mm2' : [], 'tumor_diameter_mm' : [], 'texture_features' : []}

    for image_path, mask_path in zip(images_paths, masks_paths):    
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
            image = np.array(Image.open(image_path).convert('L'))
            result = mask * image
            pixel_spacing = (0.938, 0.938)

            perimeter_, area, solidity, circularity,\
            eccentricity, tumor_area_mm2, tumor_diameter_mm,\
            texture_features = extract_features(mask, result, pixel_spacing)

            feature_dict['perimeter_'].append(perimeter_), feature_dict['area'].append(area),\
            feature_dict['solidity'].append(solidity),feature_dict['circularity'].append(circularity),\
            feature_dict['eccentricity'].append(eccentricity),feature_dict['tumor_area_mm2'].append(tumor_area_mm2),\
            feature_dict['tumor_diameter_mm'].append(tumor_diameter_mm),feature_dict['texture_features'].append(texture_features)
        except:
            print("something bad happened")

    features_dataset = pd.DataFrame(feature_dict)
    features_dataset.to_csv(config.result_folder_path.joinpath('features_dataset.csv'), index=False)
