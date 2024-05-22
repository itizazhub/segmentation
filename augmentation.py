#import libraries
import os
import random
import shutil
import cv2
from PIL import Image
import numpy as np
from config import config
from sklearn.model_selection import train_test_split
from PIL import Image
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, Transpose, Compose
)

#get all images and masks
images = [i for i in config.images_path.glob("*.png")]
masks  = [i for i in config.masks_path.glob("*.png")]

#split images and masks into train(90%) and test(10%)
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.1, random_state=config.random_state)

#declare paths
test_images_folder_path = '/content/segmentation/test_images'
test_masks_folder_path =  '/content/segmentation/test_masks'
augmented_images_folder_path = '/content/segmentation/augmented_images'
augmented_masks_folder_path =  '/content/segmentation/augmented_masks'

#make folders
os.makedirs(test_images_folder_path, exist_ok=True)
os.makedirs(test_masks_folder_path, exist_ok=True)
os.makedirs(augmented_images_folder_path, exist_ok=True)
os.makedirs(augmented_masks_folder_path, exist_ok=True)

#move test(10%) images and masks to test_images and test_masks folders
for idx in range(len(test_images)):
  shutil.copy(test_images[idx], test_images_folder_path)
  shutil.copy(test_masks[idx], test_masks_folder_path)

#do augmentation and save images and masks to augmented_images and augmented_masks
horizentalfilp = Compose([HorizontalFlip(p=1)])
verticallip = Compose([VerticalFlip(p=1)])
randomrotate90 = Compose([ RandomRotate90(p=1)])
transpose = Compose([Transpose(p=1)])

i=0
for idx in range(len(train_images)):

  #load image and mask
  image = np.array(Image.open(train_images[idx]).convert('L'))
  mask = np.array(Image.open(train_masks[idx]).convert('L'))

  #apply augmentation to image and mask
  augmented1 = horizentalfilp(image=image, mask=mask)
  augmented2 = verticallip(image=image, mask=mask)
  augmented3 = randomrotate90(image=image, mask=mask)
  augmented4 = transpose(image=image, mask=mask)

  #save all
  Image.fromarray(image).save(augmented_images_folder_path + "/" + str(i) + ".png")
  Image.fromarray(mask).save(augmented_masks_folder_path + "/"  + str(i) + ".png")
  i=i+1
  
  Image.fromarray(augmented1['image']).save(augmented_images_folder_path + "/"  + str(i) + ".png")
  Image.fromarray(augmented1['mask']).save(augmented_masks_folder_path + "/"  + str(i) + ".png")
  i=i+1

  Image.fromarray(augmented2['image']).save(augmented_images_folder_path + "/"  + str(i) + ".png")
  Image.fromarray(augmented2['mask']).save(augmented_masks_folder_path + "/"  + str(i) + ".png")
  i=i+1

  Image.fromarray(augmented3['image']).save(augmented_images_folder_path + "/"  + str(i) + ".png")
  Image.fromarray(augmented3['mask']).save(augmented_masks_folder_path + "/"  + str(i) + ".png")
  i=i+1

  Image.fromarray(augmented4['image']).save(augmented_images_folder_path + "/"  + str(i) + ".png")
  Image.fromarray(augmented4['mask']).save(augmented_masks_folder_path + "/"  + str(i) + ".png")
  i=i+1

