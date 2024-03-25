import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from unet_model import UNet
from config import config
import os

def inference(threshold=0.5):
    
    if not os.path.isdir(config.result_folder_path):
        os.mkdir(config.result_folder_path)

    if not os.path.isdir(config.combined_image_mask):
        os.mkdir(config.combined_image_mask)

    if not os.path.isdir(config.inference_out_images_path):
        os.mkdir(config.inference_out_images_path)

    if not os.path.isdir(config.inference_out_masks_path):
        os.mkdir(config.inference_out_masks_path)

    if not os.path.isdir(config.inference_images_path):
        print(f"No {config.inference_images_path} is Found")
        return 0

    transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((config.image_size, config.image_size))
        ])

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet()
    model.to(device)
    checkpoint_path = config.training_weights_path.joinpath(config.best_weights)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        return "No Weights are Found"

    model.eval()
    # print(config.inference_images_path)
    images = config.inference_images_path.glob("*.png")
    with torch.no_grad():
        for image_name in images:
            image = Image.open((image_name))            
            image1 = transformation(image)
            image1 = TF.to_tensor(image1)
            image1 = image1.unsqueeze(dim=0)
            image1 = image1.to(device)
            model.to(device)
            pred_mask = model(image1)
            pred_mask = (pred_mask > threshold)
            pred_mask = pred_mask.squeeze().detach().cpu()
            pred_mask = pred_mask.numpy()
            image_name = str(image_name).split('/')[-1]

            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')
            
            plt.suptitle(f'Image: {image_name}')
            plt.savefig(config.combined_image_mask.joinpath(image_name))
            # plt.show()
        

if __name__ == "__main__":
    inference(0.5)