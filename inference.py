import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from unet_model import UNet
from pathlib import Path
from config import config
import os

def inference(threshold=0.5):
    """ Calculate the output mask on a single input data.
    Parameters:
        data(dict): Contains the index, image, mask torch.Tensor.
                    'index': Index of the image.
                    'image': Contains the tumor image torch.Tensor.
                    'mask' : Contains the mask image torch.Tensor.
        threshold(float): Threshold value after which value will be part of output.
                            Default: 0.5

    Returns:
        image(numpy.ndarray): 512x512 Original brain scanned image.
        mask(numpy.ndarray): 512x512 Original mask of scanned image.
        output(numpy.ndarray): 512x512 Generated mask of scanned image.
        score(float): Sørensen–Dice Coefficient for mask and output.
                        Calculates how similar are the two images.
    """

    if not os.path.isdir(config.result_folder_path):
        os.mkdir(config.result_folder_path)

    if not os.path.isdir(config.combined_image_mask):
        os.mkdir(config.combined_image_mask)
    
    if not os.path.isdir(config.inference_out_images_path):
        os.mkdir(config.inference_out_images_path)

    if not os.path.isdir(config.inference_out_masks_path):
        os.mkdir(config.inference_out_masks_path)
    
    if not os.path.isdir(config.inference_images_path):
        return f"No {config.inference_images_path} is Found"

    transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((config.image_size, config.image_size))
        ])


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet()
    model.to(device)
    checkpoint_path = config.model_weights_path.joinpath("best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    # checkpoint_path = config.pre_trained_model_path.joinpath("weights.pt")
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint)
    else:
        return "No Weights are Found"

    model.eval()
    # print(config.inference_images_path)
    files = (config.images_path).glob("*.png")
    with torch.no_grad():
        for image_name in files:
            # print(image_name)
            image = Image.open((image_name))
            image = transformation(image)
            image = TF.to_tensor(image)
            image = image.unsqueeze(dim=0)
            image = image.to(device)
            model.to(device)
            pred_mask = model(image)
            pred_mask = (pred_mask > threshold)
            pred_mask = pred_mask.squeeze().detach().cpu()
            image = image.squeeze().detach().cpu()
            pred_mask = pred_mask.numpy()
            mask = Image.fromarray(pred_mask)
            # print(image.shape)
            image_name = str(image_name).split('\\')[-1]
            img = TF.to_pil_image(image)
            img = img.convert('L')
            img.save(config.inference_out_images_path.joinpath(image_name))
            mask.save(config.inference_out_masks_path.joinpath("mask"+image_name))

            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            # Plotting predicted mask
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask, cmap='gray')  # Assuming pred_mask is single-channel
            plt.title('Predicted Mask')
            plt.axis('off')
            
            plt.suptitle(f'Image: {image_name}')
            plt.savefig(config.combined_image_mask.joinpath(image_name))
            # plt.show()
        

    # image = data['image'].numpy()
    # mask = data['mask'].numpy()

    # image_tensor = torch.Tensor(data['image'])
    # image_tensor = image_tensor.view((-1, 1, 512, 512)).to(device)
    # output = model(image_tensor) #.detach().cpu()
    # output = (output > threshold)
    # # output = output.numpy()

    # # image = np.resize(image, (512, 512))
    # # mask = np.resize(mask, (512, 512))
    # # output = np.resize(output, (512, 512))
    # score = diceloss(output, mask)
    # return image, mask, output, score

if __name__ == "__main__":
    inference(0.5)