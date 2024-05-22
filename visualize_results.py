# %matplotlib inline
from config import config
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

def plot_loss():
  if not os.path.exists(config.result_folder_path):
      print("No such dir found: ", config.result_folder_path)
      return 0
  else:
    df = pd.read_csv(Path(config.result_folder_path).joinpath("results.csv"))
    epochs = range(1, config.epochs + 1)

    plt.figure(figsize=(18, 6))  # Adjust figsize as needed

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df['training_loss'], 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Validation Dice Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, df['validation_dice_score'], 'r', label='Validation Dice Score')
    plt.title('Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()

    # # Plot Learning Rate
    # plt.subplot(1, 3, 3)
    # plt.plot(epochs, df['learning_rate'], 'g', label='Learning Rate')
    # plt.title('Learning Rate')
    # plt.xlabel('Epochs')
    # plt.ylabel('Learning Rate')
    # plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(config.result_folder_path.joinpath('plot.png'))

    # Show the plot
    plt.show()

def plot_inference_result(image, mask, output, title, transparency=0.38):

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(
        20, 15), gridspec_kw={'wspace': 0.025, 'hspace': 0.010})
    fig.suptitle(title, x=0.5, y=0.92, fontsize=20)
  
    axs[0][0].set_title("Original Mask", fontdict={'fontsize': 16})
    axs[0][0].imshow(mask, cmap='gray')
    axs[0][0].set_axis_off()

    axs[0][1].set_title("Constructed Mask", fontdict={'fontsize': 16})
    axs[0][1].imshow(output, cmap='gray')
    axs[0][1].set_axis_off()

    mask_diff = np.abs(np.subtract(mask, output))
    axs[0][2].set_title("Mask Difference", fontdict={'fontsize': 16})
    axs[0][2].imshow(mask_diff, cmap='gray')
    axs[0][2].set_axis_off()

    seg_output = mask*transparency
    seg_image = np.add(image, seg_output)/2
    axs[1][0].set_title("Original Segment", fontdict={'fontsize': 16})
    axs[1][0].imshow(seg_image, cmap='gray')
    axs[1][0].set_axis_off()

    seg_output = output*transparency
    seg_image = np.add(image, seg_output)/2
    axs[1][1].set_title("Constructed Segment", fontdict={'fontsize': 16})
    axs[1][1].imshow(seg_image, cmap='gray')
    axs[1][1].set_axis_off()

    axs[1][2].set_title("Original Image", fontdict={'fontsize': 16})
    axs[1][2].imshow(image, cmap='gray')
    axs[1][2].set_axis_off()

    plt.tight_layout()
    plt.savefig(config.result_folder_path.joinpath('inference_images.png'), dpi=90, bbox_inches='tight')
    plt.show()




