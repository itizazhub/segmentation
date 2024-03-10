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
  else:
    df = pd.read_csv(Path(config.result_folder_path).joinpath("results.csv"))
    epochs = range(1, config.epochs + 1)

    plt.figure(figsize=(12, 6))

    # Plot training loss
  #   plt.subplot(1, 2, 1)
    plt.plot(epochs, df['training_loss'], 'b', label='Training loss')
    plt.plot(epochs, df['validation_dice_score'], 'r', label='validation_dice_score')
    plt.plot(epochs, df['scheduler_loss'], 'r', label='Scheduler loss')
    plt.title('Training and validation_dice_score')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training accuracy
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, df['training_accuracy'], 'b', label='Training accuracy')
    # plt.plot(epochs, df['validation_accuracy'], 'r', label='Validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    plt.savefig(config.result_folder_path.joinpath('loss_plot.png'))
    
    plt.tight_layout()
    plt.show()


def plot_inference_result(image, mask, output, title, transparency=0.38):
    """ Plots a 2x3 plot with comparisons of output and original image.
    Works best with Jupyter Notebook/Lab.
    Parameters:
        image(numpy.ndarray): Array containing the original image of MRI scan.
        mask(numpy.ndarray): Array containing the original mask of tumor.
        output(numpy.ndarray): Model constructed mask from input image.
        title(str): Title of the plot to be used.
        transparency(float): Transparency level of mask on images.
                             Default: 0.38
        save_path(str): Saves the plot to the location specified.
                        Does nothing if None. 
                        Default: None
    Return:
        None
    """

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




