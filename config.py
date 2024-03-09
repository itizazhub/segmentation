from pathlib import Path

class Config:
    def __init__(self) -> None:
        self.zipped_dataset_path = Path("../5")
        self.unzipped_dataset = Path("/content/drive/MyDrive/segmentation_project/unzipped_dataset")
        self.mat_files_path = Path("/content/drive/MyDrive/segmentation_project/mat_files")
        self.images_path = Path('/content/drive/MyDrive/segmentation_project/images')
        self.masks_path = Path('/content/drive/MyDrive/segmentation_project/masks')
        self.pre_trained_model_path = Path('pre_trained_weights')
        self.model_weights_path = Path('/content/drive/MyDrive/segmentation_project/model_weights')
        self.training_weights_path = Path('/content/drive/MyDrive/segmentation_project/all_weights')
        self.result_folder_path = Path('/content/drive/MyDrive/segmentation_project/results')
        self.inference_images_path = Path('inference_images')
        self.inference_out_images_path = Path('/content/drive/MyDrive/segmentation_project/inference_out_images')
        self.inference_out_masks_path = Path('/content/drive/MyDrive/segmentation_project/inference_out_masks')
        self.combined_image_mask = Path('/content/drive/MyDrive/segmentation_project/combined_image_mask')
        self.log_file_path = Path('/content/drive/MyDrive/segmentation_project/training.log')

        self.train_ratio = 0.9
        # self.val_ratio = 0.2
        self.test_ratio = 0.1
        self.random_state = 42
        self.image_size = 512
        self.transform = True
        self.batch_size = 6

        self.filters = [16,32,64,128,256]
        self.dropout = 0.2
        self.threshold = 0.5
        self.learning_rate = 0.001
        self.momentum = 0.5
        self.weight_decay = 0.0001
        self.load_weights = True
        self.epochs = 200


config = Config()