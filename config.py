from pathlib import Path

class Config:
    def __init__(self) -> None:
        self.zipped_dataset_path = Path("../5")
        self.unzipped_dataset = Path("unzipped_dataset")
        self.mat_files_path = Path("mat_files")
        self.images_path = Path('images')
        self.masks_path = Path('masks')
        self.pre_trained_model_path = Path('pre_trained_weights')
        self.model_weights_path = Path('/content/drive/MyDrive/segmentation_project/model_weights')
        self.training_weights_path = Path('/content/drive/MyDrive/segmentation_project/all_weights')
        self.result_folder_path = Path('/content/drive/MyDrive/segmentation_project/results')
        self.inference_images_path = Path('inference_images')
        self.inference_out_images_path = Path('inference_out_images')
        self.inference_out_masks_path = Path('inference_out_masks')
        self.combined_image_mask = Path('combined_image_mask')
        self.log_file_path = Path('/content/drive/MyDrive/segmentation_project/training.log')

        self.train_ratio = 0.8
        # self.val_ratio = 0.2
        self.test_ratio = 0.2
        self.random_state = 42
        self.image_size = 512
        # self.transform = False
        # self.DEBUG = False
        self.batch_size = 1

        self.threshold = 0.5
        self.patience = 3
        self.factor = 0.8
        self.learning_rate = 1e-5
        self.momentum = 0.999
        self.weight_decay = 1e-8
        self.epochs = 150

        self.filters = [16,32,64,128,256]
        self.dropout = 0.2
        self.load_weights = False
        


config = Config()