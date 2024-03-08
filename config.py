from pathlib import Path

class Config:
    def __init__(self) -> None:
        self.zipped_dataset_path = Path("5")
        self.unzipped_dataset = Path("unzipped_dataset")
        self.mat_files_path = Path("mat_files")
        self.images_path = Path('images')
        self.masks_path = Path('masks')
        self.pre_trained_model_path = Path('pre_trained_weights')
        self.model_weights_path = Path('model_weights')
        self.result_folder_path = Path('results')
        self.inference_images_path = Path('inference_images')
        self.inference_out_images_path = Path('inference_out_images')
        self.inference_out_masks_path = Path('inference_out_masks')
        self.combined_image_mask = Path('combined_image_mask')

        self.train_ratio = 0.9
        # self.val_ratio = 0.2
        self.test_ratio = 0.1
        self.random_state = 42
        self.image_size = 512
        self.transform = False
        self.batch_size = 6

        self.filters = [16,32,64,128,256]
        self.threshold = 0.5
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.load_weights = True
        self.epochs = 200


config = Config()