from trainer import Trainer
import warnings
import os
from config import config
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    if not os.path.exists(config.training_weights_path):
        os.mkdir(config.training_weights_path)
    if not os.path.exists(config.model_weights_path):
        os.mkdir(config.model_weights_path)


    trainer_obj = Trainer()
    trainer_obj.setup_training_env()
    trainer_obj.train_fn()


    