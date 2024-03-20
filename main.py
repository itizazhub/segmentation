from trainer import Trainer
from visualize_results import plot_loss, plot_inference_result
from inference import inference
import warnings
import os
from config import config
warnings.filterwarnings('ignore')
#%matplotlib inline

if __name__ == "__main__":
    if not os.path.exists(config.training_weights_path):
        os.mkdir(config.training_weights_path)
    if not os.path.exists(config.model_weights_path):
        os.mkdir(config.model_weights_path)


    trainer_obj = Trainer()
    trainer_obj.setup_training_env()
    trainer_obj.train_fn()
    trainer_obj.save_results_to_csv()

    