import torch
from trainer import Trainer
from visualize_results import plot_loss, plot_inference_result
from inference import inference
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

if __name__ == "__main__":
    trainer_obj = Trainer()
    trainer_obj.setup_training_env()
    trainer_obj.train_fn()
    print(trainer_obj.test_fn())
    trainer_obj.save_results_to_csv()
    plot_loss()
    # inference()
    