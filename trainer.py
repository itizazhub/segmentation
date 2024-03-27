from unet_model import UNet
from loss import DiceLoss, BCEDiceLoss
from dataset import DatasetCreator, TumorDataset
from config import config

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import pandas as pd
import logging
import os

class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = UNet()
        self.model.to(self.device)

        self.criterion = BCEDiceLoss().to(self.device)
        self.dice_loss_fn = DiceLoss().to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum, foreach=True)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min', factor=config.factor, patience=config.patience)
        if config.load_weights:
            checkpoint_path = config.model_weights_path.joinpath("best_128.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Best weights and optimizer parameters are loaded")

            else:
                print("Starting from scratch")
        else:
                print("Starting from scratch")


    def setup_training_env(self):
        self.training_loss = []
        self.validation_dice_score = []
        self.learning_rate = []
        self.train_set, self.test_set = DatasetCreator().split_data()
        self.train_dataset = TumorDataset(self.train_set)
        self.test_dataset = TumorDataset(self.test_set)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        print("Device: ", self.device)
        print("Training data: ", self.train_dataset.__len__())
        print("Test data: ", self.test_dataset.__len__())
        print("----Training started----")


    def train_fn(self):
        logging.basicConfig(filename=config.log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        best_dice_score = float(-1)
        best_loss = float("inf")
        for epoch in range(config.epochs):
            self.model.train()
            epoch_loss = 0
            for img, label in iter(self.train_loader):
                self.optimizer.zero_grad()
                img, label = torch.tensor(img), torch.tensor(label)
                img = img.to(self.device)
                label = label.to(self.device)
                pred = self.model(img)
                loss = self.criterion(pred.float(), label.float())
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            epoch_loss = epoch_loss / max(len(self.train_loader), 1)
            self.scheduler.step(epoch_loss)
            self.learning_rate.append(self.scheduler._last_lr[0])
            self.training_loss.append(epoch_loss)

            ################### Validation ###################
            dice = 0
            self.model.eval()
            with torch.no_grad():
                for img, label in iter(self.test_loader):
                    img, label = torch.tensor(img), torch.tensor(label)
                    img = img.to(self.device)
                    label = label.to(self.device)
                    pred = self.model(img)
                    pred = (pred > 0.5)
                    dice += (1.0 - self.dice_loss_fn(pred.float(), label.float()))

            self.model.train()
            dice = dice / max(self.test_dataset.__len__(), 1)
            self.validation_dice_score.append(dice.cpu().numpy())

            logging.info(f'Epoch: {epoch}, Training Loss: {self.training_loss[-1]}, Validation dice score: {self.validation_dice_score[-1]}, learning_rate: {self.learning_rate[-1]}')
            print(f'Epoch: {epoch}, Training Loss: {self.training_loss[-1]:0.4}, Validation dice score: {self.validation_dice_score[-1]:0.4}, learning_rate: {self.learning_rate[-1]}')
            self.save_results_to_csv()
            # Save the model if the validation loss improves
            if (self.validation_dice_score[-1] > best_dice_score):
                best_dice_score = self.validation_dice_score[-1]
                torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }, config.model_weights_path.joinpath(f"best_{epoch}.pth"))
                print("---Best val weights and optimizer parameters are saved---------")
            #save training weights
            if (self.training_loss[-1] < best_loss):
                best_loss = self.training_loss[-1]
                torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }, config.training_weights_path.joinpath(f"best_{epoch}.pth"))
                print("---Best training weights and optimizer parameters are saved----")

    def save_results_to_csv(self):
        data = {
            'training_loss': self.training_loss,
            'validation_dice_score': self.validation_dice_score,
            'learning_rate': self.learning_rate
        }
        df = pd.DataFrame(data)
        if not os.path.exists(config.result_folder_path):
            os.mkdir(config.result_folder_path)
        df.to_csv(config.result_folder_path.joinpath("results.csv"), index=False)

