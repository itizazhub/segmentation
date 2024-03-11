'''
need model
data loaders
train model

when you make object of this class,
1. constrcutor provides model with best weights, loss funtions, and device
2. setup_training_env(), the setup for the training prepares eg dataset, dataloaders and optimizer
3. train(), trains the model and saves best weights and optimizer
4. test(), runs over test loader and prints total loss
5. save_results_to_csv(), saves histoty in csv file

'''

from unet_model import UNet
from loss import dice_loss, dice_coeff
from dataset import DatasetCreator, TumorDataset
from config import config

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import logging
import os
from pathlib import Path
import torch.nn as nn
import math

class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = UNet()
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum, foreach=True)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'max', factor=config.factor, patience=config.patience)
        if config.load_weights:
            checkpoint_path = config.model_weights_path.joinpath("best.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Best weights and optimizer parameters are loaded")

            else:
                if os.path.exists(config.pre_trained_model_path):
                    checkpoint_path = config.pre_trained_model_path.joinpath('weights.pt')
                    checkpoint = torch.load(checkpoint_path) #, map_location=torch.device('cpu')
                    self.model.load_state_dict(checkpoint, strict=False)
                    print("Pre-trained weights are loaded")
        else:
            print("Starting from scratch")
        # print(f"Train size: {train_set.__len__()} Test size: {test_set.__len__()}")

    def setup_training_env(self):
        self.training_loss = []
        self.validation_dice_score = []
        self.scheduler_loss = []
        # self.training_accuracy = []
        # self.validation_accuracy = []
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
            # correct_predictions = 0
            # total_samples = 0
            epoch_loss = 0
            for img, label in iter(self.train_loader):
                self.optimizer.zero_grad()
                img, label = torch.tensor(img), torch.tensor(label)
                img = img.to(self.device)
                label = label.to(self.device)
                pred = self.model(img)
                loss = self.criterion(pred.squeeze(1), label.squeeze(1).float())
                loss += dice_loss(F.sigmoid(pred.squeeze(1)), label.squeeze(1).float(), multiclass=False)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                # Calculate accuracy
                # _, predicted_labels = torch.max(pred, 1)
                # correct_predictions += (predicted_labels == label).sum().item()
                # total_samples += label.size(0)
            self.training_loss.append(epoch_loss / len(self.train_loader))

            # self.training_accuracy.append(correct_predictions / total_samples)

            ################### Validation ###################
            dice = 0
            # correct_predictions = 0
            # total_samples = 0
            self.model.eval()
            with torch.no_grad():
                for img, label in iter(self.test_loader):
                    img, label = torch.tensor(img), torch.tensor(label)
                    img = img.to(self.device)
                    label = label.to(self.device)
                    self.model.to(self.device)
                    pred = self.model(img)
                    pred = (F.sigmoid(pred) > 0.5).float()
                    dice += dice_coeff(pred.squeeze(1), label.squeeze(1), reduce_batch_first=False)
                    # Calculate accuracy
                    # _, predicted_labels = torch.max(pred, 1)
                    # correct_predictions += (predicted_labels == label).sum().item()
                    # total_samples += label.size(0)
            self.model.train()
            dice_score = dice / max(len(self.test_loader), 1)
            self.scheduler.step(dice_score)
            self.scheduler_loss.append(self.scheduler._last_lr[0])
            self.validation_dice_score.append(dice_score)
            # self.validation_accuracy.append(correct_predictions / total_samples)

            logging.info(f'Epoch: {epoch}, Training Loss: {self.training_loss[-1]}, Validation dice score: {self.validation_dice_score[-1]}, scheduler: {self.scheduler_loss[-1]}')
            print(f'Epoch: {epoch}, Training Loss: {self.training_loss[-1]:0.4}, Validation dice score: {self.validation_dice_score[-1]:0.4}, scheduler: {self.scheduler_loss[-1]}')
            # self.save_results_to_csv()
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


    def test_fn(self):
        losses = []
        # correct_predictions = 0
        # total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for img, label in iter(self.test_loader):
                img, label = torch.tensor(img), torch.tensor(label)
                img = img.to(self.device)
                label = label.to(self.device)
                self.model.to(self.device)
                pred = self.model(img)
                pred = (pred > config.threshold)
                loss = self.dice_coefficient(pred, label)
                losses.append(loss.item())
                # Calculate accuracy
                # _, predicted_labels = torch.max(pred, 1)
                # correct_predictions += (predicted_labels == label).sum().item()
                # total_samples += label.size(0)
                
        print("Dice Loss: ", sum(losses) / len(losses))


    def save_results_to_csv(self):
        data = {
            'training_loss': self.training_loss,
            'validation_dice_score': self.validation_dice_score,
            'scheduler_loss': self.scheduler_loss
        }
        data_cpu = {key: value.cpu().numpy() for key, value in data.items()}
        df = pd.DataFrame(data_cpu)
        if not os.path.exists(config.result_folder_path):
            os.mkdir(config.result_folder_path)
        df.to_csv(config.result_folder_path.joinpath("results.csv"), index=False)
        # df_classes = pd.DataFrame(self.classes)
        # df_classes.to_csv(Path(config.result_folder_path).joinpath("classes.csv"), index=False)


