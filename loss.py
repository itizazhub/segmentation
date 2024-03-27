import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predicted, target):
        batch = predicted.size()[0]
        batch_loss = 0
        for index in range(batch):
            coefficient = self._dice_coefficient(
                predicted[index], target[index])
            batch_loss += coefficient

        batch_loss = batch_loss / batch

        return 1 - batch_loss

    def _dice_coefficient(self, predicted, target):
        smooth = 1
        product = torch.mul(predicted, target)
        intersection = product.sum()
        coefficient = (2*intersection + smooth) / \
            (predicted.sum() + target.sum() + smooth)
        return coefficient


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, predicted, target):
        return F.binary_cross_entropy(predicted, target) \
            + self.dice_loss(predicted, target)
