import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
# from torchsummary import summary


class Unet(nn.Module):
    """ This is the Pytorch version of U-Net Architecture.
    This is not the vanilla version of U-Net.
    For more information about U-Net Architecture check the paper here.
    Link :- https://arxiv.org/abs/1505.04597

    This network is modified to have only 4 blocks depth because of
    computational limitations. 
    The input and output of this network is of the same shape.
    Input Size of Network - (1,512,512)
    Output Size of Network - (1,512,512)
        Shape Format :  (Channel, Width, Height)
    """
    def __init__(self, input_channels=1, output_channels=1):
        super(Unet, self).__init__()
        """ Constructor for UNet class.
        Parameters:
            filters(list): Five filter values for the network.
            input_channels(int): Input channels for the network. Default: 1
            output_channels(int): Output channels for the final network. Default: 1
        """
        self.filters = config.filters

        if len(self.filters) != 5:
            raise Exception(f"Filter list size {len(self.filters)}, expected 5!")

        padding = 1
        ks = 3

        # Encoding Part of Network.
        #   Block 1
        self.conv1_1 = nn.Conv2d(input_channels, self.filters[0], kernel_size=ks, padding=padding)
        self.bn1_1 = nn.BatchNorm2d(self.filters[0])
        self.conv1_2 = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=ks, padding=padding)
        self.bn1_2 = nn.BatchNorm2d(self.filters[0])
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(p=config.dropout)
        #   Block 2
        self.conv2_1 = nn.Conv2d(self.filters[0], self.filters[1], kernel_size=ks, padding=padding)
        self.bn2_1 = nn.BatchNorm2d(self.filters[1])
        self.conv2_2 = nn.Conv2d(self.filters[1], self.filters[1], kernel_size=ks, padding=padding)
        self.bn2_2 = nn.BatchNorm2d(self.filters[1])
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(p=config.dropout)
        #   Block 3
        self.conv3_1 = nn.Conv2d(self.filters[1], self.filters[2], kernel_size=ks, padding=padding)
        self.bn3_1 = nn.BatchNorm2d(self.filters[2])
        self.conv3_2 = nn.Conv2d(self.filters[2], self.filters[2], kernel_size=ks, padding=padding)
        self.bn3_2 = nn.BatchNorm2d(self.filters[2])
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(p=config.dropout)
        #   Block 4
        self.conv4_1 = nn.Conv2d(self.filters[2], self.filters[3], kernel_size=ks, padding=padding)
        self.bn4_1 = nn.BatchNorm2d(self.filters[3])
        self.conv4_2 = nn.Conv2d(self.filters[3], self.filters[3], kernel_size=ks, padding=padding)
        self.bn4_2 = nn.BatchNorm2d(self.filters[3])
        self.maxpool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout2d(p=config.dropout)
        
        # Bottleneck Part of Network.
        self.conv5_1 = nn.Conv2d(self.filters[3], self.filters[4], kernel_size=ks, padding=padding)
        self.bn6_1 = nn.BatchNorm2d(self.filters[3])
        self.conv5_2 = nn.Conv2d(self.filters[4], self.filters[4], kernel_size=ks, padding=padding)
        self.bn6_2 = nn.BatchNorm2d(self.filters[3])
        self.conv5_t = nn.ConvTranspose2d(self.filters[4], self.filters[3], 2, stride=2)

        # Decoding Part of Network.
        #   Block 4
        self.conv6_1 = nn.Conv2d(self.filters[4], self.filters[3], kernel_size=ks, padding=padding)
        self.bn6_1 = nn.BatchNorm2d(self.filters[3])
        self.conv6_2 = nn.Conv2d(self.filters[3], self.filters[3], kernel_size=ks, padding=padding)
        self.bn6_2 = nn.BatchNorm2d(self.filters[3])
        self.conv6_t = nn.ConvTranspose2d(self.filters[3], self.filters[2], 2, stride=2)
        #   Block 3
        self.conv7_1 = nn.Conv2d(self.filters[3], self.filters[2], kernel_size=ks, padding=padding)
        self.bn7_1 = nn.BatchNorm2d(self.filters[2])
        self.conv7_2 = nn.Conv2d(self.filters[2], self.filters[2], kernel_size=ks, padding=padding)
        self.bn7_2 = nn.BatchNorm2d(self.filters[2])
        self.conv7_t = nn.ConvTranspose2d(self.filters[2], self.filters[1], 2, stride=2)
        #   Block 2
        self.conv8_1 = nn.Conv2d(self.filters[2], self.filters[1], kernel_size=ks, padding=padding)
        self.bn8_1 = nn.BatchNorm2d(self.filters[1])
        self.conv8_2 = nn.Conv2d(self.filters[1], self.filters[1], kernel_size=ks, padding=padding)
        self.bn8_2 = nn.BatchNorm2d(self.filters[1])
        self.conv8_t = nn.ConvTranspose2d(self.filters[1], self.filters[0], 2, stride=2)
        #   Block 1
        self.conv9_1 = nn.Conv2d(self.filters[1], self.filters[0], kernel_size=ks, padding=padding)
        self.bn9_1 = nn.BatchNorm2d(self.filters[0])
        self.conv9_2 = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=ks, padding=padding)
        self.bn9_2 = nn.BatchNorm2d(self.filters[0])


        # Output Part of Network.
        self.conv10 = nn.Conv2d(self.filters[0], output_channels, kernel_size=ks, padding=padding)

    def forward(self, x):
        # Encoding Part of Network.
        #   Block 1
        conv1 = F.relu(self.bn1_1(self.conv1_1(x)))
        conv1 = F.relu(self.bn1_2(self.conv1_2(conv1)))
        pool1 = self.maxpool1(conv1)
        pool1 = self.dropout1(pool1)
        #   Block 2
        conv2 = F.relu(self.bn2_1(self.conv2_1(pool1)))
        conv2 = F.relu(self.bn2_2(self.conv2_2(conv2)))
        pool2 = self.maxpool2(conv2)
        pool2 = self.dropout2(pool2)
        #   Block 3
        conv3 = F.relu(self.bn3_1(self.conv3_1(pool2)))
        conv3 = F.relu(self.bn3_2(self.conv3_2(conv3)))
        pool3 = self.maxpool3(conv3)
        pool3 = self.dropout3(pool3)
        #   Block 4
        conv4 = F.relu(self.bn4_1(self.conv4_1(pool3)))
        conv4 = F.relu(self.bn4_2(self.conv4_2(conv4)))
        pool4 = self.maxpool4(conv4)
        pool4 = self.dropout4(pool4)

        # Bottleneck Part of Network.
        conv5 = F.relu(self.bn5_1(self.conv5_1(pool4)))
        conv5 = F.relu(self.bn5_2(self.conv5_2(conv5)))

        # Decoding Part of Network.
        #   Block 4
        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        conv6 = F.relu(self.bn6_1(self.conv6_1(up6)))
        conv6 = F.relu(self.bn6_2(self.conv6_2(conv6)))
        #   Block 3
        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        conv7 = F.relu(self.bn7_1(self.conv7_1(up7)))
        conv7 = F.relu(self.bn7_2(self.conv7_2(conv7)))
        #   Block 2
        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        conv8 = F.relu(self.bn8_1(self.conv8_1(up8)))
        conv8 = F.relu(self.bn8_2(self.conv8_2(conv8)))
        #   Block 1
        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        conv9 = F.relu(self.bn9_1(self.conv9_1(up9)))
        conv9 = F.relu(self.bn9_2(self.conv9_2(conv9)))

        # Output Part of Network.
        output = torch.sigmoid(self.conv10(conv9))

        return output


    # def summary(self, input_size=(1, 512, 512), batch_size=-1, device='cuda'):
    #     """ Get the summary of the network in a chart like form
    #     with name of layer size of the inputs and parameters 
    #     and some extra memory details.
    #     This method uses the torchsummary package.
    #     For more information check the link.
    #     Link :- https://github.com/sksq96/pytorch-summary

    #     Parameters:
    #         input_size(tuple): Size of the input for the network in
    #                              format (Channel, Width, Height).
    #                              Default: (1,512,512)
    #         batch_size(int): Batch size for the network.
    #                             Default: -1
    #         device(str): Device on which the network is loaded.
    #                         Device can be 'cuda' or 'cpu'.
    #                         Default: 'cuda'

    #     Returns:
    #         A printed output for IPython Notebooks.
    #         Table with 3 columns for Layer Name, Input Size and Parameters.
    #         torchsummary.summary() method is used.
    #     """
    #     return summary(self, input_size, batch_size, device)