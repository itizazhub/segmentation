from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from config import config
from sklearn.model_selection import train_test_split
from PIL import Image
import random


class TumorDataset(Dataset):

    def __init__(self, dataset_tuple:tuple):
        self.x, self.y = dataset_tuple

        self.transform = {'hflip': TF.hflip,
                          'vflip': TF.vflip,
                          'rotate': TF.rotate}
        self.default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((config.image_size, config.image_size))
        ])
        self.DEBUG = config.DEBUG
        if not config.transform:
            self.transform = None

    def __getitem__(self, index):
        image_path = self.x[index]
        mask_path = self.y[index]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = self.default_transformation(image)
        mask = self.default_transformation(mask)

        if self.transform:
            image, mask = self._random_transform(image, mask)
        
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask


    def _random_transform(self, image, mask):
        choice_list = list(self.transform)
        for _ in range(len(choice_list)):
            choice_key = random.choice(choice_list)
            if self.DEBUG:
                print(f'Transform choose: {choice_key}')
            action_prob = random.randint(0, 1)
            if action_prob >= 0.5:
                if self.DEBUG:
                    print(f'\tApplying transformation: {choice_key}')
                if choice_key == 'rotate':
                    rotation = random.randint(15, 75)
                    if self.DEBUG:
                        print(f'\t\tRotation by: {rotation}')
                    image = self.transform[choice_key](image, rotation)
                    mask = self.transform[choice_key](mask, rotation)
                else:
                    image = self.transform[choice_key](image)
                    mask = self.transform[choice_key](mask)
            choice_list.remove(choice_key)

        return image, mask

    def __len__(self):
        return len(self.x)
    

class DatasetCreator:
    def __init__(self) -> None:
        self.x = [i for i in config.images_path.glob("*.png")]
        self.y = [i for i in config.masks_path.glob("*.png")]
        
    def split_data(self) -> tuple:
        # Split dataset into train, test, and validation sets
        train_x, test_x, train_y, test_y = train_test_split(self.x, self.y, test_size=config.test_ratio, random_state=config.random_state)
        # val_x, test_x, val_y, test_y = train_test_split(test_val_x, test_val_y, test_size=config.test_ratio/(config.test_ratio+config.val_ratio), random_state=config.random_state)
        
        return (train_x, train_y), (test_x, test_y)
    