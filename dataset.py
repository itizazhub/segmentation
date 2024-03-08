from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from config import config
from sklearn.model_selection import train_test_split
from PIL import Image
import random


class TumorDataset(Dataset):
    """ Returns a TumorDataset class object which represents our tumor dataset.
    TumorDataset inherits from torch.utils.data.Dataset class.
    """

    def __init__(self, dataset_tuple:tuple, transform=True, DEBUG=False):
        """ Constructor for our TumorDataset class.
        Parameters:
            root_dir(str): Directory with all the images.
            transform(bool): Flag to apply image random transformation.
            DEBUG(bool): To switch to debug mode for image transformation.

        Returns: None
        """
        self.x, self.y = dataset_tuple

        # self.root_dir = root_dir
        self.transform = {'hflip': TF.hflip,
                          'vflip': TF.vflip,
                          'rotate': TF.rotate}
        self.default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((config.image_size, config.image_size))
        ])
        # self.DEBUG = DEBUG
        if not config.transform:
            self.transform = None

    def __getitem__(self, index):
        """ Overridden method from inheritted class to support
        indexing of dataset such that datset[I] can be used
        to get Ith sample.
        Parameters:
            index(int): Index of the dataset sample

        Return:
            sample(dict): Contains the index, image, mask torch.Tensor.
                        'index': Index of the image.
                        'image': Contains the tumor image torch.Tensor.
                        'mask' : Contains the mask image torch.Tensor.
        """
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

        # image_name = os.path.join(self.root_dir, str(index)+'.png')
        # mask_name = os.path.join(self.root_dir, str(index)+'_mask.png')

        # image = Image.open(image_name)
        # mask = Image.open(mask_name)

        # image = self.default_transformation(image)
        # mask = self.default_transformation(mask)

        # # Custom transformations
        # if self.transform:
        #     image, mask = self._random_transform(image, mask)

        # image = TF.to_tensor(image)
        # mask = TF.to_tensor(mask)

        # sample = {'index': int(index), 'image': image, 'mask': mask}
        # return sample

    def _random_transform(self, image, mask):
        """ Applies a set of transformation in random order.
        Each transformation has a probability of 0.5
        """
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
        """ Overridden method from inheritted class so that
        len(self) returns the size of the dataset.
        """
        return len(self.x)
        # error_msg = 'Part of dataset is missing!\nNumber of tumor and mask images are not same.'
        # total_files = len(os.listdir(self.root_dir))

        # assert (total_files % 2 == 0), error_msg
        # return total_files//2
    

class DatasetCreator:
    def __init__(self) -> None:

        self.x = [i for i in config.images_path.glob("*.png")]
        self.y = [i for i in config.masks_path.glob("*.png")]
    def split_data(self) -> tuple:
        # Split dataset into train, test, and validation sets
        train_x, test_x, train_y, test_y = train_test_split(self.x, self.y, test_size=config.test_ratio, random_state=config.random_state)
        # val_x, test_x, val_y, test_y = train_test_split(test_val_x, test_val_y, test_size=config.test_ratio/(config.test_ratio+config.val_ratio), random_state=config.random_state)
        
        return (train_x, train_y), (test_x, test_y)
    
# from torch.utils.data import DataLoader
# train, test = DatasetCreator().split_data()
# train_set = TumorDataset(train)
# test_set = TumorDataset(test)
# print(f"Train size: {train_set.__len__()} Test size: {test_set.__len__()}")
# train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
# for image, mask in iter(train_loader):
#     print(image.shape, mask.shape)
#     break

# self.train_dataset, self.test_dataset, self.val_dataset = DatasetCreator()
# self.train_set = TumorDataset(self.train_dataset, config.image_size)
# self.test_set = TumorDataset(self.test_dataset, config.image_size)
# self.val_set = TumorDataset(self.val_dataset, config.image_size)
# self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True)
# self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size, shuffle=True)
# self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)