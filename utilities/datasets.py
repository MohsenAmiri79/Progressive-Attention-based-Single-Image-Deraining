from math import ceil as c
from numpy.random import rand as r

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os


class SID_dataset(Dataset):
    def __init__(self, input_path, label_path, im_size):
        self.input_path = input_path
        self.label_path = label_path

        self.input_files = os.listdir(input_path)
        
        self.transforms = transforms.Compose([
            transforms.Resize(size=im_size),
            transforms.CenterCrop([im_size, im_size]), 
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        image_name = self.input_files[idx]
        temp = image_name.split('_')
        label_name = temp[0] + temp[1][-4:]

        label_image_path = os.path.join(self.label_path, label_name)
        label_image = Image.open(label_image_path).convert('RGB')

        input_image_path = os.path.join(self.input_path, image_name)
        input_image = Image.open(input_image_path).convert('RGB')
        
        input = self.transforms(input_image)
        label = self.transforms(label_image)
        return input, label




class SID_dataset_mini(Dataset):
    def __init__(self, input_path, label_path, im_size):
        self.input_path = input_path
        self.label_path = label_path

        self.label_files = os.listdir(label_path)
        
        self.transforms = transforms.Compose([
            transforms.Resize(size=im_size),
            transforms.CenterCrop([im_size, im_size]), 
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        label_name = self.label_files[idx]
        temp = label_name.split('.')
        image_name = temp[0] + f'_{c(r()*14)}.' + temp[1]

        label_image_path = os.path.join(self.label_path, label_name)
        label_image = Image.open(label_image_path).convert('RGB')

        input_image_path = os.path.join(self.input_path, image_name)
        input_image = Image.open(input_image_path).convert('RGB')
        
        input = self.transforms(input_image)
        label = self.transforms(label_image)
        return input, label
    