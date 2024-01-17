#%%
"""
This module is responsible for creating dataset by pairing two images with each other and labels them
based on whether they belong to the same user (label 0) or to different users (label 1).

All of the mentioned functionality is encapsulated in SiameseDataset() class.
"""
import os, random
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1]=='fingerprint-authentication'][0]

out_path = f'{project_dir}\\data\\final\\'

class SiameseDataset(Dataset):
    """
    SiameseDataset class is written with usage of PyTorch framework.
    Class uses PyTorch type "Dataset", which works similarly to basic collection, however inclusion of this type
    helps us use additional features of PyTorch, such as DataLoaders.
    
    Mentioned class contains following methods:
        - __init__(image_folder, transform, device): responsible for creating pairs of images for dataset,
        - __len__(): method performing as parameter of length of the created dataset,
        - __getitem__(index): method transforming data when sample from dataset is called.
        
    More information on functionality inside documentation of each method.
    """
    def __init__(self, image_folder=out_path, transform=None, device='cpu'):
        """
        Method __init__() is a core method that performs the operation of dataset creation.
        
        Method is set with static argument values, however this method is written in a way that
        in initialization of dataset, it is possible to change the path of image storage (image_folder),
        pass additional transformations to be performed on images (transform) and choose whether or not
        initialization should be done on CPU or GPU.
        
        Firstly, method creates collection of all unique users id (usersIndexes).
        
        For each user, starts a creation of two collections:
            - usersImages: holds all images of specific user id,
            - notUsersImages: holds all images of all users besides specific user id.
        
        In the next steps: for each image that belongs to specific user, a pair of 
        similar images is added to dataset (excluding perfect pairs) and a pair of dissimilar
        images by picking random image from notUsersImages collection to pair with.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.device = device
        self.dataArray = []
        file_list = os.listdir(image_folder)
        usersIndexes = []
        for x in file_list:
          if x[0:2] not in usersIndexes:
            usersIndexes.append(x[0:2])
            
        for user in usersIndexes:
          usersImages = [x for x in file_list if str(user) in x]
          notUsersImages = [x for x in file_list if str(user) not in x]

          for userImage in usersImages:
            for userImageCopy in usersImages:
              if(userImage != userImageCopy):
                self.dataArray.append([userImage, userImageCopy, 0.0])
                self.dataArray.append([userImage, notUsersImages[random.randrange(len(notUsersImages))], 1.0])

        
    def __len__(self):
        """ 
        Method __len__() serves as an attribute for PyTorch framework to pinpoint the amount of samples in dataset.
        
        Variable 'data' is a main container for all samples of dataset.
        """
        return len(self.dataArray)
      

    def __getitem__(self, index):
        """ 
        Method __getitem__(index) is required by PyTorch framework Dataset type and serves as an algorithm that is being performed 
        if dataset sample is being called from DataLoader or Dataset itself.
        
        When dataset sample is called, first the names of images are being converted to Image format by retrieving them from '..\data\processed'
        folder, performs image transformation if any operations were passed as arguments and regardless of label value, the sample is returned 
        as tuple of first image, second image and label inside a tensor.
        """
        img1 = Image.open(out_path + '\\' + self.dataArray[index][0])
        img2 = Image.open(out_path + '\\' + self.dataArray[index][1])   

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)           
        
        if (self.dataArray[index][2] == 0.0):
          return img1, img2, torch.Tensor([0.0]) 
        elif (self.dataArray[index][2] == 1.0):
          return img1, img2, torch.Tensor([1.0])
        
if __name__ == '__main__':
   ds = SiameseDataset()
# %%
