#%%
import os, random
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1]=='fingerprint-authentication'][0]

out_path = f'{project_dir}\\data\\final\\'

class SiameseDataset(Dataset):
    def __init__(self, image_folder=out_path, transform=None, device='cpu'):
        self.image_folder = image_folder
        self.transform = transform
        self.device = device
        self.data = []
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
              self.data.append([userImage, userImageCopy, 1.0])
          for userImage in usersImages:
            for _ in usersImages:
              self.data.append([userImage, notUsersImages[random.randrange(len(notUsersImages))], 0.0])

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img1 = Image.open(out_path + '\\' + self.data[index][0])
        img2 = Image.open(out_path + '\\' + self.data[index][1])   

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)           
        
        if (self.data[index][2] == 0.0):
          return img1, img2, torch.Tensor([0.0]) 
        elif (self.data[index][2] == 1.0):
          return img1, img2, torch.Tensor([1.0])
        
if __name__ == '__main__':
   ds = SiameseDataset()
# %%
