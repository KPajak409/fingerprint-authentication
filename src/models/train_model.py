#%%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from siamese_nn import Siamese_nn
import os, random

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1]=='fingerprint-authentication'][0]

out_path = f'{project_dir}\\data\\final\\'

def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for img1, img2, label in dataloader:
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            
class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
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
        return len(self.image_folder)

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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        lossContrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return lossContrastive


transform = transforms.Compose([transforms.ToTensor()])

siameseDataset = SiameseDataset(image_folder=out_path, transform=transform)
dataloader = DataLoader(siameseDataset, shuffle=True, batch_size=200)


siameseNet = Siamese_nn()
criterion = ContrastiveLoss()
optimizer = optim.Adam(siameseNet.parameters(), lr=0.001)


train(siameseNet, dataloader, criterion, optimizer, num_epochs=20)

# %%
