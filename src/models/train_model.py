#%%
import torch, math
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from siamese_nn import Siamese_nn
from src.data.fingerprint_dataset import SiameseDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# hyperparams
batch_size =1


def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    steps = math.floor(len(dataloader)/batch_size)
    for epoch in range(num_epochs):
        for i, (img1, img2, label) in enumerate(dataloader):
            optimizer.zero_grad()
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            
            if i%10==0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], step: [{i}/{steps}] Loss: {loss.item():.4f}')
            

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

siameseDataset = SiameseDataset(transform=transform, device=device)
dataloader = DataLoader(siameseDataset, shuffle=True, batch_size=batch_size)

siameseNet = Siamese_nn().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(siameseNet.parameters(), lr=0.001)

#%%

train(siameseNet, dataloader, criterion, optimizer, num_epochs=1)

# %%
# Testing results
import matplotlib.pyplot as plt

with torch.no_grad():
    test_dataloader = DataLoader(siameseDataset, shuffle=True, batch_size=9)
    test_dataloader_iter = iter(test_dataloader)
    images1, images2, label = next(test_dataloader_iter)

    siameseNet = siameseNet.to('cpu')
    out1, out2 = siameseNet(images1, images2)
    # problem with contrastive loss
    print(out1.shape, out2.shape)
    pred = criterion.forward(out1[0], out2[0], label[0])
    pred2 = criterion.forward(out1, out2, label)
    
    print(pred, pred2)
    
    images1 = torch.permute(images1, (0,2,3,1))
    images2 = torch.permute(images2, (0,2,3,1))
    
    plt.rcParams.update({'font.size': 6})
    for i in range(images1.shape[0]):
        pair = torch.hstack((images1[i], images2[i]))
        pred = criterion.forward(out1[0], out2[0], label[0])
        plt.subplot(3,3,i+1)
        
        plt.xticks([])
        plt.yticks([])
        plt.title(f'{label[i].item()} CL: {round(pred.item(),5)}')
        plt.tight_layout()
        plt.imshow(pair)
    

# %%

class_balance = torch.tensor((0,0))
for i, (img1, img2, label) in enumerate(dataloader):
    if label == 0:
        class_balance[0] += 1
    elif label == 1:
        class_balance[1] += 1

print(class_balance, len(siameseDataset))