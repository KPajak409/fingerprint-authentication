#%%
import torch, wandb, os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import imp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from siamese_nn import Siamese_nn
from src.data.fingerprint_dataset import SiameseDataset
from pathlib import Path

#workaround 1
#siameseNNSource = imp.load_source('siamese_nn', '../siamese_nn.py')
#siameseDatasetSource = imp.load_source('fingerprint_dataset', '../fingerprint_dataset.py')

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1]=='fingerprint-authentication'][0]
os.environ["WANDB_NOTEBOOK_NAME"] = os.getcwd()
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f'Active device: {device}')

# hyperparams
config = dict(
    epochs=1,
    batch_size=1,
    learning_rate=0.001,
    split=[0.8, 0.2],
    dataset="fingerprints-dataset",
    architecture="Siamese-neural-network"
)

def make(config):
    transform = transforms.Compose([transforms.ToTensor()])

    siameseDataset = SiameseDataset(transform=transform, device=device)
    train_ds, test_ds = random_split(siameseDataset, config.split)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=config.batch_size)
    
    #To jest do podmiany
    #test_loader = DataLoader(test_ds, shuffle=True, batch_size=9)
    test_loader = DataLoader(test_ds, shuffle=True, batch_size=1)
    
    model = Siamese_nn().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer


def train_and_log(model, train_loader, test_loader, criterion, optimizer, config):
    images1, images2, label_test = next(iter(test_loader))
    images1 = images1.to(device)
    images2 = images2.to(device)
    label_test = label_test.to(device)
    wandb.watch(model, log_freq=100)
    
    for epoch in range(config.epochs):

        for i, (img1, img2, label) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
            if i%1==0:
                print(f'Epoch [{epoch + 1}/{config.epochs}], step: [{i}/{len(train_loader)}] Loss: {loss.item():.4f}')
            if i%200==0:
                if (label.item() == 1 and abs(loss - 4) < 0.0001) or (label.item() == 0 and loss < 0.0001):
                    #zakomentowane bo nie dziala mi
                    #torch.save(model.state_dict(), f'{project_dir}\models\training{i}')
                    break
                    
                
        model.eval()
        img_list = []
        with torch.no_grad():  
            out1, out2 = model(images1, images2)
            for i in range(images1.shape[0]):
                pair = torch.hstack((images1[i], images2[i]))
                pred = criterion.forward(out1[i], out2[i], label_test[i])
                wandb_img = wandb.Image(pair, caption=f'Label: {label_test[i].item()} ; CL:{pred.item():.4f}')
                img_list.append(wandb_img)
        wandb.log({'Examples': img_list}, commit=False)
                
             
def test(model, test_loader, criterion, device):
    import matplotlib.pyplot as plt

    images1, images2, label_test = next(iter(test_loader))
    images1 = images1.to(device)
    images2 = images2.to(device)
    label_test = label_test.to(device)     
    model = model.to(device)
    
    out1, out2 = model(images1, images2)

    images1 = torch.permute(images1, (0,2,3,1))
    images2 = torch.permute(images2, (0,2,3,1))

    plt.rcParams.update({'font.size': 6})
    for i in range(images1.shape[0]):
        pair = torch.hstack((images1[i], images2[i]))
        pred = criterion.forward(out1[i], out2[i], label_test[i])
        plt.subplot(3,3,i+1)
        
        plt.xticks([])
        plt.yticks([])
        plt.title(f'{label_test[i].item()} CL: {pred.item():.4f}')
        plt.tight_layout()
        plt.imshow(pair)
   
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)       

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        lossContrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return lossContrastive
#%% 

def calculateConfusionMatrixAndThreshold(model, train_loader, test_loader):
    with torch.no_grad():
        transform = transforms.Compose([transforms.ToTensor()]) 
        siameseDataset = SiameseDataset(transform=transform, device=device)
        dataLoader = DataLoader(siameseDataset, shuffle=True, batch_size=1)
        
        samples = 500
        
        iter_disp = 1
        
        predicted = []
        actual = []
        histogramValues = []
        for i in range(samples):
            #images1, images2, label = next(iter(dataLoader))
            images1, images2, label = next(iter(test_loader))
            #images1, images2, label = next(iter(train_loader))
            
            images1 = images1.to(device)
            images2 = images2.to(device)
            label = label.to(device)     
            model = model.to(device)
    
            out1, out2 = model(images1, images2)
            
            placeholder = criterion.forward(out1,out2,label)
            actual.append(siameseDataset[i][2].item())
            histogramValues.append(placeholder.item())
            if siameseDataset[i][2].item() == 1.0:
                if placeholder.item() > 3:
                    predicted.append(1.0)
                else:
                    predicted.append(0.0)
            elif siameseDataset[i][2].item() == 0.0:
                if placeholder.item() < 1:
                    predicted.append(0.0)
                else:
                    predicted.append(1.0)
            
            #print(siameseDataset[i][2].item(), ' ', predicted[i])        
                    
            print(iter_disp, '/', samples, end = "\r")
            iter_disp = iter_disp + 1
            
        listToSetHistogramValues = set(histogramValues)
        uniqueHistogramValues = list(listToSetHistogramValues)

        cm = confusion_matrix(actual, predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.figure(0)
        plt.show()
        plt.figure(1)
        plt.hist(histogramValues, bins=len(uniqueHistogramValues))
        plt.show()
    return 0

#%%
def model_pipeline(hyperparameters, wandb_mode = 'online'): 
    with wandb.init(project='fingerprint-authentication-ISU', config=hyperparameters, mode = wandb_mode):
        config = wandb.config
        
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)
        
        train_and_log(model, train_loader, test_loader, criterion, optimizer, config)
        
        test(model, test_loader, criterion, 'cpu')
        
        #calculateConfusionMatrixAndThreshold(model, train_loader, test_loader)
        
    return model, train_loader, test_loader, criterion, optimizer

#%%

# wandb_mode disabled for turn off logging
model, train_loader, test_loader, criterion, _ = model_pipeline(config, wandb_mode='disabled')

#%%

test(model, test_loader, criterion)

#%%

calculateConfusionMatrixAndThreshold(model, train_loader, test_loader)


# %%
