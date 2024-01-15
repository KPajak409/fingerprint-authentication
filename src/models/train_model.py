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
from src.models.siamese_nn import Siamese_nn
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
    epochs=20,
    batch_size=32,
    learning_rate=0.0001,
    split=[0.9, 0.1],
    dataset="fingerprints-dataset",
    architecture="Siamese-neural-network"
)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        lossContrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return lossContrastive


def make(config):
    transform = transforms.Compose([transforms.ToTensor()])

    siameseDataset = SiameseDataset(transform=transform, device=device)
    train_ds, test_ds = random_split(siameseDataset, config.split)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=config.batch_size)   
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=10)
    
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
            if i%100==0:
                print(f'Epoch [{epoch + 1}/{config.epochs}], step: [{i}/{len(train_loader)}] Loss: {loss.item():.4f}')
          
        if (epoch+1)%5==0:
            torch.save(model.state_dict(), f'{project_dir}\\models\\6cv_1fc_lr0005_e{epoch+1}')             
                
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
        pred = F.pairwise_distance(out1[i], out2[i])
        plt.subplot(2,5,i+1)
        
        plt.xticks([])
        plt.yticks([])
        plt.title(f'{label_test[i].item()} CL: {pred.item():.4f}')
        plt.tight_layout()
        plt.imshow(pair)
          
          
def calculateConfusionMatrixAndThreshold(model, data_loader, criterion, threshold=1.0):
    with torch.no_grad():
        predicted = []
        actual = []
        histogramValues = []
        for i, (images1, images2, label) in enumerate(data_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            label = label.to(device)     
            model = model.to(device)
    
            out1, out2 = model(images1, images2)
            for i in range(images1.shape[0]):
                placeholder = F.pairwise_distance(out1[i], out2[i])
                actual.append(label[i].item())
                histogramValues.append(placeholder.item())
                if label[i].item() == 1.0:
                    if placeholder.item() > threshold:
                        predicted.append(1.0)
                    else:
                        predicted.append(0.0)
                elif label[i].item() == 0.0:
                    if placeholder.item() <= threshold:
                        predicted.append(0.0)
                    else:
                        predicted.append(1.0)
                              
            
        listToSetHistogramValues = set(histogramValues)
        uniqueHistogramValues = list(listToSetHistogramValues)

        cm = confusion_matrix(actual, predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.figure(0)
        plt.show()
        plt.figure(1)
        plt.hist(histogramValues, bins=len(uniqueHistogramValues))
        #plt.ylim(50)
        plt.show()


def model_pipeline(hyperparameters, wandb_mode = 'online'): 
    with wandb.init(project='fingerprint-authentication-ISU', config=hyperparameters, mode = wandb_mode):
        config = wandb.config
        
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)
        weights =  torch.load(f'{project_dir}/models/6cv_1fc_lr0001_douczka')
        model.load_state_dict(weights)
        #train_and_log(model, train_loader, test_loader, criterion, optimizer, config)
        test(model, test_loader, criterion, 'cpu')
        
        calculateConfusionMatrixAndThreshold(model, test_loader, criterion, 1.0)
        
    return model, train_loader, test_loader, criterion, optimizer

#%%
if __name__ == '__main__':
    # wandb_mode disabled for turn off logging
    model, train_loader, test_loader, criterion, _ = model_pipeline(config, wandb_mode='disabled')
#%%
if __name__ == '__main__':
    torch.save(model.state_dict(), f'{project_dir}\\models\\3c2fc_douczacz10')
    # model = Siamese_nn()
    # weights =  torch.load(f'{project_dir}/models/reduced_params')
    # model.load_state_dict(weights)
    # calculateConfusionMatrixAndThreshold(model, test_loader, criterion)
    # test(model, test_loader, criterion, 'cpu')