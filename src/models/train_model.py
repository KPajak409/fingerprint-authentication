#%%
"""
This module is responsible for following tasks:
    - Setting training parameters and splitting data to different DataLoaders,
    - Training neural network with Contrastive Loss function,
    - Testing results of training, calculating metrics and visualizing results,
    - Tracking training progress with "Weights and Biases" online service.
    
Module consists of ContrastiveLoss class for calculating loss during training and several
functions which perform mentioned tasks.
"""
import torch, wandb, os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.models.siamese_nn import Siamese_nn
from src.data.fingerprint_dataset import SiameseDataset
from pathlib import Path

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
    """
    ContrastiveLoss class uses nn.Module sourced from PyTorch framework to calculate loss for training process.
    
    Class consists of following methods:
        - __init__(): responsible for instantiating the class and sets default margin,
        - forward(output1, output2, label): calculates loss based on the contrastive operation
                                            (more details inside documentation of function).
    """
    def __init__(self, margin=1.0):
        """
        Method is responsible for initialization of the ContrastiveLoss object and sets default margin, which
        is a part of calculation of loss itself.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Method is responsible for calculating contrastive loss by calculating distance between features.
        
        Firstly, using pairwise_distance() function we calculate Euclidean distance between
        images and the output of distance is kept within same dimensions due to "keepdim" parameter.
        
        Secondly, by calculating mean of similarity and dissimilarity by following rules:
            - for dissimilar pairs, model is penalized if distance is too small,
            - for similar pairs, model is penalized if distance is too big than margin.
            
        Values closer to 0 are considered more similar and values further from zero are considered more dissimilar.
        """
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        lossContrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return lossContrastive


def make(config):
    """
    This method is responsible for preparation of training process and all important components.
    
    It does the following:
        - creates siamese dataset based on SiameseDataset class,
        - splits data to training dataset and testing dataset based on hyperparameter of split,
        - creates DataLoaders for both training and testing data,
        - sets model weights and optimization methods for training process.
    """
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
    """
    This method is responsible for training process of neural network and logging results to
    Weights and Biases online service.
    
    Performs the following:
        - retrieves data from training DataLoader,
        - uploads it to criterion to calculate loss,
        - updates optimizer and loss function,
        - logs data to Weights and Biases via specialized object "wandb" actions.
    """
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
    """
    This method is responsible for testing trained network on test DataLoader 
    and visualizes samples with their results.
    
    Function performs the following:
        - retrieves data from testing DataLoader,
        - calculates manually distance via pairwise_distance() function,
        - visualizes results as plot using matplotlib library.
    """
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
    """
    This method is responsible for calculating confusion matrix values for analyzing
    metric. Also calculates histogram which showcases the trend of values provided
    by testing our network.
    
    This method does the following:
        - retrieves data from all available samples,
        - calculates distance for every pair,
        - appends results to prediction and label array,
        - removes duplicate values from value history for histogram,
        - visualizes confusion matrix and histogram.
    """
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

        plt.rcParams.update({'font.size': 12})
        cm = confusion_matrix(actual, predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.figure(0)
        plt.show()
        plt.figure(1)
        plt.hist(histogramValues, bins=len(uniqueHistogramValues))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        #plt.ylim(50)
        plt.show()


def model_pipeline(hyperparameters, wandb_mode = 'disabled'): 
    """
    This method is responsible for triggering all of the other methods in set order.
    
    The order of methods triggered is:
        1. make(config)
        2. train_and_log(model, train_loader, test_loader, criterion, optimizer, config)
        3. test(model, test_loader, criterion, 'cpu')
        4. calculateConfusionMatrixAndThreshold(model, test_loader, criterion, 1.0)
        
    If we want to load a pre-trained model, then we must do the following in model_pipeline() function:
        1. comment train_and_log(model, train_loader, test_loader, criterion, optimizer, config)
        2. uncomment lines:
            - weights =  torch.load(f'{project_dir}/models/6cv_1fc_lr0001_douczka'),
            - model.load_state_dict(weights).
    """
    with wandb.init(project='fingerprint-authentication-ISU', config=hyperparameters, mode = wandb_mode):
        config = wandb.config
        
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)
        weights =  torch.load(f'{project_dir}/models/trained_model')
        model.load_state_dict(weights)
        #train_and_log(model, train_loader, test_loader, criterion, optimizer, config)
        test(model, test_loader, criterion, 'cpu')
        
        calculateConfusionMatrixAndThreshold(model, test_loader, criterion, 0.5)
        
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