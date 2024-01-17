#%%
"""
This module is responsible for architecturally defining Siamese Neural Network using PyTorch Framework.

Module consists of class definition for mentioned Neural Network.

Siamese Neural Network bases its functionality on having two images processed at the same time to produce
two vectors of similarity which then in turn are being passed to Contrastive Loss function, which in the end
indicates whether or not the samples were similar or dissimilar. 
"""
import torch.nn as nn
from torchsummary import summary

class Siamese_nn(nn.Module):
    """
    Siamese_nn class inherits nn.Module from PyTorch framework to create Sequential model.
 
    Class consists of following methods:
 
        - __init__(): responsible for creating featureExtractor that retrieves fingerprint features and
                      fc which has Fully Connected layer to create similarity vectors,
        - forwardOne(x): responsible for single action of forward pass in neural network,
        - forward(input1, input2): responsible for forward pass of two samples at the same time, due to nature of
                                   Siamese Network. 
    """

    def __init__(self):
        """
        Initializes two important components of Siamese Neural Network:
            - featureExtractor: responsible for analysing input images to extract features of fingerprint,
                                it is made of several Convolutional Layers with rising amount of features
                                extracted for increased precision,
            - fc: responsible for receiving vector of features and shrinks them to reasonably sized vector.
        """
        super(Siamese_nn, self).__init__()

        self.featureExtractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(57344, 512),
            nn.ReLU(inplace=True)
        )
        
    def forwardOne(self, x):
        """
        This method performs action of forward pass of single image. 
        
        It starts from extraction of features and then turning it to
        vector of features, which is returned as an output.
        """
        x = self.featureExtractor(x)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        """
        This method performs forward pass for dataset sample which includes a pair of images.
        
        It performs forward pass for both of the images at the same time and returns them as
        outputs for further training. 
        """
        output1 = self.forwardOne(input1)
        output2 = self.forwardOne(input2)
        return output1, output2

if __name__ == '__main__':
    model = Siamese_nn().to('cuda')
    summary(model, [(1, 153, 136), (1, 153, 136)], 12)
# %%
