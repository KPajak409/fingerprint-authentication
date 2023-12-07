#%%
import torch.nn as nn

class Siamese_nn(nn.Module):

    def __init__(self):
        super(Siamese_nn, self).__init__()

        self.featureExtractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(147456, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        
    def forwardOne(self, x):
        x = self.featureExtractor(x)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forwardOne(input1)
        output2 = self.forwardOne(input2)
        return output1, output2

if __name__ == '__main__':
    print('hello')
# %%
