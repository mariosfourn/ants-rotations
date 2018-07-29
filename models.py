import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvisio.models as models

class ResNet50_RotNet(nn.Module):
    """
    Autoencoder module for intepretable transformations
    """
    def __init__(self):
        super(ResNet50_RotNet, self).__init__()

        self.device = device

        self.pretrained=models.resnet18(pretrained=True)

        self.head=nn.Sequential()
        
    def forward(self, x, params):
        #Encoder 
        x=self.encoder(x)

        #Feature transform layer
        x=feature_transformer(x, params,self.device)

        #Decoder
        x=self.decoder(x)
          
        return x


