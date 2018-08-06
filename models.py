import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Autoencoder(nn.Module):
    """
    Autoencoder module for intepretable transformations
    """
    def __init__(self,model_type,pretrained):
        super(Autoencoder, self).__init__()

        if model_type=='resnet18':
            pretrained=models.resnet18(pretrained=pretrained)
        elif model_type=='resnet34':
            pretrained=models.resnet34(pretrained=pretrained)
        elif model_type=='resnet34':
            pretrained=models.resnet50(pretrained=pretrained)


        #Replace maxpool layer with convolutional layers
        pretrained.maxpool=nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #Replace AvgPool2d witth AdaptiveAvgPool2d

        pretrained.avgpool=nn.AdaptiveAvgPool2d(1) 

        #Remove the last  fc layer anx call in encoder

        self.encoder= nn.Sequential(*list(pretrained.children())[:-1]) 

        self.decoder=Decoder()


    def forward(self, x,y,params):
        """
        Args:
            x:      untransformed images pytorch tensor
            y:      transforedm images  pytorch tensor
            params: rotations
        """
        #Encoder 
        f=self.encoder(x) #feature vector for original image
       
        f_theta=inverse_feature_transformer(self.encoder(y), params) # feature vector for tranformed image

        #Feature transform layer
        x=feature_transformer(f, params)

        #Decoder
        x=self.decoder(x)

        #Return reconstructed image, feature vector of oringial image, feature vector of transformation

        return x, f, f_theta


def feature_transformer(input, params):
    """Apply  rotation matrix every 2 dimensions

    Args:
        input: [N,c,1,1] or [N,c] tensor, where c = 2*int
        params: [N,1] tensor, with values in radians
    Returns:
        input-sized tensor
    """
    # First reshape activations into [N,c/2,2,1] matrices
    x = input.view(input.size(0),input.size(1)//2,2,1)
    # Construct the transformation matrix
    sin = torch.sin(params)
    cos = torch.cos(params)
    
    transform = torch.cat([cos, -sin, sin, cos], 1)
    transform = transform.view(transform.size(0),1,2,2)
    # Multiply: broadcasting taken care of automatically
    # [N,1,2,2] @ [N,channels/2,2,1]
    output = torch.matmul(transform, x)
    # Reshape and return
    return output.view(input.size())


def inverse_feature_transformer(input, params):
    """
    For now we assume the params are just a single rotation angle

    Args:
        input: [N,c,1,1] or [N,c] tensor, where c = 2*int
        params: [N,1] tensor, with values in randians
        device
    Returns:
        input-size tensor
    """
    # First reshape activations into [N,c/2,2,1] matrices
    x = input.view(input.size(0),input.size(1)//2,2,1)
    # Construct the transformation matrix
    sin = torch.sin(params)
    cos = torch.cos(params)
    #The inverse of a rotation matrix is its transpose

    transform = torch.cat([cos, sin, -sin, cos], 1)
    transform = transform.view(transform.size(0),1,2,2)
    # Multiply: broadcasting taken care of automatically
    # [N,1,2,2] @ [N,channels/2,2,1]
    output = torch.matmul(transform, x)
    # Reshape and return
    return output.view(input.size())


class NearestUsampling2D(nn.Module):
    def __init__(self,size):
        super(NearestUsampling2D, self).__init__()
        self.size=size #(tuple)

    def forward(self,input):
        return F.interpolate(input, size=self.size,mode='nearest')




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder=nn.Sequential(
            #2nd dconv layer
            nn.BatchNorm2d(512), # [N,512,1,1]
            NearestUsampling2D((2,2)), # [N,512,2,2]
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1), # [N,512,2,2]
            nn.BatchNorm2d(512),
            nn.RReLU(),

            #2nd dconv layer
            NearestUsampling2D((4,4)), # [N,512,4,4]
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1), # [N,256,4,4]
            nn.BatchNorm2d(256),
            nn.RReLU(),

            #3rd dconv layer
            NearestUsampling2D((7,7)), # [N,256,7,7]
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1), # [N,128,7,7]
            nn.BatchNorm2d(128),
            nn.RReLU(),

            #4th dconv layer
            NearestUsampling2D((13,13)), # [N,128,13,13]
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1), # [N,64,13,13]
            nn.BatchNorm2d(64),
            nn.RReLU(),

            #5th dconv layer
            NearestUsampling2D((25,25)), # [N,64,25,25]
            nn.Conv2d(64,32,kernel_size=5,stride=1,padding=1), # [N,32,23,23]
            nn.BatchNorm2d(32),
            nn.RReLU(),

            #6th dconv layer
            NearestUsampling2D((50,50)), # [N,32,50,50]
            nn.Conv2d(32,16,kernel_size=5,stride=1,padding=1),# [N,16,48,48]
            nn.BatchNorm2d(16),
            nn.RReLU(),

            #6th dconv layer
            NearestUsampling2D((102,102)), # [N,16,102,102]
            nn.Conv2d(16,3,kernel_size=5,stride=1,padding=1), # [N,3,100,100]
            nn.Sigmoid())

    def forward(self,x):
         return self.decoder(x)