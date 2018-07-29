from __future__ import print_function
import os
import sys
import time

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pandas as pd
import struct
from tensorboardX import SummaryWriter
import matplotlib
from scipy.ndimage.interpolation import rotate
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from model import Net_Reg


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
         nn.init.xavier_normal_(m.weight)


def save_model(args,model):
    """
    saves a checkpoint so that model weight can later be used for inference
    Args:
    model:  pytorch model
    """
    path='./model_'+args.name
    import os
    if not os.path.exists(path):
      os.mkdir(path)
    torch.save(model.state_dict(), path+'/checkpoint.pt')




def round_even(x):
    return int(round(x/2.)*2)



def rotate_tensor(input,rotation_range):
    """Nasty hack to rotate images in a minibatch, this should be parallelized
    and set in PyTorch

    Args:
        input: [N,c,h,w] **numpy** tensor
    Returns:
        rotated output and angles in radians
    """
    angles = rotation_range * np.random.rand(input.shape[0])
    angles = angles.astype(np.float32)
    outputs = []
    for i in range(input.shape[0]):
        output = rotate(input[i,...], 180*angles[i]/np.pi, axes=(1,2), reshape=False)
        outputs.append(output)
    return np.stack(outputs, 0), angles




class Penalty_Loss(nn.Module):
    """
    Penalty loss on feature vector to ensure that in encodes rotation information
    """
    
    def __init__(self,proportion, size_average=False,type='mse'):
        super(Penalty_Loss,self).__init__()
        self.size_average=size_average #flag for mena loss
        self.proportion=proportion     #proportion of feature vector to be penalised
        self.type=type
        
    def forward(self,x,y):
        """
        penalty loss bases on cosine similarity being 1

        Args:
            x: [batch,1,ndims]
            y: [batch,1,ndims]
        """
        x=x.view(x.shape[0],-1)
        y=y.view(y.shape[0],-1)
        #Number of features
        total_dims=x.shape[1]
        #Batch size
        batch_size=x.shape[0]

        #Number of features penalised
        ndims=round_even(self.proportion*total_dims)
        reg_loss=0.0

        for i in range(0,ndims-1,2):
            x_i=x[:,i:i+2]
            y_i=y[:,i:i+2]
            dot_prod=torch.bmm(x_i.view(batch_size,1,2),y_i.view(batch_size,2,1)).view(batch_size,1)
            x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
            y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)

            if type=='mse':
                reg_loss+=((dot_prod/(x_norm*y_norm)-1)**2).sum()
            else:
                reg_loss+=(abs(dot_prod/(x_norm*y_norm)-1)).sum()
                
        if self.size_average:
            reg_loss=reg_loss/x.shape[0]/(ndims//2)
        return reg_loss



def penalised_loss(args,output,targets,f_data,f_targets):
    """
    Define penalised loss
    """

    L1_loss=torch.nn.L1Loss(size_average=True)
    loss_reg = Penalty_Loss(size_average=True,proportion=args.prop,type=args.loss)
    #Add 
    reconstruction_loss=L1_loss(output,targets)
    rotation_loss=loss_reg(f_data,f_targets)
    total_loss= (1-args.alpha)*reconstruction_loss+args.alpha*rotation_loss
    return total_loss,reconstruction_loss,rotation_loss



class AntsDataset(Dataset):
    """Ants Dataset"""

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with rotations
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rotations = pd.read_csv(txt_file, sep=',', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.rotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.rotations.iloc[idx, 0])
        image = io.imread(img_name)
        rotation = self.rotations.iloc[idx, 1].astype('float')
        sample = {'image': image, 'rotation': rotation}

        if self.transform:
            sample = self.transform(sample)

        return sample



ef rotation_test(args, model, device, test_loader):
    """
    Test how well the model discrimantes roations between 2 samples from the dataset 
    """
    model.eval()
    with torch.no_grad():
        for data, rotations in test_loader:
            #Compare every 2 imgaes since they are suffled

            #Split the data in 2 
            x_data=data[:data.shape[0]//2]
            y_data=data[data.shape[0]//2:]

            rotations_x=rotations[:data.shape[0]//2]
            rotations_y=rotations[data.shape[0]//2:]

            #Get reative rotation
            angles=rotations_y-

            assert x_data.shape[0]==y_data.shape[0] , 'the batch size for rotation test must be even'
    
            #Get Feature vector for original and tranformed image
            
            x=model.encoder(x_data) #Feature vector of data
            y=mode.encoder(y_data)

            #Compare Angles            
            x=x.view(x.shape[0],-1) # collapse 3D tensor to 2D tensor 
            y=y.view(y.shape[0],-1) # collapse 3D tensor to 2D tensor
           

            #Number of features
            total_dims=x.shape[1]
            #Batch size
            batch_size=x.shape[0]
            angles_estimate=torch.zeros(batch_size,1).to(device)  

            #Number of features penalised
            ndims=round_even(args.prop*total_dims)  
            #Loop every 2 dimensions
            for i in range(0,ndims-1,2):
                x_i=x[:,i:i+2]      
                y_i=y[:,i:i+2]
                #Get dor product for the batcg
                dot_prod=torch.bmm(x_i.view(batch_size,1,2),y_i.view(batch_size,2,1)).view(batch_size,1)

                #Get euclidean norm
                x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
                y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)

                #Get the cosine of the angel for example
                angles_estimate+=dot_prod/(x_norm*y_norm)

            angles_estimate=torch.acos(angles_estimate/(ndims//2))*180/np.pi # average and in degrees
            angles_estimate=angles_estimate.cpu()
            error=angles_estimate.numpy()-(angles.cpu().numpy()*180/np.pi)
            average_error=abs(error).mean()
            error_std=error.std(ddof=1)

            break
    return average_error,error_std



def main():

    # Training settings
    parser = argparse.ArgumentParser(description='ResNet AutoEncoder for Ants ')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size-recon', type=int, default=10, metavar='N',
                        help='input batch size for reconstruction testing (default: 10)')
    parser.add_argument('--test-batch-eval', type=int, default=100, metavar='N',
                        help='input batch size for rotation disrcimination testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--store-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before storing training loss')
    parser.add_argument('--alpha',type=float, default=0.5, metavar='Lambda',
                        help='proportion of penalty loss of the total loss (default=0.5)')
    parser.add_argument('--name', type=str, default='',
                        help='name of the run that is added to the output directory')
    parser.add_argument('--prop',type=float, default=1.0,
                        help='proportion of feature vector with penalty loss (Default=1.0)')
    parser.add_argument('--rotation-range',type=float, default=90,
                        help='Range of relative rotations (Default=90)')


    args = parser.parse_args()

    # Create save path
    args.rotation_range=args.rotation_range*np.pi/180

    path = "./output_" +args.name
    if not os.path.exists(path):
        os.makedirs(path)

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    root_dir_Dataset='../AntsDataset'
    rotation_file='./AntsDataset/ants_rotations.csv'

    # Set up dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        AntsDataset(root_dir_Dataset, rotation_file,transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    train_loader = torch.utils.data.DataLoader(
        AntsDataset(root_dir_Dataset, rotation_file,transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_eval, shuffle=True, **kwargs)

    # Init model and optimizer
    model = Net_Reg(device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(comment='ResNet AutoEncoder for Ants')

     # Where the magic happens
    sys.stdout.write('Start training\n')
    sys.stdout.flush()

    n_iter=0
      for epoch in range(1, args.epochs + 1):
        sys.stdout.write('Epoch {}/{} \n '.format(epoch,args.epochs))
        sys.stdout.flush()

        for batch_idx, (data,_) in enumerate(train_loader):
            model.train()

            targets, angles= rotate_tensor(data,args.rotation_range)
            data=data.to(device)
            targets = torch.from_numpy(targets).to(device)
            angles = torch.from_numpy(angles).to(device)
            angles = angles.view(angles.size(0), 1)

            # Forward pass
            optimizer.zero_grad()
            #(output of autoencoder, feature vector of input, feature vector of rotated data)
            output, f_data, f_targets = model(data, targets,angles) 

            #Loss
            loss,reconstruction_loss,penalty_loss=penalised_loss(args,output,targets,f_data,f_targets)

            # Backprop
            loss.backward()
            optimizer.step()

            # Log training loss
            if batch_idx % args.log_interval==0:
                writer.add_scalar('Reconstruction Training Loss', reconstruction_loss.item(), n_iter)
                writer.add_scalar('Penatly Training Loss', penalty_loss.item(), n_iter )
                writer.add_scalar('Total Training Loss',loss.item(), n_iter)

            #Run rotation evalutation test on subset of training data 
            if batch_idx % args.store_interval==0:






