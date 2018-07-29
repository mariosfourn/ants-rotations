
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

import matplotlib
from scipy.ndimage.interpolation import rotate
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.models as models
from tensorboardX import SummaryWriter
from PIL import Image



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
         nn.init.xavier_normal_(m.weight)


def save_model(args,model,epoch):
    """
    saves a checkpoint so that model weight can later be used for inference
    Args:
    model:  pytorch model
    """
    path='./model_'+args.name
    import os
    if not os.path.exists(path):
      os.mkdir(path)
    torch.save(model.state_dict(), path+'/checkpoint_epoch_{}.pt'.format(epoch))




class AntsDataset(Dataset):
    """Ants Dataset"""

    def __init__(self, root_dir, csv_file, outsize, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with rotations
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rotations = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.outsize = outsize

    def __len__(self):
        return len(self.rotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.rotations.iloc[idx, 0])
        image = plt.imread(img_name,format='RGB')
        rotation = self.rotations.iloc[idx, 1].astype('float')

        ##Pad imaga to desired size
        lh_pad=(self.outsize-image.shape[0])//2
        rh_pad=self.outsize-image.shape[0]- lh_pad
        tv_pad=(self.outsize-image.shape[1])//2
        bv_pad=self.outsize-image.shape[1]-tv_pad

        #Filter, widht, height
        image=np.pad(image,((lh_pad,rh_pad),(tv_pad,bv_pad),(0,0)),'constant')

        if self.transform is not None:
            image=self.transform(image)

        return (image, rotation)


def map_to_circle(input):
    """
    Maps amgels with negative value to [0,1]
    """

    return (torch.max(input, 360+input))/360

def angle_error_regression(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a float number between 0 and 1.
    """
    return (angle_difference(y_true * 360, y_pred * 360)).mean()


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
 
    return 180 - torch.abs(torch.abs(x - y) - 180)



def evaluate_loss(args, model,dataloader):

    model.eval()
    #Store errors
    
    errors=np.zeros((args.test_batch_number*dataloader.batch_size,1))
    counter=0
    with torch.no_grad():
        
        for batch_idx, (data,rotations) in enumerate(dataloader):

            #Set data to device
            #data=data.to(device)
            #rotations=rotations.to(device)

            #conver negative anlge to positive
            rotations=map_to_circle(rotations).view(-1,1).float() #in range[0,1)

            # Forward pass
            output=model(data)
         
            errors[counter:counter+rotations.shape[0]]=angle_difference(360*output,360*rotations).cpu().numpy()


            counter+=rotations.shape[0]

            if batch_idx==args.test_batch_number-1: break

    mean_error=errors.mean()
    error_std=errors.std()

    return mean_error, error_std




class ResNet50_RotNet(nn.Module):
    """
    Autoencoder module for intepretable transformations
    """
    def __init__(self):
        super(ResNet50_RotNet, self).__init__()

    
        pretrained=models.resnet18(pretrained=True)

        #Replace last fc layer

        pretrained.fc=nn.Linear(512,1)

        self.network=nn.Sequential(pretrained,
            nn.Sigmoid())
        
    def forward(self, x):

        y=self.network(x)

        return y

def main():

    # Training settings
    list_of_choices=['Adam', 'SGD']
    parser = argparse.ArgumentParser(description='ResNet50 Regressor for Ants ')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-test-batch-size', type=int, default=100, metavar='N',
                        help='test batch size (default: 100)')
    parser.add_argument('-test-batch-number', type=int, default=10, metavar='N',
                        help='number of test batches (default: 10)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
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
    parser.add_argument('--name', type=str, default='',
                        help='name of the run that is added to the output directory')
    parser.add_argument('--optimizer', type=str, default='Adam', choices= list_of_choices,
                        help="Choose optimiser between 'Adam' (default) and 'SGD' with momentum")
    parser.add_argument('--lr-scheduler', action='store_true', default=False, 
                        help='set up lernaring rate scheduler (Default off)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to wait until learning rate is reduced in plateua')

    args = parser.parse_args()

    path = "./data" +args.name
    if not os.path.exists(path):
        os.makedirs(path)

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    #Test dataset
    test_root_dir='./ants1_dataset'
    test_rot_file='./ants1_dataset/ants1_rotations.csv'


    #Trainign dataset
    train_root_dir='./ants2_dataset'
    train_rot_file='./ants2_dataset/ants2_rotations.csv'

    #Torchvision transformation
    input_size=224
    transformations=transforms.Compose([transforms.ToTensor()])

    # Set up dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        AntsDataset(train_root_dir, train_rot_file,input_size,transform=transformations),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    train_loader_eval = torch.utils.data.DataLoader(
        AntsDataset(train_root_dir, train_rot_file,input_size,transform=transformations),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        AntsDataset(test_root_dir, test_rot_file,input_size,transform=transformations),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Init model and optimizer

    model = ResNet50_RotNet()

    if args.optimizer=='Adam':
        optimizer=optim.Adam(model.parameters(), lr=args.lr)

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.lr_scheduler:
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=args.patience)

    logging_dir='./logs_'+args.name

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    writer = SummaryWriter(logging_dir,comment='ResNet RotNet for Ants')

     # Where the magic happens
    sys.stdout.write('Start training\n')
    sys.stdout.flush()

    n_iter=0
    for epoch in range(1, args.epochs + 1):
        sys.stdout.write('Starting epoch {}/{} \n '.format(epoch,args.epochs))
        sys.stdout.flush()

        for batch_idx, (data,rotations) in enumerate(train_loader):
            model.train()

            #Set data to device
            #data=data.to(device)
            #rotations=rotations.to(device)

            #conver negative anlge to positive
            rotations=map_to_circle(rotations).view(-1,1).float()   #in the range [0,1]

            # Forward pass
            output=model(data)
            optimizer.zero_grad()

            #Loss
            
            loss=angle_error_regression(rotations,output)
            # Backprop
            loss.backward()
            optimizer.step()

            # Log training loss
            if batch_idx % args.log_interval==0:

                train_mean, train_std=evaluate_loss(args, model,train_loader_eval)
                test_mean, test_std= evaluate_loss(args, model,test_loader)
                writer.add_scalars('scalar_group',{'Train Loss':train_mean,
                                    'Train stddev': train_std,
                                     'Test Loss':  test_mean,
                                     'Test stddev':test_std}, n_iter)
            n_iter+=1

        sys.stdout.write('Ended epoch {}/{} and savign checkpoint \n '.format(epoch,args.epochs))
        sys.stdout.flush()

        save_model(args,model,epoch)
                

if __name__ == '__main__':
    main()







