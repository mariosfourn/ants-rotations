
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

    def __init__(self, root_dir, csv_file, transform=None):
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

    def __len__(self):
        return len(self.rotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.rotations.iloc[idx, 0])
        image = plt.imread(img_name,format='RGB')
        rotation = self.rotations.iloc[idx, 1].astype('float')

        if self.transform is not None:
            image=self.transform(image)

        return (image, rotation)


def map_to_circle(input):
    """
    Maps amgels with negative value to [0,1]
    """

    input[input<0]=input[input<0]+360
    
    return input/360


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
    
    errors=np.zeros((len(dataloader.dataset)*5,1))
    counter=0
    with torch.no_grad():
        
        for batch_idx, (data,rotations) in enumerate(dataloader):
            #data is a 5d tensor, target is 1D
  
            bs, ncrops, c, h, w = data.size()
            

            rotations=map_to_circle(rotations.view(-1,1).repeat(1,5).view(-1,1)).float() #in range[0,1)

            # Forward pass
            output=model(data.view(-1,c,h,w)) # fuse batch size and ncrops

            errors[counter:counter+rotations.shape[0]]=angle_difference(360*output,360*rotations).cpu().numpy()


            counter+=rotations.shape[0]

            # if batch_idx==args.test_batch_number-1: break

    mean_error=errors.mean()
    error_std=errors.std()

    return mean_error, error_std




class RotNet(nn.Module):
    """
    Autoencoder module for intepretable transformations
    """
    def __init__(self,model_type):
        super(RotNet, self).__init__()

        if model_type=='resnet18':
            pretrained=models.resnet18(pretrained=True)
        elif model_type=='resnet34':
            pretrained=models.resnet34(pretrained=True)
        elif model_type=='resnet34':
            pretrained=models.resnet50(pretrained=True)

        #Add Batchnorm 2d at the beginnign instead of nornalising image

        #Replace average pooling with adaptive pooling

        pretrained.avgpool=nn.AdaptiveAvgPool2d(1)

        # #Replace last fc layer
        pretrained.fc=nn.Sequential(nn.Linear(512,256),
                                     nn.Linear(256,1))
        #pretrained.fc=nn.Linear(512,1)


        self.network=nn.Sequential(
            nn.BatchNorm2d(3),
            pretrained,
            nn.Sigmoid())
        
    def forward(self, x):

        y=self.network(x)

        return y


def centre_crop(img, cropx, cropy):
    c,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]


def rotate_tensor(args,input):
    """
    Roteates images by reflect padding, rotating and the  cropping
    Args:
        input: [N,c,h,w] tensor
    Returns:
        rotated torch tensor and angels in degrees
    """

    #First apply reflection pad
    vertical_pad=input.shape[-1]//2
    horizontal_pad=input.shape[-2]//2
    pad2D=(horizontal_pad,horizontal_pad,vertical_pad,vertical_pad)
    padded_input=F.pad(input,pad2D,mode='reflect')

    angles = args.random_rotation_range*np.random.uniform(-1,1,input.shape[0])

    angles = angles.astype(np.float32)

    outputs = []

    for i in range(input.shape[0]):
        output = rotate(padded_input.numpy()[i,...], angles[i], axes=(1,2), reshape=False)
        output=centre_crop(output, args.random_crop_size,args.random_crop_size)
        outputs.append(output)
    

    outputs=np.stack(outputs, 0)


    return torch.from_numpy(outputs), torch.from_numpy(angles)

def main():

    # Training settings
    list_of_choices=['Adam', 'SGD']
    list_of_models=['resnet18,resnet34,resnet50']
    parser = argparse.ArgumentParser(description='ResNet50 Regressor for Ants ')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-test-batch-size', type=int, default=20, metavar='N',
                        help='test batch size (default: 20)')
    parser.add_argument('-test-batch-number', type=int, default=10, metavar='N',
                        help='number of test batches (default: 10)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='',
                        help='name of the run that is added to the output directory')
    parser.add_argument('--optimizer', type=str, default='Adam', choices= list_of_choices,
                        help="Choose optimiser between 'Adam' (default) and 'SGD' with momentum")
    parser.add_argument('--lr-scheduler', action='store_true', default=False, 
                        help='set up lernaring rate scheduler (Default off)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of epochs to wait until learning rate is reduced in plateua (default=3)')
    parser.add_argument('--print-progress', action='store_true', default=False,
                        help='print the progress on screen, Recommended for AWS')
    parser.add_argument('--resnet-type', type=str, default='resnet18', choices= list_of_models,
                        help='choose resnet type [resnet18,resnet34,resnet50] (default=resnet18)')
    parser.add_argument('--image-resize', type=int, default=120,
                        help='size for resizing input image')
    parser.add_argument('--random-crop-size', type=int, default=100,
                        help='random crop image size in pixel')
    parser.add_argument('--brightness', type=float, default=0.2,
                        help='brightness factor for ColorJitter augmentation')
    parser.add_argument('--contrast', type=float, default=0.2,
                        help='contrast factor for ColorJitter augmentation')
    parser.add_argument('--saturation', type=float, default=0.1,
                        help='saturation factor for ColorJitter augmentation')
    parser.add_argument('--hue', type=float, default=0.07,
                        help='hue factor for ColorJitter augmentation')
    parser.add_argument('--random-rotation-range', type=float, default=30, metavar='theta',
                        help='random rotation range in degrees for training [-theta,+theta)')
    parser.add_argument('--save', type=int, default=5, metavar='N',
                        help='save model every this number of epochs')
    parser.add_argument('--amsgrad', action='store_true', default=False, 
                        help='Turn on amsgrad in Adam optimiser')
    parser.add_argument('--rot-augment', action='store_true', default=False, 
                        help='Augment Rotations')


    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    print('Amgrad set to {}\n'.format(args.amsgrad))
    print('Rotation augmentation set to {}\n'.format(args.rot_augment))


    torch.manual_seed(args.seed)


    #1st sequnce of ants
    ants1_root_dir='./ants1_dataset_ratio3'
    ants1_rot_file='./ants1_dataset_ratio3/ants1_rotations.csv'


    #2nd sequnece of ants
    ants2_root_dir='./ants2_dataset_ratio3'
    ants2_rot_file='./ants2_dataset_ratio3/ants2_rotations.csv'

    data_root_dir='./ants_dataset_ratio3_combined'
    train_rot_dir='./ants_dataset_ratio3_combined/train_rotations.csv'
    test_rot_dir='./ants_dataset_ratio3_combined/test_rotations.csv'

    #Torchvision transformation
    train_transformations=transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((args.image_resize,args.image_resize)),
        transforms.RandomCrop(size=args.random_crop_size, pad_if_needed=True),
        transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
        transforms.ToTensor()])

    eval_transformations=transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((args.image_resize,args.image_resize)),
        transforms.FiveCrop(args.random_crop_size),
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ]) 
    #Apply tranformtations

    train_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,train_rot_dir,transform=train_transformations),
        batch_size=args.batch_size, shuffle=True)

    # train_loader_eval = torch.utils.data.DataLoader(train_dataset
    #     batch_size=args.test_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,test_rot_dir,transform=eval_transformations),
        batch_size=args.test_batch_size, shuffle=False)

    # Init model and optimizer

    model = RotNet(args.resnet_type)

    #Estimate memoery usage

    if args.optimizer=='Adam':
        optimizer=optim.Adam(model.parameters(), lr=args.lr,amsgrad=args.amsgrad)

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.lr_scheduler:
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=args.patience, threshold=0.1,verbose=True)

    logging_dir='./logs_'+args.name

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    writer = SummaryWriter(logging_dir, comment='ResNet RotNet for Ants')

     # Where the magic happens
    sys.stdout.write('Start training\n')
    sys.stdout.flush()

    n_iter=0
    for epoch in range(1, args.epochs + 1):
        sys.stdout.write('Starting epoch {}/{} \n '.format(epoch,args.epochs))
        sys.stdout.flush()

        for batch_idx, (data,rotations) in enumerate(train_loader):
            model.train()

            #conver negative anlge to positive
            if args.rot_augment:
                data,random_rotation=rotate_tensor(args,data) 

                rotations=rotations.float()+random_rotation

            rotations=map_to_circle(rotations).view(-1,1).float()   #in the range [0,1]8

            # Forward pass
            output=model(data)
            optimizer.zero_grad()

            #Loss
            
            loss=angle_error_regression(rotations,output)
            # Backprop
            loss.backward()
            optimizer.step()

            writer.add_scalar('Mini-batch train loss',loss.item(),n_iter)

            if args.print_progress:

                sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                sys.stdout.flush()

            
            n_iter+=1


        #train_mean, train_std=evaluate_loss(args, model,train_loader_eval)
        test_mean, test_std= evaluate_loss(args, model,test_loader)
        writer.add_scalars('scalar_group',{'Test Loss':  test_mean,
                                     'Test stddev':test_std}, epoch)

        sys.stdout.write('Ended epoch {}/{} \n '.format(epoch,args.epochs))
        sys.stdout.flush()

        if args.lr_scheduler:
            scheduler.step(test_mean)

        if epoch%args.save==0:
            save_model(args,model,epoch)
                

if __name__ == '__main__':
    main()







