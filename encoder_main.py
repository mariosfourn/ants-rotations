
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
import random
import itertools
import pytorch_ssim

#import matplotlib
from scipy.ndimage.interpolation import rotate
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.models as models
from tensorboardX import SummaryWriter
from PIL import Image





class Encoder(nn.Module):
    """
    Encoder to 2-dimnesional space
    """
    def __init__(self,model_type):
        super(Encoder, self).__init__()

        if model_type=='resnet18':
            pretrained=models.resnet18(pretrained=True)
        elif model_type=='resnet34':
            pretrained=models.resnet34(pretrained=True)
        elif model_type=='resnet50':
            pretrained=models.resnet50(pretrained=True)


        #Replace maxpool layer with convolutional layers
        pretrained.maxpool=nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #Replace AvgPool2d witth AdaptiveAvgPool2d

        pretrained.avgpool=nn.AdaptiveAvgPool2d(1) 

        #Remove the last  fc layer anx call in encoder

        self.encoder= nn.Sequential(*list(pretrained.children())[:-1], 
                     nn.Conv2d(512,2,1)) 

    def forward(self,x):
        return self.encoder(x)

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


def convert_to_convetion(input):
    """
    Coverts all anlges to convecntion used by atan2
    """

    input[input<180]=input[input<180]+360
    input[input>180]=input[input>180]-360
    
    return input

def evaluate_rot_loss(args, model,dataloader,writer,epoch):

    model.eval()

    absolute_angles=np.zeros((len(dataloader.dataset),1))
    counter=0

    with torch.no_grad():
        
        for batch_idx, (data,rotations) in enumerate(dataloader):
   
            f_data=model(data)

            f_data_y= f_data[:,1] #Extract y coordinates
            f_data_x= f_data[:,0] #Extract x coordinates

            absolute_angles[counter:counter+data.shape[0]]=torch.atan2(f_data_y,f_data_x).cpu().numpy().reshape(-1,1)*180/np.pi #Calculate absotulue angel of vectoe

            counter+=data.shape[0]
    
    #Get list of dataset absolute rotations from then dataset
    rotations=np.array(dataloader.dataset.rotations[1]).astype(float).reshape(-1,1)

    #Sample 1000 pairs of indices
    length=len(dataloader.dataset)

    idx_samples=np.array(random.sample(list(itertools.product(range(length),range(length))),args.samples))

    #Get the difference in the rotation adn covert to range [-180,180]

    rotation_difference=convert_to_convetion(rotations[idx_samples[:,1]]-rotations[idx_samples[:,0]])

    #Exclude differnce beyond  the specified limits

    valid_idx_samples=idx_samples[(abs(rotation_difference)<=args.rotation_range).flatten()]
    valid_rotation_difference=rotation_difference[abs(rotation_difference)<=args.rotation_range]
 
    estimated_rotation=convert_to_convetion(absolute_angles[valid_idx_samples[:,1]]-absolute_angles[valid_idx_samples[:,0]])

    error=estimated_rotation-valid_rotation_difference

    writer.add_histogram('Error Histogram', error, epoch)
    
    mean_error = abs(error).mean()
    error_std = error.std(ddof=1)
    
    return mean_error, error_std


def define_loss(args, x,y):
    """
    Return the loss based on the user's arguments

    Args:
        x:  [N,2,1,1]    output of encoder model
        y:  [N,2,1,1]    output of encode model
    """

    if args.loss=='forbenius':
        forb_distance=torch.nn.PairwiseDistance()
        x_polar=x.view(-1,2)
        x_polar=x/x.norm(p=2,dim=1,keepdim=True)
        y_polar=y.view(-1,2)
        y_polar=y/y.norm(p=2,dim=1,keepdim=True)
        loss=(forb_distance(x_polar,y_polar)**2).mean()

    elif args.loss=='cosine_mse':

        cosine_similarity=nn.CosineSimilarity(dim=2)
        loss=((cosine_similarity(x.view(x.size(0),1,2),y.view(y.size(0),1,2))-1.0)**2).mean()

    elif args.loss=='cosine_abs':

        cosine_similarity=nn.CosineSimilarity(dim=2)
        loss=torch.abs(cosine_similarity(x.view(x.size(0),1,2),y.view(y.size(0),1,2))-1.0).mean()

    return loss


def centre_crop(img, cropx, cropy):
    c,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]


def rotate_tensor(args,input,plot=False):
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

    angles = args.rotation_range*np.random.uniform(-1,1,input.shape[0])

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
    list_of_losses=['cosine_abs','cosine_mse','forbenius']
    list_of_choices=['Adam', 'SGD']
    list_of_models=['resnet18,resnet34,resnet50']
    parser = argparse.ArgumentParser(description='ResNet50 Regressor for Ants ')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-eval-batch-size', type=int, default=100, metavar='N',
                        help='eval batch size (default: 100)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
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
                        help='size for resizing input image (Default=120)')
    parser.add_argument('--random-crop-size', type=int, default=100,
                        help='random crop image size in pixel (Default=100)')
    parser.add_argument('--brightness', type=float, default=0,
                        help='brightness factor for ColorJitter augmentation')
    parser.add_argument('--contrast', type=float, default=0,
                        help='contrast factor for ColorJitter augmentation')
    parser.add_argument('--saturation', type=float, default=0,
                        help='saturation factor for ColorJitter augmentation')
    parser.add_argument('--hue', type=float, default=0,
                        help='hue factor for ColorJitter augmentation')
    parser.add_argument('--rotation-range', type=float, default=180, metavar='theta',
                        help='rotation range in degrees for training,(Default=180), [-theta,+theta)')
    parser.add_argument('--amsgrad', action='store_true', default=False, 
                        help='Turn on amsgrad in Adam optimiser')
    parser.add_argument('--save', type=int, default=5, metavar='N',
                        help='save model every this number of epochs (Default=5)')
    parser.add_argument('--loss', type=str, default='cosine_abs', choices= list_of_losses,
                        help="type of loss for atan2 penalty loss [abs,mse,forbenius] (Default= abs)")
    parser.add_argument('--samples', type=int, default=1000, metavar='N',
                        help='No of test samples (Default=1,000)')
    parser.add_argument('--threshold', type=float, default=1e-4, metavar='l',
                        help='ReduceLROnPlateau signifance threshold (Default=1e-4)')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    #Print arguments
    for arg in vars(args):
        sys.stdout.write('{} = {} \n'.format(arg,  getattr(args, arg)))
        sys.stdout.flush()
    sys.stdout.write('Random torch seed:{}\n'.format( torch.initial_seed()))
    sys.stdout.flush()

    #torch.manual_seed(args.seed)

    ImageNet_mean=[0.485, 0.456, 0.406]
    ImageNet_STD=[0.229, 0.224, 0.225]


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
        transforms.CenterCrop(size=args.random_crop_size),
        #transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
        transforms.ToTensor()])

    eval_transformations=transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((args.image_resize,args.image_resize)),
        transforms.CenterCrop(size=args.random_crop_size),
        #transforms.FiveCrop(args.random_crop_size),
        #(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))

        transforms.ToTensor()]) 
    #Apply tranformtations

    train_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,train_rot_dir,transform=train_transformations),
        batch_size=args.batch_size, shuffle=True)

    train_loader_eval = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,train_rot_dir,transform=train_transformations),
        batch_size=args.eval_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,test_rot_dir,transform=eval_transformations),
        batch_size=args.eval_batch_size, shuffle=False)

    # Init model and optimizer

    model = Encoder(args.resnet_type)

    #Estimate memoery usage

    if args.optimizer=='Adam':
        optimizer=optim.Adam(model.parameters(), lr=args.lr,amsgrad=args.amsgrad)

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.lr_scheduler:
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=args.patience,verbose=True,threshold=args.threshold)

    logging_dir='./logs_'+args.name

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    writer = SummaryWriter(logging_dir, comment='Encoder for ants')

    
    sys.stdout.write('Start training\n')
    sys.stdout.flush()

    test_error_mean_log=[]
    test_error_std_log=[]

    n_iter=0
    for epoch in range(1, args.epochs + 1):
        sys.stdout.write('Starting epoch {}/{} \n '.format(epoch,args.epochs))
        sys.stdout.flush()

        for batch_idx, (data,rotations) in enumerate(train_loader):
            model.train()

            targets,relative_rotations=rotate_tensor(args,data) 
            relative_rotations=relative_rotations.view(-1,1)

            # Forward pass
            f_data=model(data)
            f_targets=model(targets)
            f_data_trasformed = feature_transformer(f_data,relative_rotations*np.pi/180)
          
            optimizer.zero_grad()


            #Define loss

            loss=define_loss(args,f_data_trasformed,f_targets)

            # Backprop
            loss.backward()
            optimizer.step()

            writer.add_scalar('Mini-batch train loss',  loss.item(),n_iter)
            #                         
            if args.print_progress:

                sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                sys.stdout.flush()

            n_iter+=1


        #train_loss=evaluate_loss(args,model,train_loader_eval)
        #sys.stdout.write('Ended epoch {}/{}, Train ={:4f}\n '.format(epoch,args.epochs,train_loss))
        #sys.stdout.flush()



        test_mean, test_std=evaluate_rot_loss(args,model,test_loader,writer,epoch)

        test_error_mean_log.append(test_mean)
        test_error_std_log.append(test_std)

        if args.lr_scheduler:
            if args.scheduler_loss=='test':
                scheduler.step(test_mean)
            elif args.scheduler_loss=='train':
                scheduler.step(train_loss)
            else:
                print('Wrong Loss Type')
                break

        if epoch%args.save==0:
            save_model(args,model,epoch)

    plot_error(args,np.array(test_error_mean_log),np.array(test_std),logging_dir)


def plot_error(args,average_error,error_std,path):
    """
    Plots error
    """

    with plt.style.context('ggplot'):

        fig, ax =plt.subplots()

        line,=ax.plot(average_error,label='Mean Abs Tets Error',linewidth=1.25,color='g')
        ax.fill_between(range(len(average_error)),average_error-error_std,average_error+error_std,
            alpha=0.2,facecolor=line.get_color(),edgecolor=line.get_color())
        ax.set_ylabel('Degrees',fontsize=10)
        ax.set_xlabel('Epochs',fontsize=10)
        ax.set_xlim(0,None)
        ax.set_ylim(-10,20)
 


        #Control colour of ticks
        ax.tick_params(colors='gray', direction='out')
        for tick in ax.get_xticklabels():
            tick.set_color('gray')
        for tick in ax.get_yticklabels():
            tick.set_color('gray')

        fig.tight_layout()
        fig.savefig(path+'/Test_error')
        fig.clf()



                

if __name__ == '__main__':
    main()







