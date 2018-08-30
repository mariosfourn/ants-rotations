
from __future__ import print_function, division
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
import ipdb



class Encoder(nn.Module):
    """
    Encoder to 2-dimnesional space
    """
    def __init__(self,model_type,pretrained):
        super(Encoder, self).__init__()

        if model_type=='resnet18':
            pretrained=models.resnet18(pretrained=pretrained)
        elif model_type=='resnet34':
            pretrained=models.resnet34(pretrained=pretrained)
        elif model_type=='resnet50':
            pretrained=models.resnet50(pretrained=pretrained)


        #Replace maxpool layer with convolutional layers
        pretrained.maxpool=nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #Replace AvgPool2d witth AdaptiveAvgPool2d

        pretrained.avgpool=nn.AdaptiveAvgPool2d(1) 

        #Remove the last  fc layer anx call in encoder

        self.encoder= nn.Sequential(*list(pretrained.children())[:-1], 
                     nn.ReLU(),
                     nn.Conv2d(512,256,1),
                     nn.ReLU(),
                     nn.Conv2d(256,2,1))

    def forward(self,x):
        return self.encoder(x)


def eval_synthetic_rot_loss(args,model,data_loader):

    model.eval()
    #Number of features penalised
    error=np.zeros(len(data_loader.dataset))

    start_index=0

    with torch.no_grad():
        for data,_ in data_loader:
            ## Reshape data
            data=data[:,4,...]
         
            targets,rotations=rotate_tensor(args,data,args.eval_rotation_range)

            # Forward passes
            f_data=model(data) # [N,2,1,1]
            f_targets=model(targets) #[N,2,1,1]

            f_data=f_data.view(-1,2)  # [N,2]
            f_targets=f_targets.view(-1,2) #[N,2]

            f_data_y= f_data[:,1] #Extract y coordinates
            f_data_x= f_data[:,0] #Extract x coordinates

            f_targets_y= f_targets[:,1] #Extract y coordinates
            f_targets_x= f_targets[:,0] #Extract x coordinates

            theta_data=torch.atan2(f_data_y,f_data_x).cpu().numpy()*180/np.pi #Calculate absotulue angel of vector
            theta_targets=torch.atan2(f_targets_y,f_targets_x).cpu().numpy()*180/np.pi #Calculate absotulue angel of vector            
       
            estimated_angle=theta_targets-theta_data
            
            estimated_angle=convert_to_convetion(estimated_angle)

            error[start_index:start_index+data.shape[0]]=convert_to_convetion(estimated_angle-rotations.numpy())
          
            start_index+=data.shape[0]
           
    return  abs(error).mean(), error.std()


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

def evaluate_rot_loss(args, model,dataloader):

    model.eval()

    absolute_angles=np.zeros((len(dataloader.dataset),1))
    rotations=np.zeros((len(dataloader.dataset),1))
    counter=0

    with torch.no_grad():
        
        for batch_idx, (data,batch_rotations) in enumerate(dataloader):

            bs, ncrops, c, h, w = data.size()
   
            f_data=model(data.view(-1,c,h,w))

            #f_data=model(data)

            #Average results from 5 crops

            f_data_avg = f_data.view(bs, ncrops, -1).mean(1)

            #f_data_avg=f_data

            f_data_y= f_data_avg[:,1] #Extract y coordinates
            f_data_x= f_data_avg[:,0] #Extract x coordinates

            batch_size=batch_rotations.shape[0]

            absolute_angles[counter:counter+batch_size]=torch.atan2(f_data_y,f_data_x).cpu().numpy().reshape(-1,1)*180/np.pi #Calculate absotulue angel of vectoe
            rotations[counter:counter+batch_size]=batch_rotations.reshape(-1,1)
            counter+=batch_size

    length=rotations.shape[0]


    #Get all possible combinations from the test dataset
    idx_samples=np.array(list(itertools.product(range(length),range(length))))

    # idx_samples=np.array(random.sample(list(itertools.product(range(length),range(length))),args.samples))

    rotation_difference=convert_to_convetion(rotations[idx_samples[:,1]]-rotations[idx_samples[:,0]])

    valid_idx_samples=idx_samples[(abs(rotation_difference)<=args.eval_rotation_range).flatten()]

    valid_rotation_difference=rotation_difference[(abs(rotation_difference)<=args.eval_rotation_range).flatten()].reshape(-1,1)
 
    estimated_rotation=convert_to_convetion(absolute_angles[valid_idx_samples[:,1]]-absolute_angles[valid_idx_samples[:,0]])

    error=convert_to_convetion(estimated_rotation-valid_rotation_difference)   
    
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


def random_crop(img, cropx, cropy,offset):
    c,y,x = img.shape
    startx = x//2-(cropx//2) + int(np.random.uniform(offset))*(-1)**(np.random.binomial(1, p=0.5))
    starty = y//2-(cropy//2) + int(np.random.uniform(offset))*(-1)**(np.random.binomial(1, p=0.5))
    return img[:,starty:starty+cropy,startx:startx+cropx]


def rotate_tensor(args,input,rotation_range):
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

    angles = rotation_range*np.random.uniform(-1,1,input.shape[0])

    angles = angles.astype(np.float32)

    outputs = []
    offset=input.shape[-1]-args.random_crop_size

    for i in range(input.shape[0]):
        output = rotate(padded_input.numpy()[i,...], angles[i], axes=(1,2), reshape=False)
        output=random_crop(output, args.random_crop_size,args.random_crop_size,offset)
        outputs.append(output)

    outputs=np.stack(outputs, 0)

    return torch.from_numpy(outputs), torch.from_numpy(angles)


def sample_data(args, data, rotations):
    #Returns a pairs of images withint the batch

    if args.sample_mini_batch:

        length=data.shape[0]

        idx_samples=np.array(random.sample(list(itertools.product(range(length),range(length))),200))

        rotation_difference=convert_to_convetion(rotations[idx_samples[:,1]]-rotations[idx_samples[:,0]])

        valid_idx_samples=idx_samples[(abs(rotation_difference.numpy())<=args.train_rotation_range).flatten()]


        valid_rotation_difference=rotation_difference[torch.ByteTensor(1*(abs(rotation_difference.numpy())\
            <=args.train_rotation_range))].reshape(-1,1)

        if valid_idx_samples.shape[0]>length:

            sample1=data[valid_idx_samples[:length,0]]

            sample2=data[valid_idx_samples[:length,1]]

            relative_rotations=valid_rotation_difference[:length]


        else:

            sample1=data[valid_idx_samples[:,0]]

            sample2=data[valid_idx_samples[:,1]]

            relative_rotations=valid_rotation_difference

    else: 
            sample1=roll(data, shift=1, axis=0)
            sample2=data

            relative_rotations=convert_to_convetion(rotations-roll(rotations,shift=1,axis=0))


    return sample1,sample2,relative_rotations


def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def main():

    # Training settings
    list_of_losses=['cosine_abs','cosine_mse','forbenius']
    list_of_choices=['Adam', 'SGD']
    list_of_resnet=['resnet18','resnet34','resnet50']
    parser = argparse.ArgumentParser(description='ResNet50 Regressor for Ants ')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--eval-batch-size', type=int, default=100, metavar='N',
                        help='eval batch size (default: 100)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
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
    parser.add_argument('--scheduler-loss', type=str, default='test',
                        help="choose which loss to apply the lr-scheduler on [test, train] (Default=test)")
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to wait until learning rate is reduced in plateua (default=5)')
    parser.add_argument('--print-progress', action='store_true', default=False,
                        help='con the progress on screen, Recommended for AWS')
    parser.add_argument('--resnet-type', type=str, default='resnet18', choices= list_of_resnet,
                        help='choose resnet type [resnet18,resnet34,resnet50] (default=resnet18)')
    parser.add_argument('--image-resize', type=int, default=120,
                        help='size for resizing input image (Default=120)')
    parser.add_argument('--random-crop-size', type=int, default=100,
                        help='random crop image size in pixel (Default=100)')
    parser.add_argument('--brightness', type=float, default=0.2,
                        help='brightness factor for ColorJitter augmentation')
    parser.add_argument('--contrast', type=float, default=0.2,
                        help='contrast factor for ColorJitter augmentation')
    parser.add_argument('--saturation', type=float, default=0.1,
                        help='saturation factor for ColorJitter augmentation')
    parser.add_argument('--hue', type=float, default=0.07,
                        help='hue factor for ColorJitter augmentation')
    parser.add_argument('--eval-rotation-range', type=float, default=90, metavar='theta',
                        help='evalutation rotation range in degrees for training,(Default=90), [-theta,+theta)')
    parser.add_argument('--train-rotation-range', type=float, default=180, metavar='theta',
                        help='training rotation range in degrees for training,(Defaul=180), [-theta,+theta)')
    parser.add_argument('--sample-mini-batch', action='store_true', default=False, 
                        help='Sample Mini-batch for pairs')
    parser.add_argument('--amsgrad', action='store_true', default=False, 
                        help='Turn on amsgrad in Adam optimiser')
    parser.add_argument('--save', type=int, default=10, metavar='N',
                        help='save model every this number of epochs (Default=5)')
    parser.add_argument('--loss', type=str, default='cosine_abs', choices= list_of_losses,
                        help="type of loss for atan2 penalty loss [abs,mse,forbenius] (Default= abs)")
    parser.add_argument('--samples', type=int, default=1000, metavar='N',
                        help='No of test samples (Default=1,000)')
    parser.add_argument('--threshold', type=float, default=0.1, metavar='l',
                        help='ReduceLROnPlateau signifance threshold (Default=0.1)')
    parser.add_argument('--no-pretrained', action='store_true', default=False,
                        help='start from non-pretrained reset (Default=False)')


    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    #Print arguments
    for arg in vars(args):
        sys.stdout.write('{} = {} \n'.format(arg,  getattr(args, arg)))
        sys.stdout.flush()
    sys.stdout.write('Random torch seed:{}\n'.format( torch.initial_seed()))
    sys.stdout.flush()


    torch.manual_seed(args.seed)

    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


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
        transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
        transforms.ToTensor(),
        normalise])

    eval_transformations=transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((args.image_resize,args.image_resize)),
            transforms.FiveCrop(args.random_crop_size),
            (lambda crops: torch.stack([transforms.Compose(
                [transforms.ToTensor(), normalise])(crop) for crop in crops]))])

    #Apply tranformtations

    train_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,train_rot_dir,transform=train_transformations),
        batch_size=args.batch_size, shuffle=True)

    train_loader_eval = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,train_rot_dir,transform=eval_transformations),
        batch_size=args.eval_batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,test_rot_dir,transform=eval_transformations),
        batch_size=args.eval_batch_size, shuffle=False)

    # Init model and optimizer

    model = Encoder(args.resnet_type,not args.no_pretrained)


    if args.optimizer=='Adam':
        optimizer=optim.Adam(model.parameters(), lr=args.lr,amsgrad=args.amsgrad)

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.lr_scheduler:
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            patience=args.patience,verbose=True,threshold=args.threshold)

    logging_dir='./logs_'+args.name

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    writer = SummaryWriter(logging_dir, comment='Encoder for ants')

    
    sys.stdout.write('Start training\n')
    sys.stdout.flush()


    n_iter=0
    for epoch in range(1, args.epochs + 1):
        sys.stdout.write('Starting epoch {}/{} \n '.format(epoch,args.epochs))
        sys.stdout.flush()

        for batch_idx, (data,rotations)  in enumerate(train_loader):
            model.train()

            #Rotate inputs

            targets,relative_rotations=rotate_tensor(args,data,args.train_rotation_range) 

            relative_rotations=relative_rotations.view(-1,1)

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

        #Evaluate real rotation loss

        test_real_mean, test_real_std=evaluate_rot_loss(args,model,test_loader)

        #Evaluate synthetic loss
        test_synth_mean, test_synth_std= eval_synthetic_rot_loss(args,model,test_loader)

        writer.add_scalars('Real Rotation Losses',
            {'Test Mean': test_real_mean, 
            'Test STD':test_real_std},epoch)
        writer.add_scalars('Synthetic Rotation Losses',
            {'Test Mean': test_synth_mean, 
            'Test STD':test_synth_std},epoch)


        if args.lr_scheduler:
            if args.scheduler_loss=='test':
                scheduler.step(test_synth_mean)
            elif args.scheduler_loss=='train':
                scheduler.step(train_mean)
            else:
                print('Wrong Loss Type')
                break

        if epoch%args.save==0:
            save_model(args,model,epoch)

    # plot_error(args,np.array(train_error_mean_log),np.array(train_error_std_log),
    #    np.array(test_error_mean_log),np.array(test_error_std_log) ,logging_dir)


def plot_error(args,train_mean,train_std,test_mean,test_std,path):
    """
    Plots error
    """

    with plt.style.context('ggplot'):

        fig, ax =plt.subplots()

        line1,=ax.plot(train_mean,label='Mean Train Error',linewidth=1.25,color='r')
        line2,=ax.plot(test_mean,label='Mean Test Error',linewidth=1.25,color='g')

        ax.fill_between(range(len(train_mean)),train_mean-train_std,train_mean+train_std,
            alpha=0.2,facecolor=line1.get_color(),edgecolor=line1.get_color())

        ax.fill_between(range(len(test_mean)),test_mean-test_std,test_mean+test_std,
            alpha=0.2,facecolor=line2.get_color(),edgecolor=line2.get_color())

        ax.set_ylabel('Degrees',fontsize=10)
        ax.set_xlabel('Epochs',fontsize=10)
        ax.set_xlim(0,None)
        #ax.set_ylim(-5,20)
 
        #Control colour of ticks
        ax.tick_params(colors='gray', direction='out')
        for tick in ax.get_xticklabels():
            tick.set_color('gray')
        for tick in ax.get_yticklabels():
            tick.set_color('gray')

        fig.tight_layout()
        fig.savefig(path+'/Learning curves')
        fig.clf()

if __name__ == '__main__':
    main()







