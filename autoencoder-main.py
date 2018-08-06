
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
from models import Autoencoder
from PIL import Image

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
         nn.init.xavier_normal_(m.weight)
    if classname.find('Linear') != -1:
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

def evaluate_reconstruction_loss(args,model, dataloader):
    """
    Calculate recostruction loss
    """
    total_loss=0.0
    model.eval()
        with torch.no_grad():
            for data,rotations in dataloader:
                targets,relative_rotations=rotate_tensor(args,data) 
                relative_rotations=relative_rotations.view(-1,1)

                # Forward pass
                output, _,_ = model(data, targets,relative_rotations*np.pi/180) 

                L1_loss = torch.nn.L1Loss(reduction='elementwise_mean')
                total_loss+= L1_loss(output,targets).item()* data.shape[0]

    return total_loss/len(dataloader.dataset)


def evaluate_rot_loss(args, model,dataloader,writer, epoch):

    model.eval()
    #Store errors
    
    #errors=np.zeros((len(dataloader.dataset)*5,1))
    abs_angles=np.zeros((len(dataloader.dataset),args.num_dims//2))
    counter=0

    with torch.no_grad():
        
        for batch_idx, (data,rotations) in enumerate(dataloader):
   
            #bs, ncrops, c, h, w = data.size()
            # rotations=(rotations.view(-1,1).repeat(1,5).view(-1,1)).float()
            #Get feature vector
            f_data=model.encoder(data)
            f_data=f_data.view(f_data.shape[0],-1)

            f_data_y= f_data[:,range(1,f_data.shape[1],2)] #Extract y coordinates
            f_data_x= f_data[:,range(0,f_data.shape[1],2)] #Extract x coordinate 

            theta_data=torch.atan2(f_data_y,f_data_x).numpy()*180/np.pi #Calculate absotulue angel of vector

            abs_angles[counter:counter+data.shape[0]]=theta_data[:,:args.num_dims//2]#Store values

            counter+=data.shape[0]

    #Get list of dataset absolute rotations from then dataset

    rotations=np.array(dataloader.dataset.rotations[1]).astype(float).reshape(-1,1)

    #Sample 1000 pairs of indices
    length=len(dataloader.dataset)

    idx_samples=np.array(random.sample(list(itertools.product(range(length),range(length))),args.samples))

    #Get the difference in the rotation adn covert to range [-180,180]

    rotation_difference=convert_to_convetion(rotations[idx_samples[:,1]]-rotations[idx_samples[:,0]])

    #Exclude differnce beyond  the specified limits

    valid_idx_samples=idx_samples[(abs(rotation_difference)<=args.random_rotation_range).flatten()]
    valid_rotation_difference=rotation_difference[abs(rotation_difference)<=args.random_rotation_range]

    estimated_rotation=abs_angles[valid_idx_samples[:,1]]-abs_angles[valid_idx_samples[:,0]]

    error=estimated_rotation-valid_rotation_difference

    if args.num_dims>2:
        std_rotation =  np.nanstd(estimated_rotation,axis=1,ddof=1)
        writer.add_histogram('Rotation STD hist', std_rotation, epoch)
    

    mean_error = abs(error).mean()
    error_std = error.std(ddof=1)
    
    return mean_error, error_std


class atan2_Loss(nn.Module):
    """
    Penalty loss on feature vector to ensure that in encodes rotation information
    """
    
    def __init__(self,num_dims,type, size_average=True):
        super(atan2_Loss,self).__init__()
        self.size_average=size_average #flag for mena loss
        self.num_dims=num_dims  # 2*int (numnber of dimensions to be penalised)
        self.type=type
        
    def forward(self,input,target):
        """
        Args:
            input: [batch,ndims,1,1]
            target: [batch,ndmins,1,1]
        """
        ndims=input.shape[1]
        input=input.view(input.shape[0],-1)
        target=target.view(target.shape[0],-1)

        #Batch size
        batch_size=input.shape[0]

        input_y=input[:,range(1,ndims,2)] #Extract y coordinates
        input_x=input[:,range(0,ndims,2)] #Extract x coordinate 

        target_x=target[:,range(0,ndims,2)] #Extract x coordinate 
        target_y=target[:,range(1,ndims,2)] #Extract u coordinate 

        #Calcuate agles using atan2(y,x)
        theta_input= torch.atan2(input_y,input_x)
        theta_target= torch.atan2(target_y,target_x)

        error=theta_target-theta_input

        if self.type=='mse':
            loss=(error[:,:self.num_dims//2]**2).mean()

        elif self.type=='abs':
            loss=abs(error[:,:self.num_dims//2]).mean()

        else:
            sys.stdout.write('wrong loss type\n')
            sys.stdout.flush()
            raise

        return loss


def double_loss(args,output,targets,f_data,f_targets):
    """
    Define double loss
    """

    #Recostriction L1 Loss
    L1_loss = torch.nn.L1Loss(reduction='elementwise_mean')
    atan2_loss =atan2_Loss(num_dims=args.num_dims, size_average=True,type=args.loss_type)
    
    #Combine
    reconstruction_loss=L1_loss(output,targets)
    rotation_loss=atan2_loss(f_data,f_targets)
    total_loss= (1-args.alpha)*reconstruction_loss+args.alpha*rotation_loss
    return total_loss,reconstruction_loss,rotation_loss


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


def reconstruction_test(args, model, test_loader, epoch,path):

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # Reshape data: apply multiple angles to the same minibatch, hence
            # repeat
            n,c,w,h=data.shape
            data = data.unsqueeze(1)
            data = data.repeat(1,1,n,1,1)
            data = data.view(n**2,c,w,h)
            target = torch.zeros_like(data)

            angles = torch.linspace(-args.random_rotation_range, args.random_rotation_range, steps=test_loader.batch_size)
            angles = angles.view(n, 1)
            angles = angles.repeat(1, n)
            angles = angles.view(n**2, 1)


            # Forward pass
            output,_,_ = model(data, target, angles)
            break
        save_images(args,output.cpu(), epoch,path)



def save_images(args,images, epoch, path, nrow=None):
    """Save the images in a grid format

    Args:
        images: array of shape [N,c,h,w],
    """
     if nrow == None:
         nrow = int(np.floor(np.sqrt(images.size(0)
             )))

    img = torchvision.utils.make_grid(images,nrow=nrow).numpy()
    img = np.transpose(img, (1,2,0))

    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.suptitle(r'Reconstruction, epoch={}, $\alpha$={}'.format(epoch,args.alpha))
    plt.savefig(path+"/Reconstruction_Epoch{:04d}".format(epoch))
    plt.close()



def main():

    # Training settings
    list_of_losses=['mse','abs']
    list_of_choices=['Adam', 'SGD']
    list_of_loss_to_monitor=['train', 'test']
    list_of_models=['resnet18,resnet34,resnet50']
    parser = argparse.ArgumentParser(description='ResNet50 Regressor for Ants ')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-test-batch-size', type=int, default=20, metavar='N',
                        help='test batch size (default: 20)')
    parser.add_argument('-recon-batch-size', type=int, default=5, metavar='N',
                        help='Number of Training images to be used for reconstruction test (Default=5)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
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
    parser.add_argument('--scheduler-loss', type=str, default='test', choices=list_of_loss_to_monitor,
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
    parser.add_argument('--random-rotation-range', type=float, default=90, metavar='theta',
                        help='random rotation range in degrees for training,(Default=90), [-theta,+theta)')
    parser.add_argument('--amsgrad', action='store_true', default=False, 
                        help='Turn on amsgrad in Adam optimiser')
    parser.add_argument('--save', type=int, default=5, metavar='N',
                        help='save model every this number of epochs (Default=5)')
    parser.add_argument('--recon-test-epochs', type=int, default=5, metavar='N',
                        help='Epochs intervals fro reconstruction test (Default=5)')
    parser.add_argument('--loss-type', type=str, default='abs', choices= list_of_choices,
                        help="type of loss for atan2 penalty loss [abs,mse] (Default= abs)")
    parser.add_argument('--alpha', type=float, default=0.5, metavar='a',
                        help='propotion of atan2 penalty loss (Default=0.5)')
    parser.add_argument('--samples', type=int, default=1000, metavar='N',
                        help='No of test samples (Default=1,000)')
    parser.add_argument('--num-dims', type=int, default=2, metavar='D',
                        help='Number of dimensions to penalise (Default=2)')
    parser.add_argument('--threshold', type=float, default=0.1, metavar='l',
                        help='ReduceLROnPlateau signifance threshold (Default=0.1)')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    #Print arguments
    for arg in vars(args):
        sys.stdout.write('{} = {} \n'.format(arg,  getattr(args, arg)))
        sys.stdout.flush()


    torch.manual_seed(args.seed)


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
        batch_size=100, shuffle=False)

    train_reconstrunction_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,train_rot_dir,transform=train_transformations),
        batch_size=args.recon_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,test_rot_dir,transform=eval_transformations),
        batch_size=args.test_batch_size, shuffle=True)

    # Init model and optimizer

    model = Autoencoder(args.resnet_type)

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

    writer = SummaryWriter(logging_dir, comment='Autoencoder for ants')

    
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
            output, f_data, f_targets = model(data, targets,relative_rotations*np.pi/180) 
            optimizer.zero_grad()


            #Loss
            L1_loss = torch.nn.L1Loss(reduction='elementwise_mean')
            loss=L1_loss(output,targets)
            #loss,reconstruction_loss,atan2_loss=double_loss(args,output,targets,f_data,f_targets)
            # Backprop
            loss.backward()
            optimizer.step()

            # writer.add_scalars('Mini-batch loss',{'Total Loss':  loss.item(),
            #                          'Reconstruction Loss':reconstruction_loss.item() ,
            #                          'ata2 Loss': atan2_loss }, n_iter)

            if args.print_progress:

                sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                sys.stdout.flush()

            
            n_iter+=1

        ## test_mean, test_std= evaluate_rot_loss(args, model,test_loader,writer, epoch)
        ## writer.add_scalar('Test error',test_mean,epoch)

        # test_error_mean_log.append(test_mean)
        # test_error_std_log.append(test_std)


          
        sys.stdout.write('Ended epoch {}/{} \n '.format(epoch,args.epochs))
        sys.stdout.flush()


        if args.lr_scheduler:
            if args.scheduler_loss=='test':
                scheduler.step(test_mean)
            elif args.scheduler_loss=='train':
                scheduler.step(evaluate_reconstruction_loss(args,model,train_loader_eval ))
            else:
                print('Wrong Loss Type')
                break

        if epoch%args.save==0:
            save_model(args,model,epoch)

        if epoch%args.recon_test_epochs==0:
            reconstruction_test(args, model, train_reconstrunction_loader, epoch,logging_dir)


    # plot_error(args,np.array(test_error_mean_log),np.array(test_std),logging_dir)


def plot_error(args,average_error,error_std,path):
    """
    Plots error
    """

    with plt.style.context('ggplot'):

        fig, ax =plt.subplots()

        line,=ax.plot(average_error,label='Means absolute test error',linewidth=1.25,color='g')
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

        fig.suptitle(r'Test error for $\alpha$={},$D$={}'.format(args.alpha,args.num_dims))
        fig.tight_layout()
        fig.savefig(path+'/Test_error')
        fig.clf()



                

if __name__ == '__main__':
    main()







