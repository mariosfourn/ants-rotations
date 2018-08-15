
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
from models import Autoencoder
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


def convert_to_convetion(input):
    """
    Coverts all anlges to convecntion used by atan2
    """

    input[input<180]=input[input<180]+360
    input[input>180]=input[input>180]-360
    
    return input


def round_even(x):
    return int(round(x/2.)*2)

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

            total_loss+= reconstruction_loss(args,output,targets).item()* data.shape[0]

    return total_loss/len(dataloader.dataset)


def reconstruction_loss(args,x,y):
    L1_loss = torch.nn.L1Loss(reduction='elementwise_mean') 
    
    ssim_loss =pytorch_ssim.SSIM(window_size= args.window_size) #Average
    loss=(1-args.beta) * L1_loss(x,y) + args.beta * torch.clamp( (1-ssim_loss(x,y))/2,0,1)

    return loss


def eval_synthetic_rot_loss(args,model,data_loader):

    model.eval()
    #Number of features penalised
    ndims=args.num_dims
    error=np.zeros(len(data_loader.dataset))
    counter=0
    with torch.no_grad():
        for data,_ in data_loader:
            ## Reshape data
            targets,rotations=rotate_tensor(args,data)

            # Forward passes
            f_data=model.encoder(data)
            f_data=f_data.view(f_data.shape[0],-1) #convert 3D vector to 2D

            f_data_y= f_data[:,range(1,f_data.shape[1],2)] #Extract y coordinates
            f_data_x= f_data[:,range(0,f_data.shape[1],2)] #Extract x coordinate 

            f_targets=model.encoder(targets)
            f_targets=f_targets.view(f_targets.shape[0],-1) #convert 3D vector to 2D

            f_targets_y= f_targets[:,range(1,f_targets.shape[1],2)] #Extract y coordinates
            f_targets_x= f_targets[:,range(0,f_targets.shape[1],2)] #Extract x coordinate 

            theta_data=torch.atan2(f_data_y,f_data_x).cpu().numpy()*180/np.pi #Calculate absotulue angel of vector
            theta_targets=torch.atan2(f_targets_y,f_targets_x).cpu().numpy()*180/np.pi #Calculate absotulue angel of vector

            estimated_angle=np.nanmean(theta_targets[:,:ndims//2]-theta_data[:,:ndims//2],axis=1)
            
            estimated_angle=convert_to_convetion(estimated_angle)

            error[counter:counter+data.shape[0]]=estimated_angle-rotations.numpy()
            
            counter+=data.shape[0]

           
    return  abs(error).mean(), error.std()


def evaluate_real_rot_loss(args, model,dataloader,writer, epoch):

    model.eval()

    #Number of features penalised
    ndims=args.num_dims

    abs_angles=np.zeros((len(dataloader.dataset),ndims//2))
    rotations=np.zeros((len(dataloader.dataset),1))

    counter=0

    with torch.no_grad():
        
        for batch_idx, (data,batch_rotations) in enumerate(dataloader):
   
            #Get feature vector

            f_data=model.encoder(data)
            f_data=f_data.view(f_data.shape[0],-1) #convert 3D vector to 2D

            f_data_y= f_data[:,range(1,f_data.shape[1],2)] #Extract y coordinates
            f_data_x= f_data[:,range(0,f_data.shape[1],2)] #Extract x coordinate 

            theta_data=torch.atan2(f_data_y,f_data_x).numpy()*180/np.pi #Calculate absotulue angel of vector

            abs_angles[counter:counter+data.shape[0]]=theta_data[:,:ndims//2]#Store values
            rotations[counter:counter+data.shape[0]]=batch_rotations.reshape(-1,1)

            counter+=data.shape[0]


    length=rotations.shape[0]

    idx_samples=np.array(random.sample(list(itertools.product(range(length),range(length))),args.samples))

    #Get the difference in the rotation adn covert to range [-180,180]

    rotation_difference=convert_to_convetion(rotations[idx_samples[:,1]]-rotations[idx_samples[:,0]])

    #Exclude differnce beyond  the specified limits

    valid_idx_samples=idx_samples[(abs(rotation_difference)<=args.eval_rotation_range).flatten()]

    valid_rotation_difference=rotation_difference[(abs(rotation_difference)<=args.eval_rotation_range).flatten()].reshape(-1,1)

    estimated_rotation=convert_to_convetion(abs_angles[valid_idx_samples[:,1]]-abs_angles[valid_idx_samples[:,0]])

    error=estimated_rotation-valid_rotation_difference

    if valid_rotation_difference.shape[1]>2:
        std_rotation =  np.nanstd(estimated_rotation,axis=1,ddof=1)
        writer.add_histogram('Rotation STD hist', std_rotation, epoch)
    
    error=error.mean(axis=1)


    mean_error = abs(error).mean()
    error_std = error.std(ddof=1)
    
    return mean_error, error_std


class FeatureVectorLoss(nn.Module):
    """
    Penalty loss on feature vector to ensure that in encodes rotation information
    """
    
    def __init__(self, type , size_average=True):
        super(FeatureVectorLoss,self).__init__()
        self.size_average=size_average #flag for mena loss
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
        ndims=x.shape[1]
        #Batch size
        batch_size=x.shape[0]

        reg_loss=0.0

        cosine_similarity=nn.CosineSimilarity(dim=2)

        for i in range(0,ndims-1,2):
            x_i=x[:,i:i+2]
            y_i=y[:,i:i+2]
            # dot_prod=torch.bmm(x_i.view(batch_size,1,2),y_i.view(batch_size,2,1)).view(batch_size,1)
            # x_norm=torch.norm(x_i, p=2, dim=1, keepdim=True)
            # y_norm=torch.norm(y_i, p=2, dim=1, keepdim=True)

            if self.type=='mse':
                reg_loss+= torch.abs(cosine_similarity(x_i.view(x_i.size(0),1,2),y_i.view(y_i.size(0),1,2))-1.0).sum()
                #reg_loss+=((dot_prod/(x_norm*y_norm)-1)**2).sum()
            elif self.type=='abs':
               
                reg_loss+= torch.abs(cosine_similarity(x_i.view(x_i.size(0),1,2),y_i.view(y_i.size(0),1,2))-1.0).sum()
                #eg_loss+=(abs(dot_prod/(x_norm*y_norm)-1)).sum()
              
            elif self.type=='L2_norm':
                forb_distance=torch.nn.PairwiseDistance()
                x_polar=x_i/torch.maximum(x_norm, 1e-08)
                y_polar=y_i/torch.maximu(y_norm,1e-08)
                reg_loss+=(forb_distance(x_polar,y_polar)**2).sum()
           
        if self.size_average:
            reg_loss=reg_loss/x.shape[0]/(ndims//2)



        return reg_loss


def double_loss(args,output,targets,f_data,f_targets):
    """
    Define double loss
    """

    #Recostriction L1 Loss
    L1_loss = torch.nn.L1Loss(reduction='elementwise_mean')
    feature_vector_loss =FeatureVectorLoss(type=args.loss_type)
    
    #Combine
    reconstruction_loss=L1_loss(output,targets)

    rotation_loss=feature_vector_loss(f_data[:,:args.num_dims],f_targets[:,:args.num_dims])

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

    angles = args.train_rotation_range*np.random.uniform(-1,1,input.shape[0])

    angles = angles.astype(np.float32)

    outputs = []

    for i in range(input.shape[0]):
        output = rotate(padded_input.numpy()[i,...], angles[i], axes=(1,2), reshape=False)
        output=centre_crop(output, args.random_crop_size,args.random_crop_size)
        outputs.append(output)

    outputs=np.stack(outputs, 0)

    return torch.from_numpy(outputs), torch.from_numpy(angles)


def reconstruction_test(args, model, test_loader, epoch,path,steps=8):

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # Reshape data: apply multiple angles to the same minibatch, hence
            # repeat
            n,c,w,h=data.shape
            data2 = data.unsqueeze(1)
            data2 = data2.repeat(1,1,steps,1,1)
            data2 = data2.view(n*steps,c,w,h)
            target = torch.zeros_like(data2)

            angles = torch.linspace(-args.train_rotation_range-10, args.train_rotation_range+10, steps=steps)
            angles = angles.view(1,steps)
            angles=angles.repeat(1,n).view(-1,1)

            # Forward pass
            output,_,_ = model(data2, target, angles*np.pi/180)

            output=output.view(n,steps,3,w,h)

            output=torch.cat((output,data.view(n,1,c,w,h)),dim=1)

            output=output.view(-1,c,w,h)

            break
        save_images(args,output.cpu(), epoch,path,nrow=steps+1)


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
    list_of_losses=['mse','abs','L2_norm']
    list_of_choices=['Adam', 'SGD']
    list_of_loss_to_monitor=['train', 'test']
    list_of_models=['resnet18','resnet34','resnet50']
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
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='',
                        help='name of the run that is added to the output directory')
    parser.add_argument('--optimizer', type=str, default='Adam', choices= list_of_choices,
                        help="Choose optimiser between 'Adam' (default) and 'SGD' with momentum")
    parser.add_argument('--lr-scheduler', action='store_true', default=False, 
                        help='set up lernaring rate scheduler (Default off)')
    parser.add_argument('--scheduler-loss', type=str, default='test', choices=list_of_loss_to_monitor,
                        help='which loss to track for lr-scheduler [train, test] (Default=test)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to wait until learning rate is reduced in plateua (default=5)')
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
    parser.add_argument('--train-rotation-range', type=float, default=180, metavar='theta',
                        help='rotation range in degrees for training,(Default=180), [-theta,+theta)')
    parser.add_argument('--eval-rotation-range', type=float, default=90, metavar='theta',
                        help='rotation range in degrees for evaluation,(Default=90), [-theta,+theta)')
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
                        help='Number of dimensiosn to be penalised (Default=2.0)')
    parser.add_argument('--threshold', type=float, default=1e-4, metavar='l',
                        help='ReduceLROnPlateau signifance threshold (Default=1e-4)')
    parser.add_argument('--beta', type=float , default=0.85, 
                        help='Blending coeffecient for SSIM loss and L1 loss (Default=0.85)')
    parser.add_argument('--window-size', type=int , default=11, 
                        help='Window size for SSIM loss (Default=11 pixels)')
    parser.add_argument('--dropout-rate', type=float, default=0.3, metavar='P',
                        help='dropout applied to feature vector during (Default=0.3)')

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
        batch_size=100, shuffle=False)

    train_reconstrunction_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,train_rot_dir,transform=train_transformations),
        batch_size=args.recon_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        AntsDataset(data_root_dir,test_rot_dir,transform=eval_transformations),
        batch_size=args.test_batch_size, shuffle=True)

    # Init model and optimizer

    model = Autoencoder(args.resnet_type,dropout_rate=args.dropout_rate,num_dims=args.num_dims)
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

            loss,reconstruction_loss,rotation_loss=double_loss(args,output,targets,f_data,f_targets)
            # Backprop
            loss.backward()
            optimizer.step()

            writer.add_scalars('Mini-batch loss',{'Total Loss':  loss.item(),
                                      'Reconstruction Loss':reconstruction_loss.item() ,
                                      'Rotation Loss ': rotation_loss }, n_iter)
            if args.print_progress:

                sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                sys.stdout.flush()

            
            n_iter+=1


        #Calculae synthetic rotation loss
        test_synthetic_mean, test_synthetic_std = eval_synthetic_rot_loss(args, model,test_loader)

        #Calculate real rotation loss
        test_real_mean, test_real_std = evaluate_real_rot_loss(args, model,test_loader,writer, epoch)

        
        writer.add_scalars('Test error', {'synthetic': test_synthetic_mean,'real': test_real_mean},epoch)

        # test_synthetic_error_mean_log.append(test_mean)
        # test_error_std_log.append(test_std)

        #train_loss=evaluate_reconstruction_loss(args,model,train_loader_eval)
        #sys.stdout.write('Ended epoch {}/{}, Reconstruction loss on train set ={:4f}\n '.format(epoch,args.epochs,train_loss))
        #sys.stdout.flush()


        if args.lr_scheduler:
            if args.scheduler_loss=='test':
                scheduler.step(test_synthetic_mean)
            elif args.scheduler_loss=='train':
                scheduler.step(train_loss)
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







