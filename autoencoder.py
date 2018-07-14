
# coding: utf-8

# In[1]:


########
#IMPORTS
########

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import math
import numpy as np

import imageio
import matplotlib.pyplot as plt

import Disk


# In[2]:


###########
#PARAMETERS
###########

#Disk parameters
scale = 2
center_x = 32*scale
center_y = 32*scale
width = 64*scale
height = 64*scale
max_radius = int(math.floor(min(width, height)/2))
radius_list = np.arange(1, max_radius)
sample_number = int(len(radius_list)/2)

#Neural network parameters
epoch = 50
layer_1 = width*height


# In[3]:


########
#SAMPLES
########

#Randomly generate a list of disks with the same center
diskList = []
for r in np.random.choice(radius_list, sample_number, replace=False):
    d = Disk.DiskImage(r, center_x, center_y, width, height)
    diskList.append(d)
    #d.writeImage()
    
#Convert a list of disks to a torch tensor
def makeTensor(diskList):
    height = diskList[0].height
    width = diskList[0].width
    
    W = torch.Tensor(len(diskList), 1, height, width)
    A = list(map(lambda d:d.TorchImage, diskList))
    
    W = torch.cat(A, out=W)
    return W

#Samples comprise a torch tensor derived from a list of disks.
samples = makeTensor(diskList)


# In[5]:


############
#AUTOENCODER
############

class AutoEncoder(nn.Module):
    
    def Encode(i,j):
        maps = nn.Sequential(
                nn.Conv2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
                nn.MaxPool2d(2, padding=0),
                nn.LeakyReLU(0.2)
                )
        return maps

    def Decode(i,j):
        maps = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
                nn.LeakyReLU(0.2)
            )
        return maps
    
    def __init__(self):
        
        super().__init__()
        
        self.encode = nn.Sequential(
            AutoEncoder.Encode(1,8),
            AutoEncoder.Encode(8,4),
            AutoEncoder.Encode(4,4),
            AutoEncoder.Encode(4,3),
            AutoEncoder.Encode(3,2),
            AutoEncoder.Encode(2,1)
        )
        
        self.decode = nn.Sequential(
            AutoEncoder.Decode(1,2),
            AutoEncoder.Decode(2,3),
            AutoEncoder.Decode(3,4),
            AutoEncoder.Decode(4,4),
            AutoEncoder.Decode(4,8),
            AutoEncoder.Decode(8,1)
        )
        
    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x

#AutoEncoder instance and optimizer.
model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters()) #Adam's default learning rate is 0.001

c = 1
for j in range (0, epoch):
    output = model(samples)
    loss = criterion(output, samples)
    loss.backward()
    optimizer.step()

    if c%25==0:
        print("ITERATION:", c)
        print("LOSS:", loss)
        for param in model.parameters():
            print(" PARAMETER:", param.size(), "\n", "PARAMETER NORM:", torch.norm(param))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    c = c + 1
    
for param in model.parameters():
    print(param)
    
#TO DO: interpret the action of the autoencoder on your dataset.


# In[6]:


##############
#VISUALIZATION
##############

def getDiskGrid(tensor):
    if tensor.requires_grad == True:
        tensor = tensor.detach()
    #Make a visualizable grid of type torch.Tensor, convert to a numpy array, 
    #convert the dtype of the numpy array to 'uint8'.
    grid = torchvision.utils.make_grid(tensor, nrow=5, padding = 100)
    grid = grid.permute(1,2,0)
    grid = grid.numpy()
    grid = grid.astype("uint8")
    return grid

# Plot the image here using Matplotlib.
def plotSingleDisk(tensor):
    if tensor.requires_grad == True:
        tensor = tensor.detach()
    # imshow needs a numpy array with the channel dimension
    # as the the last dimension so we have to transpose things.
    z = tensor.reshape(tensor.size()[1],tensor.size()[2])
    plt.imshow(z, cmap = "gray")
    plt.show()
    
def writeDiskGrid(tensor, filename):
    grid = getDiskGrid(tensor)
    plt.imshow(grid, cmap = "gray")
    imageio.imwrite(filename, grid)

#plotSingleDisk(samples[0])
#plotSingleDisk(model(samples)[0])
    
writeDiskGrid(samples, "circles.jpeg")
writeDiskGrid(model(samples), "encoded_circles.jpeg")

