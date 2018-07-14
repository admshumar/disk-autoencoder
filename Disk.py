
# coding: utf-8

# In[81]:


#Imports
import os
import math
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

class Disk:
    #Parametrization of the boundary of a disk.
    def Boundary(x, radius, center_x, center_y, sign):
        w = (radius)**2 - (x-center_x)**2
        if w >= 0:
            y = center_y + (sign) * math.floor(math.sqrt(w))
        else:
            #If x gives an undefined output, we set y = -1, which is an
            #undefined pixel index.
            y = -1
        return y

    def __init__(self, radius, center_x, center_y):
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y
        self.center = [center_x, center_y]

class DiskImage(Disk):

    def makeImage(radius, center_x, center_y, width, height):
        X = np.zeros((width, height), dtype=float)
        for x in range(0, height):
            lower_bound = Disk.Boundary(x, radius, center_x, center_y, -1)
            upper_bound = Disk.Boundary(x, radius, center_x, center_y, 1)
            for y in range(lower_bound, upper_bound+1):
                if y != -1:
                    X[x,y] = 255.0
        return X

    def makeTorchRow(X):
        m = X.size
        d = torch.from_numpy(X)
        d = torch.reshape(d,(1, m))
        d = d.float()
        return d

    def makeTorchColumn(X):
        return torch.t(makeTorchRow(X))

    def __init__(self, radius, center_x, center_y, width, height):
        Disk.__init__(self, radius, center_x, center_y)
        self.width = width
        self.height = height
        self.size = width * height
        self.image = DiskImage.makeImage(self.radius, self.center_x, self.center_y, self.width, self.height)
        self.TorchImage = torch.reshape((torch.from_numpy(self.image)).float(), (1, 1, self.width, self.height))
        self.sample = DiskImage.makeTorchRow(self.image)

    def showImage(self):
        return plt.show(plt.imshow(self.image, cmap='gray'))

    def writeImage(self):
        directory = "images/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + str(self.width) + "_" + str(self.height) + "_" + str(self.radius) + "_" + str(self.center_x) + "_" + str(self.center_y) +".jpeg"
        new_p = Image.fromarray(self.image*255) #JPEG takes values between 0 and 255, NOT 0 and 1
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(filename, "JPEG")
