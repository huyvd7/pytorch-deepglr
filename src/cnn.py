import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import os
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd         # computation graph
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import torchvision
import torchvision.transforms as transforms
from graph import *

class cnnf(nn.Module):
    def __init__(self):
        super(cnnf, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer2a = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.layer3a = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # self.maxpool
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # DECONVO
        self.upsample = nn.Upsample(scale_factor=2)
        self.deconvo1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=1, padding=1)  # output size should be 90x90
        # CONCAT with output of layer2
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # DECONVO
        # self.upsample
        self.deconvo2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=1, padding=1)  # output size should be 180x180

        # CONCAT with output of layer1
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
            # nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        outl1 = self.layer1(x)
        outl2 = self.layer2a(outl1)
        size2 = outl2.size()
        outl2 = self.maxpool(outl2)
        outl2 = self.layer2(outl2)
        outl3 = self.layer3a(outl2)
        size3 = outl3.size()
        outl3 = self.maxpool(outl3)
        outl3 = self.layer3(outl3)
        outl3 = self.upsample(outl3)
        outl3 = self.deconvo1(outl3, output_size=size3)
        outl3 = torch.cat((outl3, outl2), dim=1)
        outl4 = self.layer4(outl3)
        outl4 = self.upsample(outl4)
        outl4 = self.deconvo2(outl4, output_size=size2)
        outl4 = torch.cat((outl4, outl1), dim=1)
        del outl1, outl2, outl3, size2, size3
        out = self.layer5(outl4)
        return out


class cnny(nn.Module):
    def __init__(self):
        super(cnny, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out = identity + out
        del identity
        return out


class cnnu(nn.Module):
    def __init__(self):
        super(cnnu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(2*2*32, 1*1*32),
        #     nn.Linear(1*1*32, 1)

        # )
        
        self.fc = nn.Sequential(
            nn.Linear(3*3*32, 1*1*32),
            nn.Linear(1*1*32, 1)
        )
    
    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
        

class RENOIR_Dataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.npath = os.path.join(img_dir, 'noisy')
        self.rpath =  os.path.join(img_dir, 'ref')
        self.nimg_name = os.listdir(self.npath)
        self.rimg_name = os.listdir(self.rpath)
        self.transform = transform

    def __len__(self):
        return len(self.nimg_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        nimg_name = os.path.join(self.npath, self.nimg_name[idx])
        nimg = cv2.imread(nimg_name)
        rimg_name = os.path.join(self.rpath, self.rimg_name[idx])
        rimg = cv2.imread(rimg_name)
        
        sample = {'nimg': nimg, 'rimg': rimg}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class bgr2rgb(object):
    """Convert opencv BGR to RGB order."""
    def __init__(self, scale=None):
        self.scale=scale

    def __call__(self, sample, scale=None):
        nimg, rimg = sample['nimg'], sample['rimg']
        if self.scale:
            nimg = cv2.resize(nimg, (0,0), fx=self.scale, fy=self.scale) 
            rimg = cv2.resize(rimg, (0,0), fx=self.scale, fy=self.scale) 
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        nimg = nimg/255
        rimg = rimg/255
        return {'nimg': nimg,
                'rimg': rimg}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        nimg, rimg = sample['nimg'], sample['rimg']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        nimg = nimg.transpose((2, 0, 1))
        rimg = rimg.transpose((2, 0, 1))
        return {'nimg': torch.from_numpy(nimg),
                'rimg': torch.from_numpy(rimg)}


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        
class GLR(nn.Module):
    def __init__(self, width=28, cuda=False):
        super(GLR, self).__init__()
        self.cnnf = cnnf()
        self.cnny = cnny()
        self.cnnu = cnnu()
        self.cnnf.apply(weights_init_normal)
        self.cnny.apply(weights_init_normal)
        self.cnnu.apply(weights_init_normal)
        self.wt = width
        self.I = torch.eye(self.wt**2, self.wt**2)
            
    def forward(self, x):
        xf = x.unsqueeze(1)
        E = self.cnnf.forward(xf).squeeze(0)
        Y = self.cnny.forward(xf).squeeze(0)
        u = self.cnnu.forward(xf)
        u[u>((250-1)/(2*8))]= (250-1)/(2*8)
        img_dim = self.wt
        if dataloader.batch_size==1:
            E = E.unsqueeze(0)
            Y = Y.unsqueeze(0)
        E = E.view(E.shape[0], E.shape[1], img_dim**2)
        Y = Y.view(Y.shape[0], img_dim**2, 1)

        L = laplacian_construction(width=img_dim, F=E)
        
        out = qpsolve(L=L, u=u, y=Y, I=self.I, wt=img_dim)
        return out.view(-1, img_dim, img_dim)

    def predict(self, x):
        xf = x.unsqueeze(1)
        E = self.cnnf.forward(xf).squeeze(0)
        Y = self.cnny.forward(xf).squeeze(0)
        u = self.cnnu.forward(xf)
        u[u>((250-1)/(2*8))]= (250-1)/(2*8)
        img_dim = self.wt
        I = torch.eye(img_dim**2, img_dim**2)
        if x.shape[0]==1:
            E = E.unsqueeze(0)
            Y = Y.unsqueeze(0)
        E = E.view(E.shape[0], E.shape[1], img_dim**2)
        Y = Y.view(Y.shape[0], img_dim**2, 1)

        L = laplacian_construction(width=img_dim, F=E)
        
        out = qpsolve(L=L, u=u, y=Y, I=I, wt=img_dim)
#         out = out.view(-1, img_dim, img_dim)
        return out.view(-1, img_dim, img_dim)