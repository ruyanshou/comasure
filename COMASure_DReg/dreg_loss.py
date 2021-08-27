import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg19
from torch.autograd import Variable
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from pytorch_msssim import ssim as ssim_sys, ms_ssim as ms_ssim_sys, SSIM, MS_SSIM
import pandas as pd
import random 
import copy

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg19_model = vgg19(pretrained=True).cuda()
        self.linear_layer = nn.Sequential(nn.Linear(1000, 1), nn.Sigmoid()) # Input size depends on patch size
        for _, parameters in self.vgg19_model.named_parameters():
            parameters.requires_grad = False

    def forward_img(self, img):
        x = self.vgg19_model(img)
        x = x.view(x.size()[0], -1)
        x = self.linear_layer(x)
        return x

    def forward(self, img, mode, patch_size_lr, downscaling_factor):
        if mode == "support":
          output = self.forward_img(img)
        else:
          patch_list = []
          for i in range(downscaling_factor):
            for j in range(downscaling_factor):
              patch_list.append(img[:,:,i*patch_size_lr:(i+1)*patch_size_lr,j*patch_size_lr:(j+1)*patch_size_lr])
          input = torch.cat(patch_list, dim=0)
          output = self.forward_img(input)
        loss = torch.mean(output)
        # if mode == "support":
        #   print(mode, output.shape, loss.shape)
        # else:
        #   print(mode, output.shape, loss.shape)
        return loss