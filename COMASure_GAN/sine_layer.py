import torch
import torch.nn as nn
import torchvision
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from pytorch_msssim import ssim as ssim_sys, ms_ssim as ms_ssim_sys, SSIM, MS_SSIM
import pandas as pd
import random 
import copy

from PIL import Image
import matplotlib.pyplot as plt

class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, is_last=False, omega_0=200.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last        
        self.in_features = in_features                
        self.linear = nn.Linear(in_features, out_features, bias=bias)        
        self.init_weights()
    
    def init_weights(self):
        if self.is_first:
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
        else:
            self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def init_weights_meta(self, para_w, para_b):
        # self.linear.weight = torch.nn.Parameter(para_w)
        # self.linear.bias = torch.nn.Parameter(para_b)
        self.linear.weight = para_w
        self.linear.bias = para_b
                
    def forward(self, input):
        if self.is_last == False:
          return torch.sin(self.omega_0 * self.linear(input))
        else:
          return self.linear(input) + 0.5