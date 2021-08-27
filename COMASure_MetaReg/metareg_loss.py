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

class MetaReg(nn.Module):
    def __init__(self, meta_reg_lr, last_hidden_layer_size, output_layer_size):
        super(MetaReg, self).__init__()
        self.meta_reg_lr = meta_reg_lr
        self.last_hidden_layer_size = last_hidden_layer_size
        self.output_layer_size = output_layer_size
        self.linear_layer = nn.Sequential(nn.Linear((self.last_hidden_layer_size+1)*self.output_layer_size, 1), nn.Sigmoid())
        # self.linear_layer = nn.Sequential(nn.Linear((self.last_hidden_layer_size+1)*self.output_layer_size, 1))

    def forward(self, model_paras):
        loss = self.linear_layer(torch.abs(model_paras))
        return loss