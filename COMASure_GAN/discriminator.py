import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from pytorch_msssim import ssim as ssim_sys, ms_ssim as ms_ssim_sys, SSIM, MS_SSIM
import pandas as pd
import random 
import copy

class Discriminator(nn.Module):
    def __init__(self, C, H, W):
        super(Discriminator, self).__init__()

        self.C, self.H, self.W = C, H, W
        self.output_shape = (1, H, W)
        self.in_channels = C

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        layers = []
        in_filters = self.in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Flatten())

        layers_support = []
        layers_support.append(nn.Linear(4, 1024)) # The input size to this layer depends on the actual image patch size
        layers_support.append(nn.LeakyReLU(0.2))
        layers_support.append(nn.Linear(1024, 1))
        layers_support.append(nn.Sigmoid())

        layers_query = []
        layers_query.append(nn.Linear(64, 1024)) # The input size to this layer depends on the actual image patch size
        layers_query.append(nn.LeakyReLU(0.2))
        layers_query.append(nn.Linear(1024, 1))
        layers_query.append(nn.Sigmoid())

        self.model_support = nn.Sequential(*(layers+layers_support))
        self.model_query = nn.Sequential(*(layers+layers_query))

    def forward(self, img, mode):
      if mode == "support":
        return self.model_support(img).squeeze(1) # Img in (C, H, W)
      else:
        return self.model_query(img).squeeze(1) # Img in (C, H, W)