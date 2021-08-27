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

class FourierFeature(nn.Module):

    def __init__(self, N_features, sigma, channels):
        super().__init__()
        self.register_buffer("a", torch.ones(N_features)) 
        self.register_buffer("B", torch.randn(N_features, channels) * sigma)

    def forward(self, X):

        # Only apply fourier feature to Cartesian coordinates
        h = X @ self.B.T
        h =  2 * math.pi * h

        return torch.cat((self.a*h.cos(), self.a*h.sin()), -1)