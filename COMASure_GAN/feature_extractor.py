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

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
        for _, parameters in self.feature_extractor.named_parameters():
            parameters.requires_grad = False

    def forward(self, img):
        return self.feature_extractor(img)