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
        vgg19_model = vgg19(pretrained=True).cuda()
        self.activation_layers = [4,18,26] # Activation layers of vgg19
        self.feature_extractors = []
        self.linear_layer = nn.Sequential(nn.Linear(90112, 1024), nn.Sigmoid()) # Input size depends on patch size
        for i in self.activation_layers:
          feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:i])
          for _, parameters in feature_extractor.named_parameters():
              parameters.requires_grad = False
          self.feature_extractors.append(feature_extractor)

    def forward_img(self, img):
        results = []
        for i in range(len(self.feature_extractors)):
          x = self.feature_extractors[i](img)
          x = x.view(x.size()[0], -1)
          results.append(x)
        result = torch.cat(results, dim=1)
        result = self.linear_layer(result)
        return result

    def forward(self, img1, img2, mode, patch_size_lr, downscaling_factor):
        if mode == "support":
          output1 = self.forward_img(img1)
          output2 = self.forward_img(img2)
        else:
          patch1_list = []
          patch2_list = []
          for i in range(downscaling_factor):
            for j in range(downscaling_factor):
              patch1_list.append(img1[:,:,i*patch_size_lr:(i+1)*patch_size_lr,j*patch_size_lr:(j+1)*patch_size_lr])
              patch2_list.append(img2[:,:,i*patch_size_lr:(i+1)*patch_size_lr,j*patch_size_lr:(j+1)*patch_size_lr])
          input1 = torch.cat(patch1_list, dim=0)
          input2 = torch.cat(patch2_list, dim=0)
          output1 = self.forward_img(input1)
          output2 = self.forward_img(input2)
        cos_sim_loss = nn.CosineEmbeddingLoss()(output1, output2, torch.tensor([1]).cuda())
        # if mode == "support":
        #   print(mode, cos_sim_loss, img1.shape, output1.shape)
        # else:
        #   print(mode, cos_sim_loss, input1.shape, output1.shape)
        return cos_sim_loss