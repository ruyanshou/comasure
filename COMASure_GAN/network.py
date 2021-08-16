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

from sine_layer import SineLayer

class MLP(nn.Module):
    def __init__(self, layer_sizes, use_sine_layer):
        
        super(MLP, self).__init__()

        self.layer_sizes = layer_sizes
        self.first_omega_0 = 200.0
        self.hidden_omega_0 = 20.0
        self.use_sine_layer = use_sine_layer
        self.vars = nn.ParameterList()

        if self.use_sine_layer:
          # Layers
          for i in range(len(self.layer_sizes)-2):
            if i == 0:
              w = torch.empty(self.layer_sizes[i], self.layer_sizes[i+1]).T # output, input
              nn.init.uniform_(w, -1/self.layer_sizes[i], 1/self.layer_sizes[i])       
              self.vars.append(nn.Parameter(w))
              self.vars.append(nn.Parameter(torch.zeros(self.layer_sizes[i+1])))
            else:
              w = torch.empty(self.layer_sizes[i], self.layer_sizes[i+1]).T
              nn.init.uniform_(w, -np.sqrt(6 / self.layer_sizes[i]) / self.hidden_omega_0, np.sqrt(6 / self.layer_sizes[i]) / self.hidden_omega_0)       
              self.vars.append(nn.Parameter(w))
              self.vars.append(nn.Parameter(torch.zeros(self.layer_sizes[i+1])))
          w = torch.empty(self.layer_sizes[len(self.layer_sizes)-2], self.layer_sizes[len(self.layer_sizes)-1]).T
          nn.init.uniform_(w, -np.sqrt(6 / self.layer_sizes[len(self.layer_sizes)-2]) / self.hidden_omega_0, np.sqrt(6 / self.layer_sizes[len(self.layer_sizes)-2]) / self.hidden_omega_0)       
          self.vars.append(nn.Parameter(w))
          self.vars.append(nn.Parameter(torch.zeros(self.layer_sizes[len(self.layer_sizes)-1])))   
        else:
          # Layers
          for i in range(len(self.layer_sizes)-2):
            if i == 0:
              w = torch.empty(self.layer_sizes[i], self.layer_sizes[i+1]).T
              nn.init.uniform_(w)       
              self.vars.append(nn.Parameter(w))
              self.vars.append(nn.Parameter(torch.zeros(self.layer_sizes[i+1])))
            else:
              w = torch.empty(self.layer_sizes[i], self.layer_sizes[i+1]).T
              nn.init.uniform_(w)       
              self.vars.append(nn.Parameter(w))
              self.vars.append(nn.Parameter(torch.zeros(self.layer_sizes[i+1])))
          w = torch.empty(self.layer_sizes[len(self.layer_sizes)-2], self.layer_sizes[len(self.layer_sizes)-1]).T
          nn.init.uniform_(w)       
          self.vars.append(nn.Parameter(w))
          self.vars.append(nn.Parameter(torch.zeros(self.layer_sizes[len(self.layer_sizes)-1])))        

    def forward(self, X, given_vars=None):
      if given_vars is None:
          given_vars = self.vars
      idx = 0
      if self.use_sine_layer:
        # Layers
        for i in range(len(self.layer_sizes)-2):
            w, b = given_vars[idx], given_vars[idx + 1]
            if i == 0:
                X = torch.sin(self.first_omega_0*nn.functional.linear(X, w, b))
            else:
                X = torch.sin(self.hidden_omega_0*nn.functional.linear(X, w, b))
            idx += 2
        w, b = given_vars[idx], given_vars[idx + 1]
        idx += 2
        # X = 0.5+nn.functional.linear(X, w, b)
        X = nn.Sigmoid()(nn.functional.linear(X, w, b))
      else:
        # Layers
        for i in range(len(self.layer_sizes)-2):
            w, b = given_vars[idx], given_vars[idx + 1]
            X = nn.ReLU(nn.functional.linear(X, w, b))
            idx += 2
        w, b = given_vars[idx], given_vars[idx + 1]
        idx += 2
        X = nn.Sigmoid()(nn.functional.linear(X, w, b))
      assert idx == len(given_vars)        

      return X  