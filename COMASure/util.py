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
from fourier_transformation import FourierFeature

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

plt.rc('font', **font)

# This function can be treated as a version of meshgrid, but results in a 6-value vector for each pixel
def generate_coordinates(image_size_x, image_size_y, use_sine_layer, coordinate_sys_option, use_trimensional = False): 
    if coordinate_sys_option == 6:

      X = []
      for x in range(0, image_size_x):
          for y in range(0, image_size_y):

              # For 1 pixel's coordinate: 

              if use_sine_layer: # all coordinate values need to be within [-1,1] (except radius since negative sign is controlled by angle) - condition of Periodic Activation Function input
                # Coordinates from top left corner of the image
                x_tl = x/image_size_x
                y_tl = y/image_size_y

                # Coordinates from bottom right corner of the image
                x_br = 1 - x_tl
                y_br = 1 - y_tl

                # Radius and angle in radians from center of the image
                x_c = (x_tl - 1 / 2)
                y_c = (y_tl - 1 / 2)
                r = ((x_c ** 2 + y_c ** 2) ** 0.5) / (((1/2) ** 2 + (1/2) ** 2) ** 0.5)
                a = math.atan2(y_c, x_c)/math.pi

                # X.append((x_tl*2-1, y_tl*2-1, x_br*2-1, y_br*2-1, r, a))
                X.append((x_tl*2-1, y_tl*2-1, x_br*2-1, y_br*2-1, x_c*2, y_c*2))
              
              else: # all coordinate values need to be within [0,1] - condition of Fourier Feature transformation
                # Coordinates from top left corner of the image
                x_tl = x/image_size_x
                y_tl = y/image_size_y

                # Coordinates from bottom right corner of the image
                x_br = 1 - x_tl
                y_br = 1 - y_tl

                # Radius and angle in radians from center of the image
                x_c = (x_tl - 1 / 2)
                y_c = (y_tl - 1 / 2)
                r = ((x_c ** 2 + y_c ** 2) ** 0.5) / (((1/2) ** 2 + (1/2) ** 2) ** 0.5)
                a = (math.atan2(y_c, x_c)/math.pi + 1)/2
                
                # X.append((x_tl, y_tl, x_br, y_br, r, a))
                X.append((x_tl, y_tl, x_br, y_br, x_c, y_c))             

      X = torch.Tensor(X)
      if (use_trimensional == False): # Flattened version
          return X
      else:
        return X.reshape((1, image_size_x, image_size_y, 6))

    elif coordinate_sys_option == 2:
      eps_i = 1 / (2*(image_size_x-1))
      eps_j = 1 / (2*(image_size_y-1))
      X = torch.stack(torch.meshgrid(torch.linspace(eps_i, 1-eps_i, image_size_x), torch.linspace(eps_j, 1-eps_j, image_size_y))).reshape(2, -1).T 
      if (use_trimensional == False): # Flattened version
          return X
      else:
        return X.reshape((1, image_size_x, image_size_y, 6))      

def generate_proper_img_tensor(img_tensor, use_trimensional = False):
    Y = img_tensor.permute(1,2,0) # Ensure correct sequence of row and col pixels when flatten (row+row+...)
    if use_trimensional == False: # Flattened version
        return Y.reshape((-1,3))
    else:
        return torch.unsqueeze(Y, 0)
    
def split_img_in_batches(coords, scaling_factor = 1):
  _, H, W, _ = coords.shape
  batch_H = H//scaling_factor
  batch_W = W//scaling_factor
  batches = []

  for i in range(scaling_factor):
    for j in range(scaling_factor):  
      if i < scaling_factor-1 or j < scaling_factor-1:
        batches.append(coords[:, batch_H*i : batch_H*(i+1), batch_W*j : batch_W*(j+1), :])
      elif i == scaling_factor-1 and j < scaling_factor-1:
        batches.append(coords[:, batch_H*i : H, batch_W*j : batch_W*(j+1), :])
      elif i < scaling_factor-1 and j == scaling_factor-1:
        batches.append(coords[:, batch_H*i : batch_H*(i+1), batch_W*j : W, :])
      elif i == scaling_factor-1 and j == scaling_factor-1:
        batches.append(coords[:, batch_H*i : H, batch_W*j : W, :])
  
  return batches

def add_noise_to_coords(coords_base, factor):
  with torch.no_grad():
    return torch.clamp(factor*torch.randn(coords_base.shape)+coords_base, min=0.0, max=1.0)

def add_noise_to_target(target_base, factor):
  with torch.no_grad():
    return torch.clamp(factor*torch.rand(target_base.shape)/255.0+target_base, min=0.0, max=1.0)

def add_noise_to_coords_cuda(coords_base, factor):
  with torch.no_grad():
    return torch.clamp(factor*torch.randn(coords_base.shape).cuda()+coords_base.cuda(), min=0.0, max=1.0)

def add_noise_to_target_cuda(target_base, factor):
  with torch.no_grad():
    return torch.clamp(factor*torch.rand(target_base.shape).cuda()/255.0+target_base.cuda(), min=0.0, max=1.0)

def mse(x, x_hat):
  return ((x-x_hat)**2).mean()

def psnr(x, x_hat):
  return compare_psnr(x, x_hat)

def ssim(x, x_hat):
  # return compare_ssim(x, x_hat, multichannel = True) # This one is not really correct, ignore
  x = x.unsqueeze(0).cpu().detach()
  x_hat = x_hat.unsqueeze(0).cpu().detach()
  return ssim_sys(x, x_hat, data_range=1.0, size_average=False)

def print_performance(img_a_base, img_b_base, to_print=True):
  img_a = img_a_base.permute(1,2,0)
  img_b = img_b_base.permute(1,2,0)
  img_a = img_a.cpu().detach().numpy()
  img_b = img_b.cpu().detach().numpy()
  PSNR_value = psnr(img_a, img_b).item()
  SSIM_value = ssim(img_a_base, img_b_base).item()
  if to_print:
      print("PSNR:", PSNR_value)
      print("SSIM:", SSIM_value)
      print("MSE Loss (w/o Regularizer):", mse(img_a, img_b).item())
      print("L1 Loss (w/o Regularizer):", nn.L1Loss()(img_a_base, img_b_base).item())
  return PSNR_value, SSIM_value

def generate_patches(X_lr_oigin, X_hr_oigin, patch_size_lr, patch_count, downscaling_factor, get_random_patch=False):
  _, h, w = X_lr_oigin.shape
  h_range = (0, h-patch_size_lr) # Inclusive
  w_range = (0, w-patch_size_lr) # Inclusive
  lr_patchs = []
  hr_patchs = []
  for _ in range(patch_count):
    # LR
    if get_random_patch:
        h_start = random.randint(h_range[0],h_range[1])
        w_start = random.randint(w_range[0],w_range[1])
        h_end = h_start+patch_size_lr
        w_end = w_start+patch_size_lr
    else:
        one_side = patch_size_lr//2
        h_start = h//2-one_side
        w_start = w//2-one_side
        h_end = h_start+patch_size_lr
        w_end = w_start+patch_size_lr    
    
    lr_patchs.append(X_lr_oigin[:, h_start:h_end, w_start:w_end])
    
    # HR
    hr_patchs.append(X_hr_oigin[:, h_start*downscaling_factor:h_end*downscaling_factor, w_start*downscaling_factor:w_end*downscaling_factor])
    # print(
    #     h_start, 
    #     h_end, 
    #     w_start, 
    #     w_end
    #   )
    # print(
    #     h_start*downscaling_factor, 
    #     h_end*downscaling_factor,
    #     w_start*downscaling_factor, 
    #     w_end*downscaling_factor
    #   )
  return lr_patchs, hr_patchs

def obtain_images(img_name_list, dataset_name, img_name_list_start, img_name_list_end, img_type, downscaling_factor, downscaling_method, requires_hr, is_train_set):

    src = "/content/"
    
    IM_lr_dict = {}
    X_lr_dict = {}
    IM_hr_dict = {}
    X_hr_dict = {}
    
    to_tensor = torchvision.transforms.ToTensor()
    to_image = torchvision.transforms.ToPILImage()

    for img_name in img_name_list:
    
      if dataset_name == "DIV2K":
    
        if is_train_set:
          if img_name_list_start == "0001" and img_name_list_end == "0800":
              IM_hr = Image.open(src+"DIV2K_train_HR/{}.{}".format(img_name,img_type))
          else:
              IM_hr = Image.open(src+"DIV2K_train_HR_{}_{}/{}.{}".format(img_name_list_start,img_name_list_end,img_name,img_type))
        else:
          IM_hr = Image.open(src+"DIV2K_valid_HR/{}.{}".format(img_name,img_type))
        X_hr = to_tensor(IM_hr)
    
        if downscaling_factor == 1:
          IM_lr = IM_hr
          X_lr = X_hr
        else:
          if downscaling_method == 'Bicubic':
            if is_train_set:
              IM_lr = Image.open(src+"DIV2K_train_LR_bicubic/X{}/{}x{}.{}".format(downscaling_factor,img_name,downscaling_factor,img_type))
            else:
              IM_lr = Image.open(src+"DIV2K_valid_LR_bicubic/X{}/{}x{}.{}".format(downscaling_factor,img_name,downscaling_factor,img_type))
            X_lr = to_tensor(IM_lr)
          elif downscaling_method == 'Subsample':
            X_lr = X_hr[:,::downscaling_factor,::downscaling_factor] 
            IM_lr = to_image(X_lr)    
    
      elif dataset_name == "Celeb-A":
    
        IM_hr = Image.open("/content/gdrive/MyDrive/MSc DSML/UCL/Main/COMP0158 MSc DSML Project/Datasets/"+"Celeb-A/{}.{}".format(img_name,img_type))
        X_hr = to_tensor(IM_hr)
    
        if downscaling_factor == 1:
          IM_lr = IM_hr
          X_lr = X_hr
        else:
          if downscaling_method == 'Bicubic':
            IM_lr = IM_hr.resize((IM_hr.size[0]//downscaling_factor,IM_hr.size[1]//downscaling_factor))
            X_lr = to_tensor(IM_lr)
          elif downscaling_method == 'Subsample':
            X_lr = X_hr[:,::downscaling_factor,::downscaling_factor] 
            IM_lr = to_image(X_lr)
      
      if requires_hr:
        IM_hr_dict[img_name] = IM_hr
        X_hr_dict[img_name] = X_hr
      else:
        IM_lr_dict[img_name] = IM_lr
        X_lr_dict[img_name] = X_lr
    
    return IM_lr_dict, X_lr_dict, IM_hr_dict, X_hr_dict

def chunks(ls):
  result = []
  for i in range(len(ls)):
    support = [ls[i]]
    query = [ls[i]]
    result.append({"Support":support, "Query":query})
  return result

def assign_img_data_to_tasks(tasks, dataset_name, img_name_list_start, img_name_list_end, img_type, downscaling_factor, downscaling_method, is_train_set):
    result = []
    for i in range(len(tasks)):
        task = tasks[i]
        query = task["Query"]
        support = task["Support"]
        query_data =  obtain_images(query, dataset_name, img_name_list_start, img_name_list_end, img_type, downscaling_factor, downscaling_method, True, is_train_set)
        support_data =  obtain_images(support, dataset_name, img_name_list_start, img_name_list_end, img_type, downscaling_factor, downscaling_method, False, is_train_set)
        result.append({"Support":support_data, "Query":query_data})
    return result        

def generate_patches_meta_support_query(task_count, query_set_size, support_set_size, tasks, tasks_with_imgs, patch_size_lr, patch_count, downscaling_factor):

    tasks_with_imgs_patchs = []
    for i in range(task_count):
      result = {}
    
      query_data_lr_patches = {}
      query_data_hr_patches = {}
   
      support_data_lr_patches = {}
      support_data_hr_patches = {}

      for j in range(support_set_size):
        img_name = tasks[i]["Support"][j]
        X_lr_origin = tasks_with_imgs[i]["Support"][1][img_name]
        X_hr_origin = tasks_with_imgs[i]["Query"][3][img_name] 
        lr_patchs, hr_patchs = generate_patches(X_lr_origin, X_hr_origin, patch_size_lr, patch_count, downscaling_factor) ##
        support_data_lr_patches[img_name] = lr_patchs
        query_data_hr_patches[img_name] = hr_patchs
        
      result["Support"] = (support_data_lr_patches, support_data_hr_patches)
      result["Query"] = (query_data_lr_patches, query_data_hr_patches)
      tasks_with_imgs_patchs.append(result)
     
    return tasks_with_imgs_patchs

def process_img_tensor(task_count, set_size, patch_count, data_type, tasks, tasks_with_imgs_patchs, hr, 
                            use_trimensional, use_sine_layer, coordinate_sys_option, fourier_feature_size, sigma, use_fourier,
                            phi, use_noise, coord_noise_factor, target_noise_factor):
  target_lists = []

  for i in range(task_count):
    input_list = []
    target_list = []
    for j in range(set_size):
      img_name = tasks[i][data_type][j]
      for k in range(patch_count):
        patch = tasks_with_imgs_patchs[i][data_type][hr][img_name][k] 

        target = generate_proper_img_tensor(patch, use_trimensional)
        target_list.append(target)

        if len(input_list) == 0: # No need to repeatedly generate coordinates for the same image patch size
          C, H, W = patch.shape
          if phi.B.is_cuda:
              coords = generate_coordinates(H, W, use_sine_layer, coordinate_sys_option, use_trimensional).cuda()
          else:
              coords = generate_coordinates(H, W, use_sine_layer, coordinate_sys_option, use_trimensional)
          if use_fourier:
            input = phi(coords)
          else:
            input = coords
          input_list.append(input)

    target_lists.append(target_list)
        
  return input_list[0], target_lists, C, H, W

def prepare_meta_test_image(selected_images_test, task_count_test, support_set_size_test, query_set_size_test, img_name_list_test, dataset_name, 
              img_name_list_test_start, img_name_list_test_end, img_type, downscaling_factor, downscaling_method,
              patch_size_lr, patch_count, use_trimensional, use_sine_layer, coordinate_sys_option, fourier_feature_size, sigma, use_fourier,
              phi, use_noise, coord_noise_factor, target_noise_factor):
  
  to_tensor = torchvision.transforms.ToTensor()
  to_image = torchvision.transforms.ToPILImage()

  tasks_test = chunks(selected_images_test)
  tasks_with_imgs_test = assign_img_data_to_tasks(tasks_test, dataset_name, img_name_list_test_start, img_name_list_test_end, img_type, downscaling_factor, downscaling_method, False)
  tasks_with_imgs_patchs_test = generate_patches_meta_support_query(task_count_test, query_set_size_test, support_set_size_test, tasks_test, tasks_with_imgs_test, patch_size_lr, patch_count, downscaling_factor)

  # Meta-Test - Support: LR (same image)
  meta_test_support_input, meta_test_support_target, meta_test_support_C, meta_test_support_H, meta_test_support_W = process_img_tensor(task_count_test, support_set_size_test, patch_count, "Support", tasks_test, tasks_with_imgs_patchs_test, 0, 
                                        use_trimensional, use_sine_layer, coordinate_sys_option, fourier_feature_size, sigma, use_fourier,
                                        phi, use_noise, coord_noise_factor, target_noise_factor)

  # Meta-Test - Query: HR (same image) 
  meta_test_query_input, meta_test_query_target, meta_test_query_C, meta_test_query_H, meta_test_query_W = process_img_tensor(task_count_test, support_set_size_test, patch_count, "Query", tasks_test, tasks_with_imgs_patchs_test, 1,
                                      use_trimensional, use_sine_layer, coordinate_sys_option, fourier_feature_size, sigma, use_fourier,
                                      phi, use_noise, coord_noise_factor, target_noise_factor)    

  X_lr = meta_test_support_target[0][0].reshape(meta_test_support_H, meta_test_support_W, meta_test_support_C).permute(2,0,1)
  X_hr = meta_test_query_target[0][0].reshape(meta_test_query_H, meta_test_query_W, meta_test_query_C).permute(2,0,1)
  coords_base = generate_coordinates(X_lr.size(1), X_lr.size(2), use_sine_layer, coordinate_sys_option, use_trimensional)
  target_base = generate_proper_img_tensor(X_lr, use_trimensional)
  coords_hr = generate_coordinates(X_hr.size(1), X_hr.size(2), use_sine_layer, coordinate_sys_option, use_trimensional)
  target_hr = generate_proper_img_tensor(X_hr, use_trimensional)
  
  img_name = tasks_test[0]["Support"][0]
  support = tasks_with_imgs_patchs_test[0]["Support"][0][img_name][0] # LR
  query = tasks_with_imgs_patchs_test[0]["Query"][1][img_name][0] # HR
  IM_lr = to_image(support)
  IM_hr = to_image(query)
  bicubic_prediction = to_tensor(IM_lr.resize(IM_hr.size, resample=Image.BICUBIC))
  ground_truth = query

  if IM_lr.size == IM_hr.size:
      PSNR_bicubic_prediction, SSIM_bicubic_prediction = np.inf, 1.0
  else:
      PSNR_bicubic_prediction, SSIM_bicubic_prediction = print_performance(ground_truth, bicubic_prediction, False)

  return coords_base, target_base, meta_test_support_C, meta_test_support_H, meta_test_support_W, meta_test_query_C, meta_test_query_H, meta_test_query_W, coords_hr, target_hr, PSNR_bicubic_prediction, SSIM_bicubic_prediction, bicubic_prediction, ground_truth