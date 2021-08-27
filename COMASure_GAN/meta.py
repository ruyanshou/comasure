from collections import OrderedDict
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
import pprint
# Import modules

from util import *
from sine_layer import *
from network import *
from fourier_transformation import *
from model import *
from feature_extractor import *
from discriminator import *

font = {'family' : 'normal',
      'weight' : 'bold',
      'size'  : 15}

plt.rc('font', **font)

class Meta(nn.Module):

  def __init__(self, 
            task_count, tasks, dataset_name, img_name_list_start, img_name_list_end, img_type, downscaling_factor, downscaling_method, patch_size_lr,

            update_lr, meta_lr, meta_test_update_lr,

            support_set_size, query_set_size,

            update_step,
            layer_sizes, use_sine_layer,
            patch_count, fourier_feature_size, sigma,
            use_noise, coord_noise_factor, target_noise_factor, coordinate_sys_option,
            use_trimensional, use_fourier, alpha, use_regularization, reg_coeff,
            
            gamma, phi
          ):

    super(Meta, self).__init__()

    self.task_num = task_count
    self.tasks = tasks
    self.dataset_name = dataset_name
    self.img_name_list_start = img_name_list_start
    self.img_name_list_end = img_name_list_end
    self.img_type = img_type
    self.downscaling_factor = downscaling_factor
    self.downscaling_method = downscaling_method
    self.patch_size_lr = patch_size_lr

    self.update_lr = update_lr # Inner loop lr
    self.meta_lr = meta_lr # Outer loop lr
    self.meta_test_update_lr = meta_test_update_lr # Meta-Test lr

    self.support_set_size = support_set_size
    self.query_set_size = query_set_size

    self.update_step = update_step # Number of gradient updates at Meta-Training time
    self.layer_sizes = layer_sizes
    self.use_sine_layer = use_sine_layer
    self.patch_count = patch_count
    self.fourier_feature_size = fourier_feature_size
    self.sigma = sigma
    self.use_noise = use_noise
    self.coord_noise_factor = coord_noise_factor
    self.target_noise_factor = target_noise_factor
    self.coordinate_sys_option = coordinate_sys_option
    self.use_trimensional = use_trimensional
    self.use_fourier = use_fourier
    self.alpha = alpha
    self.use_regularization = use_regularization
    self.reg_coeff = reg_coeff
    
    self.gamma = gamma

    self.phi = copy.deepcopy(phi)

    self.net = MLP(self.layer_sizes, self.use_sine_layer).cuda()
    self.meta_opt = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
    self.criterion = nn.L1Loss()
    # self.criterion = nn.MSELoss()

    self.discriminator = Discriminator(3, patch_size_lr*downscaling_factor, patch_size_lr*downscaling_factor).cuda()
    self.feature_extractor = FeatureExtractor().cuda()
    self.feature_extractor.eval() # Evaluation mode without gradient calculation
    self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5,0.999))
    # self.criterion_content = nn.L1Loss()
    self.criterion_content = nn.MSELoss()
    # self.criterion_GAN = nn.MSELoss()
    self.criterion_GAN = nn.BCELoss()

  def compute_loss(self, result, out, fast_weights, C, H, W, mode, discriminator):
    loss = 0

    loss = loss + self.criterion(result, out)
    SSIM_loss = 1 - ssim_sys(result.T.reshape((1, C, H, W)), out.T.reshape((1, C, H, W)), data_range=1.0, size_average=True)
    loss = (1-self.alpha)*loss+self.alpha*SSIM_loss
    if self.use_regularization:
      l2_loss = 0.0
      for param in fast_weights:
        l2_loss += torch.norm(param)**2
      loss = loss + self.reg_coeff * l2_loss
    # print(loss)

    # Content Loss
    gen_features = self.feature_extractor(result.reshape(H, W, C).permute(2,0,1).unsqueeze(0))
    real_features = self.feature_extractor(out.reshape(H, W, C).permute(2,0,1).unsqueeze(0))
    loss_content = self.criterion_content(gen_features, real_features.detach()) 
    loss = loss + 0.06*loss_content 

    # Adversarial Loss
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    valid = Variable(Tensor(np.ones((1))), requires_grad=False) 
    loss_GAN = self.criterion_GAN(discriminator(result.reshape(H, W, C).permute(2,0,1).unsqueeze(0), mode), valid) 
    loss = loss + 0.1*loss_GAN

    # print("normal: {}, {}".format(loss_content, 0.1*loss_GAN))

    return loss, 0.06*loss_content, 0.1*loss_GAN

  def compute_discriminator_loss(self, result, out, C, H, W, mode):

    # Discriminator Loss
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    valid = Variable(Tensor(np.ones((1))), requires_grad=False) 
    fake = Variable(Tensor(np.zeros((1))), requires_grad=False)  
    loss_real = self.criterion_GAN(self.discriminator(out.reshape(H, W, C).permute(2,0,1).unsqueeze(0).cuda(), mode), valid)
    loss_fake = self.criterion_GAN(self.discriminator(result.clone().detach().reshape(H, W, C).permute(2,0,1).unsqueeze(0), mode), fake)
    loss_d_total = (loss_real + loss_fake) / 2

    # print("discriminator: {}".format(loss_d_total))

    return loss_d_total, loss_real, loss_fake

  def compute_meta_loss(self, tasks, tasks_with_imgs, tasks_with_imgs_patchs, fast_weights, loss_type):

    # Meta-Train - Query: HR
    meta_train_query_input, meta_train_query_target, meta_train_query_C, meta_train_query_H, meta_train_query_W = process_img_tensor(len(tasks), self.query_set_size, self.patch_count, "Query", tasks, tasks_with_imgs_patchs, 1, 
                                          self.use_trimensional, self.use_sine_layer, self.coordinate_sys_option, self.fourier_feature_size, self.sigma, self.use_fourier,
                                          self.phi, self.use_noise, self.coord_noise_factor, self.target_noise_factor)
    
    if loss_type == "perceptual":
    
      total_loss = 0.0
      total_loss_content = 0.0
      total_loss_adv = 0.0

      psnr_total = 0.0
      ssim_total = 0.0

      for out in meta_train_query_target[0]:
        result = self.net.forward(meta_train_query_input.cuda(), given_vars=fast_weights)
        loss, loss_content, loss_adv = self.compute_loss(result, out.cuda(), fast_weights, meta_train_query_C, meta_train_query_H, meta_train_query_W, "query", copy.deepcopy(self.discriminator))
        total_loss = total_loss + loss
        total_loss_content = total_loss_content + loss_content
        total_loss_adv = total_loss_adv + loss_adv
        
        PSNR_value, SSIM_value = print_performance(out.cuda().reshape(meta_train_query_H, meta_train_query_W, meta_train_query_C).permute(2,0,1)
                              , result.reshape(meta_train_query_H, meta_train_query_W, meta_train_query_C).permute(2,0,1)
                              , False) 
        psnr_total = psnr_total + PSNR_value
        ssim_total = ssim_total + SSIM_value

      return total_loss/(self.query_set_size*self.patch_count), total_loss_content/(self.query_set_size*self.patch_count), total_loss_adv/(self.query_set_size*self.patch_count), \
          psnr_total/(self.query_set_size*self.patch_count), ssim_total/(self.query_set_size*self.patch_count)

    else:

      total_discriminator_loss = 0.0
      total_loss_real = 0.0
      total_loss_fake = 0.0

      for out in meta_train_query_target[0]:
        result = self.net.forward(meta_train_query_input.cuda(), given_vars=fast_weights)
        discriminator_loss, loss_real, loss_fake = self.compute_discriminator_loss(result.clone().detach(), out.cuda(), meta_train_query_C, meta_train_query_H, meta_train_query_W, "query")
        total_discriminator_loss = total_discriminator_loss + discriminator_loss
        total_loss_real = total_loss_real + loss_real
        total_loss_fake = total_loss_fake + loss_fake

      return total_discriminator_loss/(self.query_set_size*self.patch_count), total_loss_real/(self.query_set_size*self.patch_count), total_loss_fake/(self.query_set_size*self.patch_count)

  ########## Meta-Training Time ##########

  def execute_grad_step(self, tasks, tasks_with_imgs, tasks_with_imgs_patchs, fast_weights, step):

    # Meta-Train - Support: LR
    meta_train_support_input, meta_train_support_target, meta_train_support_C, meta_train_support_H, meta_train_support_W = process_img_tensor(len(tasks), self.support_set_size, self.patch_count, "Support", tasks, tasks_with_imgs_patchs, 0, 
                                          self.use_trimensional, self.use_sine_layer, self.coordinate_sys_option, self.fourier_feature_size, self.sigma, self.use_fourier,
                                          self.phi, self.use_noise, self.coord_noise_factor, self.target_noise_factor)

    for out in meta_train_support_target[0]:
      
      # self.opt_d.zero_grad()
        
      if step == 0:
          result = self.net.forward(meta_train_support_input.cuda())
          loss, _, _ = self.compute_loss(result, out.cuda(), self.net.parameters(), meta_train_support_C, meta_train_support_H, meta_train_support_W, "support", copy.deepcopy(self.discriminator))
          grad = torch.autograd.grad(loss, self.net.parameters())

          # Check no gradient for network parameters during grad step
          if self.net.vars[0].grad != None:
            assert len(torch.nonzero(self.net.vars[0].grad)) < 1
    
          fast_weights = list(map(lambda p: p[1]-self.update_lr*p[0]/(self.support_set_size*self.patch_count) , zip(grad, self.net.parameters())))
    
          # Check no gradient for network parameters during grad step
          if self.net.vars[0].grad != None:
            assert len(torch.nonzero(self.net.vars[0].grad)) < 1
    
      else:
          result = self.net.forward(meta_train_support_input.cuda(), given_vars=fast_weights)
          loss, _, _ = self.compute_loss(result, out.cuda(), fast_weights, meta_train_support_C, meta_train_support_H, meta_train_support_W, "support", copy.deepcopy(self.discriminator))
          grad = torch.autograd.grad(loss, fast_weights, retain_graph = True, create_graph=True)     
    
          # Check no gradient for network parameters during grad step
          if self.net.vars[0].grad != None:
            assert len(torch.nonzero(self.net.vars[0].grad)) < 1
    
          fast_weights = list(map(lambda p: p[1]-self.update_lr*p[0]/(self.support_set_size*self.patch_count) , zip(grad, fast_weights)))
    
          # Check no gradient for network parameters during grad step
          if self.net.vars[0].grad != None:
            assert len(torch.nonzero(self.net.vars[0].grad)) < 1

      # loss_discriminator, _, _ = self.compute_discriminator_loss(result.clone().detach(), out.cuda(), meta_train_support_C, meta_train_support_H, meta_train_support_W, "support")
      # loss_discriminator.backward()
      # self.opt_d.step() 

    return fast_weights

  def meta_train(self, all_tasks=None):

    if all_tasks == None:
      task_num_list = list(range(self.task_num))
      random.shuffle(task_num_list) # Randomly shuffle tasks
    else:
      task_num_list = list(range(len(all_tasks)))
      random.shuffle(task_num_list) # Randomly shuffle tasks      

    psnr_avg_list = []
    ssim_avg_list = []
    fast_weights_list = []

    self.meta_opt.zero_grad()
    self.opt_d.zero_grad()

    total_meta_loss = 0
    total_content_loss = 0
    total_adv_loss = 0

    for i in task_num_list: # Outer loop
      
      fast_weights = list(map(lambda p: p[0], zip(self.net.parameters()))) # Initialize as original network paras (theta)

      # Each time load 1 image to avoid exceeding memory limit    
      if all_tasks == None:
        tasks = self.tasks[i:i+1]
      else:
        tasks = all_tasks[i:i+1]
      # print("Task {}".format(tasks[0]))

      tasks_with_imgs = assign_img_data_to_tasks(tasks, self.dataset_name, self.img_name_list_start, self.img_name_list_end, self.img_type, self.downscaling_factor, self.downscaling_method, True)
      tasks_with_imgs_patchs = generate_patches_meta_support_query(len(tasks), self.query_set_size, self.support_set_size, tasks, tasks_with_imgs, self.patch_size_lr, self.patch_count, self.downscaling_factor)

      for step in range(self.update_step): # Inner loop
        # print("Step {}".format(step+1))
        fast_weights = self.execute_grad_step(tasks, tasks_with_imgs, tasks_with_imgs_patchs, fast_weights, step)

      fast_weights_list.append(fast_weights)

      meta_loss, content_loss, adv_loss, psnr_avg, ssim_avg = self.compute_meta_loss(tasks, tasks_with_imgs, tasks_with_imgs_patchs, fast_weights, "perceptual")
      total_meta_loss = total_meta_loss + meta_loss
      total_content_loss = total_content_loss + content_loss
      total_adv_loss = total_adv_loss + adv_loss

      psnr_avg_list.append(psnr_avg)
      ssim_avg_list.append(ssim_avg)   

    # Check no gradient for network parameters before backward
    if self.net.vars[0].grad != None:
      assert len(torch.nonzero(self.net.vars[0].grad)) < 1

    (total_meta_loss/len(task_num_list)).backward()
    # print("meta loss:{}".format(total_meta_loss/len(task_num_list)))
    
    # Check there is gradient for network parameters after backward
    assert self.net.vars[0].grad != None

    self.meta_opt.step() 

    total_discriminator_loss = 0
    total_real_loss = 0
    total_fake_loss = 0

    for i in task_num_list: # Outer loop
      
      fast_weights = fast_weights_list[i]

      # Each time load 1 image to avoid exceeding memory limit    
      if all_tasks == None:
        tasks = self.tasks[i:i+1]
      else:
        tasks = all_tasks[i:i+1]
      # print("Task {}".format(tasks[0]))

      tasks_with_imgs = assign_img_data_to_tasks(tasks, self.dataset_name, self.img_name_list_start, self.img_name_list_end, self.img_type, self.downscaling_factor, self.downscaling_method, True)
      tasks_with_imgs_patchs = generate_patches_meta_support_query(len(tasks), self.query_set_size, self.support_set_size, tasks, tasks_with_imgs, self.patch_size_lr, self.patch_count, self.downscaling_factor)

      discriminator_loss, real_loss, fake_loss = self.compute_meta_loss(tasks, tasks_with_imgs, tasks_with_imgs_patchs, fast_weights, "discriminator")
      total_discriminator_loss = total_discriminator_loss + discriminator_loss
      total_real_loss = total_real_loss + real_loss
      total_fake_loss = total_fake_loss + fake_loss

    (total_discriminator_loss/len(task_num_list)).backward()
    # print("meta discriminator loss:{}".format(total_discriminator_loss/len(task_num_list)))

    self.opt_d.step()

    return np.mean(np.array(psnr_avg_list)), np.mean(np.array(ssim_avg_list)), \
        total_meta_loss/len(task_num_list), total_content_loss/len(task_num_list), total_adv_loss/len(task_num_list), \
        total_discriminator_loss/len(task_num_list), total_real_loss/len(task_num_list), total_fake_loss/len(task_num_list)

  ########## Meta-Testing Time ##########

  def meta_test_query(self, input, net):

    # Query

    with torch.no_grad():     
      result = net.forward(input)
    return result

  def meta_test_with_ground_truth(self, coords_base, target_base, meta_test_support_H, meta_test_support_W, meta_test_support_C, meta_test_query_H, meta_test_query_W, meta_test_query_C, update_step_test, meta_test_evaluation_steps, coords_hr, target_hr):

    # A copy for meta-test
    net = copy.deepcopy(self.net) 
    update_lr = copy.deepcopy(self.meta_test_update_lr)
    gamma = copy.deepcopy(self.gamma)
    opt = torch.optim.Adam(net.parameters(), lr=update_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    best_PSNR_value = 0.0
    best_SSIM_value = 0.0
    best_step = -1
    sr_interpolate_results = []
    steps = []

    if self.use_fourier:
      phi_coords_hr = self.phi(coords_hr.cuda())
      phi_coords_base = self.phi(coords_base.cuda())
    else:
      phi_coords_hr = coords_hr.cuda()
      phi_coords_base = coords_base.cuda()    

    # Support

    # step_list = list(range(update_step_test))
    # for i in tqdm(step_list):
    for i in range(update_step_test):

      if self.use_noise:
        coords = add_noise_to_coords(coords_base, self.coord_noise_factor/(2**((i+1)/meta_test_evaluation_steps)))
        target = add_noise_to_target(target_base, self.target_noise_factor/(2**((i+1)/meta_test_evaluation_steps)))
      else:
        coords = coords_base
        target = target_base

      if self.use_fourier: 
        input = self.phi(coords.cuda())
      else:
        input = coords.cuda()

      if i==0: # before learning taking place
        steps.append(i)
        sr_interpolate_result = self.meta_test_query(phi_coords_hr, net)
        sr_interpolate_results.append(sr_interpolate_result) # SR Interpolate
        PSNR_value, SSIM_value = print_performance(target_hr.cuda().reshape(meta_test_query_H, meta_test_query_W, meta_test_query_C).permute(2,0,1)
                                , sr_interpolate_result.reshape(meta_test_query_H, meta_test_query_W, meta_test_query_C).permute(2,0,1)
                                , False) 
        if PSNR_value > best_PSNR_value:
            best_PSNR_value = PSNR_value
            best_step = i+1
        if SSIM_value > best_SSIM_value:
            best_SSIM_value = SSIM_value

      opt.zero_grad() 
      result = net.forward(input)
      out = target.cuda()
      loss, _, _ = self.compute_loss(result, out, net.parameters(), meta_test_support_C, meta_test_support_H, meta_test_support_W, "support", copy.deepcopy(self.discriminator))
      loss.backward()       
      opt.step()

      if (i+1)<=10 or ((i+1)<=200 and (i+1)%meta_test_evaluation_steps==0) or ((i+1)<=2000 and (i+1)%1000==0) or i+1==update_step_test:
        steps.append(i+1)
        sr_interpolate_result = self.meta_test_query(phi_coords_hr, net)
        sr_interpolate_results.append(sr_interpolate_result) # SR Interpolate
        PSNR_value, SSIM_value = print_performance(target_hr.cuda().reshape(meta_test_query_H, meta_test_query_W, meta_test_query_C).permute(2,0,1)
                                , sr_interpolate_result.reshape(meta_test_query_H, meta_test_query_W, meta_test_query_C).permute(2,0,1)
                                , False) 
        if PSNR_value > best_PSNR_value:
            best_PSNR_value = PSNR_value
            best_step = i+1
        if SSIM_value > best_SSIM_value:
            best_SSIM_value = SSIM_value

      if (i+1) % meta_test_evaluation_steps == 0:
        scheduler.step()   

    result = sr_interpolate_results[-1]
    loss, loss_content, loss_adv = self.compute_loss(result, target_hr.cuda(), net.parameters(), meta_test_query_C, meta_test_query_H, meta_test_query_W, "query", copy.deepcopy(self.discriminator))
    discriminator_loss, loss_real, loss_fake = self.compute_discriminator_loss(result.clone().detach(), target_hr.cuda(), meta_test_query_C, meta_test_query_H, meta_test_query_W, "query")

    del net

    return sr_interpolate_results, best_step, best_PSNR_value, best_SSIM_value, steps, loss, loss_content, loss_adv, discriminator_loss, loss_real, loss_fake
    
  def meta_test_without_ground_truth(self, coords_base, target_base, meta_test_support_C, meta_test_support_H, meta_test_support_W, update_step_test, meta_test_evaluation_steps, coords_hr):

    # A copy for meta-test
    net = copy.deepcopy(self.net) 
    update_lr = copy.deepcopy(self.meta_test_update_lr)
    gamma = copy.deepcopy(self.gamma)
    opt = torch.optim.Adam(net.parameters(), lr=update_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    best_PSNR_value = 0.0
    best_SSIM_value = 0.0
    best_step = -1
    sr_interpolate_results = []
    recon_results = []
    steps = []

    if self.use_fourier:
      phi_coords_hr = self.phi(coords_hr.cuda())
      phi_coords_base = self.phi(coords_base.cuda())
    else:
      phi_coords_hr = coords_hr.cuda()
      phi_coords_base = coords_base.cuda()    

    # Support

    # step_list = list(range(update_step_test))
    # for i in tqdm(step_list):
    for i in range(update_step_test):

      if self.use_noise:
        coords = add_noise_to_coords(coords_base, self.coord_noise_factor/(2**((i+1)/meta_test_evaluation_steps)))
        target = add_noise_to_target(target_base, self.target_noise_factor/(2**((i+1)/meta_test_evaluation_steps)))
      else:
        coords = coords_base
        target = target_base

      if self.use_fourier: 
        input = self.phi(coords.cuda())
      else:
        input = coords.cuda()

      if i==0: # before learning taking place
        steps.append(i)
        sr_interpolate_results.append(self.meta_test_query(phi_coords_hr, net)) # SR Interpolate
        recon = self.meta_test_query(phi_coords_base, net)
        recon_results.append(recon) # Recon
        PSNR_value, SSIM_value = print_performance(out.reshape(meta_test_support_H, meta_test_support_W, meta_test_support_C).permute(2,0,1)
                                , recon.reshape(meta_test_support_H, meta_test_support_W, meta_test_support_C).permute(2,0,1)
                                , False) 
        if PSNR_value > best_PSNR_value:
            best_PSNR_value = PSNR_value
            best_step = i+1
        if SSIM_value > best_SSIM_value:
            best_SSIM_value = SSIM_value

      opt.zero_grad() 
      result = net.forward(input)
      out = target.cuda()
      loss, _, _ = self.compute_loss(result, out, net.parameters(), meta_test_support_C, meta_test_support_H, meta_test_support_W, "support", copy.deepcopy(self.discriminator))
      loss.backward()       
      opt.step()

      if (i+1)<=10 or ((i+1)<=200 and (i+1)%meta_test_evaluation_steps==0) or ((i+1)<=2000 and (i+1)%1000==0) or i+1==update_step_test:
        steps.append(i+1)
        sr_interpolate_results.append(self.meta_test_query(phi_coords_hr, net)) # SR Interpolate
        recon = self.meta_test_query(phi_coords_base, net)
        recon_results.append(recon) # Recon
        PSNR_value, SSIM_value = print_performance(out.reshape(meta_test_support_H, meta_test_support_W, meta_test_support_C).permute(2,0,1)
                                , recon.reshape(meta_test_support_H, meta_test_support_W, meta_test_support_C).permute(2,0,1)
                                , False) 
        if PSNR_value > best_PSNR_value:
            best_PSNR_value = PSNR_value
            best_step = i+1
        if SSIM_value > best_SSIM_value:
            best_SSIM_value = SSIM_value

      if (i+1) % meta_test_evaluation_steps == 0:
        scheduler.step()   

    del net

    return sr_interpolate_results, recon_results, best_step, best_PSNR_value, best_SSIM_value, steps