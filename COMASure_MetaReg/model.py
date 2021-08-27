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
from tqdm.notebook import tqdm
from time import sleep

from PIL import Image
import matplotlib.pyplot as plt

from util import add_noise_to_coords, add_noise_to_target, split_img_in_batches, mse, generate_coordinates, print_performance
from network import MLP

def train_model(coords_base, phi, use_fourier, target_base, use_noise, coord_noise_factor, target_noise_factor, H, W, no_epochs, evaluation_epochs, model, 
                opt, scheduler, reg_coeff, use_regularization, use_batching, scaling_factor, use_trimensional, use_lr_decay,
                name, alpha,
                test_X, test_trimensional, test_batching, test_scaling_factor,
                coordinate_sys_option,
                use_sine_layer):

    losses = []
    epoch_list = list(range(no_epochs))
    best_PSNR_value = 0.0
    best_model = copy.deepcopy(model)
    best_epoch = -1
    
    for epoch in tqdm(epoch_list):
        current_epoch = epoch + 1

        if use_noise:
          coords = add_noise_to_coords(coords_base, coord_noise_factor/(2**(current_epoch/evaluation_epochs)))
          target = add_noise_to_target(target_base, target_noise_factor/(2**(current_epoch/evaluation_epochs)))
        else:
          coords = coords_base
          target = target_base

        #########################################################
        # Batch Case (specifically for large images)
        #########################################################
        if use_batching:
          if use_trimensional:
            coords_in_batches = split_img_in_batches(coords, scaling_factor)
            target_in_batches = split_img_in_batches(target, scaling_factor)
            batch_losses = []
            for i in range(len(coords_in_batches)):   
              opt.zero_grad()
              inp = coords_in_batches[i].reshape((-1,coordinate_sys_option))
              out = target_in_batches[i].reshape((-1,3))
              _, H_this_batch, W_this_batch, _ = coords_in_batches[i].shape
              if use_fourier:
                input_one_batch = phi(inp).cuda()
              else:
                input_one_batch = inp.cuda()   
              target_one_batch = out.cuda()  
              result = model.forward(input_one_batch)
              if name == "L2 (MSE)":
                loss_no_reg = mse(result, target_one_batch)
              elif name == "L1":
                loss_no_reg = nn.L1Loss()(result, target_one_batch)
              elif name == "L2+SSIM":
                MSE = mse(result, target_one_batch)
                SSIM_loss = 1 - ssim_sys(result.T.reshape((1, 3, H_this_batch, W_this_batch)), target_one_batch.T.reshape((1, 3, H_this_batch, W_this_batch)), data_range=1.0, size_average=True)
                loss_no_reg = (1-alpha)*MSE+alpha*SSIM_loss
              elif name == "L1+SSIM":
                L1 = nn.L1Loss()(result, target_one_batch)
                SSIM_loss = 1 - ssim_sys(result.T.reshape((1, 3, H_this_batch, W_this_batch)), target_one_batch.T.reshape((1, 3, H_this_batch, W_this_batch)), data_range=1.0, size_average=True)
                loss_no_reg = (1-alpha)*L1+alpha*SSIM_loss
              elif name == "L2+MS-SSIM":
                MSE = mse(result, target_one_batch)
                MS_SSIM_loss = 1 - ms_ssim_sys(result.T.reshape((1, 3, H_this_batch, W_this_batch)), target_one_batch.T.reshape((1, 3, H_this_batch, W_this_batch)), data_range=1.0, size_average=True)
                loss_no_reg = (1-alpha)*MSE+alpha*MS_SSIM_loss
              elif name == "L1+MS-SSIM":
                L1 = nn.L1Loss()(result, target_one_batch)
                MS_SSIM_loss = 1 - ms_ssim_sys(result.T.reshape((1, 3, H_this_batch, W_this_batch)), target_one_batch.T.reshape((1, 3, H_this_batch, W_this_batch)), data_range=1.0, size_average=True)
                loss_no_reg = (1-alpha)*L1+alpha*MS_SSIM_loss
              if use_regularization:
                # L2 Norm
                l2_loss = 0.0
                for param in model.parameters():
                  l2_loss += torch.norm(param)**2
                loss = loss_no_reg + reg_coeff * l2_loss
                loss.backward()
              else:
                loss_no_reg.backward()         
              opt.step()
              batch_losses.append(loss_no_reg.item())
            losses.append(np.mean(batch_losses))

          else:
            coords_in_batches = torch.split(coords, coords.shape[0]//(scaling_factor**2), dim = 0)
            target_in_batches = torch.split(target, target.shape[0]//(scaling_factor**2), dim = 0)
            batch_losses = []
            for i in range(len(coords_in_batches)):   
              opt.zero_grad()
              inp = coords_in_batches[i]
              out = target_in_batches[i]
              if use_fourier:
                input_one_batch = phi(inp).cuda()
              else:
                input_one_batch = inp.cuda()   
              target_one_batch = out.cuda()  
              result = model.forward(input_one_batch)
              side_length = int(np.sqrt(result.shape[0]))
              if side_length > 1:
                result_copy = result[:side_length**2,:]
                target_one_batch_copy = target_one_batch[:side_length**2,:]
                if name == "L2 (MSE)":
                  loss_no_reg = mse(result, target_one_batch)
                elif name == "L1":
                  loss_no_reg = nn.L1Loss()(result, target_one_batch)
                elif name == "L2+SSIM":
                  MSE = mse(result, target_one_batch)
                  SSIM_loss = 1 - ssim_sys(result_copy.T.reshape((1, 3, side_length, side_length)), target_one_batch_copy.T.reshape((1, 3, side_length, side_length)), data_range=1.0, size_average=True)
                  loss_no_reg = (1-alpha)*MSE+alpha*SSIM_loss
                elif name == "L1+SSIM":
                  L1 = nn.L1Loss()(result, target_one_batch)
                  SSIM_loss = 1 - ssim_sys(result_copy.T.reshape((1, 3, side_length, side_length)), target_one_batch_copy.T.reshape((1, 3, side_length, side_length)), data_range=1.0, size_average=True)
                  loss_no_reg = (1-alpha)*L1+alpha*SSIM_loss
                elif name == "L2+MS-SSIM":
                  MSE = mse(result, target_one_batch)
                  MS_SSIM_loss = 1 - ms_ssim_sys(result_copy.T.reshape((1, 3, side_length, side_length)), target_one_batch_copy.T.reshape((1, 3, side_length, side_length)), data_range=1.0, size_average=True)
                  loss_no_reg = (1-alpha)*MSE+alpha*MS_SSIM_loss
                elif name == "L1+MS-SSIM":
                  L1 = nn.L1Loss()(result, target_one_batch)
                  MS_SSIM_loss = 1 - ms_ssim_sys(result_copy.T.reshape((1, 3, side_length, side_length)), target_one_batch_copy.T.reshape((1, 3, side_length, side_length)), data_range=1.0, size_average=True)
                  loss_no_reg = (1-alpha)*L1+alpha*MS_SSIM_loss
                if use_regularization:
                  # L2 Norm
                  l2_loss = 0.0
                  for param in model.parameters():
                    l2_loss += torch.norm(param)**2
                  loss = loss_no_reg + reg_coeff * l2_loss
                  loss.backward()
                else:
                  loss_no_reg.backward()         
                opt.step()
                batch_losses.append(loss_no_reg.item())
            losses.append(np.mean(batch_losses))

        #########################################################
        # Non-batch Case (for small images)
        #########################################################
        else:  
          opt.zero_grad()  
          if use_fourier:
            input = phi(coords).cuda()
          else:
            input = coords.cuda()
          result = model.forward(input)
          out = target.cuda()
          if name == "L2 (MSE)":
            loss_no_reg = mse(result, out)
          elif name == "L1":
            loss_no_reg = nn.L1Loss()(result, out)
          elif name == "L2+SSIM":
            MSE = mse(result, out)
            SSIM_loss = 1 - ssim_sys(result.T.reshape((1, 3, H, W)), out.T.reshape((1, 3, H, W)), data_range=1.0, size_average=True)
            loss_no_reg = (1-alpha)*MSE+alpha*SSIM_loss
          elif name == "L1+SSIM":
            L1 = nn.L1Loss()(result, out)
            SSIM_loss = 1 - ssim_sys(result.T.reshape((1, 3, H, W)), out.T.reshape((1, 3, H, W)), data_range=1.0, size_average=True)
            loss_no_reg = (1-alpha)*L1+alpha*SSIM_loss
          elif name == "L2+MS-SSIM":
            MSE = mse(result, out)
            MS_SSIM_loss = 1 - ms_ssim_sys(result.T.reshape((1, 3, H, W)), out.T.reshape((1, 3, H, W)), data_range=1.0, size_average=True)
            loss_no_reg = (1-alpha)*MSE+alpha*MS_SSIM_loss
          elif name == "L1+MS-SSIM":
            L1 = nn.L1Loss()(result, out)
            MS_SSIM_loss = 1 - ms_ssim_sys(result.T.reshape((1, 3, H, W)), out.T.reshape((1, 3, H, W)), data_range=1.0, size_average=True)
            loss_no_reg = (1-alpha)*L1+alpha*MS_SSIM_loss
          if use_regularization:
            # L2 Norm
            l2_loss = 0.0
            for param in model.parameters():
              l2_loss += torch.norm(param)**2
            loss = loss_no_reg + reg_coeff * l2_loss
            loss.backward()
          else:
            loss_no_reg.backward()        
          opt.step()
          losses.append(loss_no_reg.item())

        if use_lr_decay and current_epoch % evaluation_epochs == 0:
          scheduler.step()   

        #########################################################
        # Display and Update
        #########################################################

        # if current_epoch % 100 == 0: 
        #   print('[Epoch: %d]' % (current_epoch))
        #   print('Train Loss: %.5f' % (losses[-1]))
        if current_epoch % evaluation_epochs == 0: 
          coords_SR = generate_coordinates(test_X.size(1), test_X.size(2), use_sine_layer, coordinate_sys_option, test_trimensional)
          recon_SR = eval_model(coords_SR, phi, use_fourier, model, test_batching, test_scaling_factor, test_X)
          PSNR_value, SSIM_value = print_performance(test_X.cuda(), recon_SR, False)   
          if PSNR_value > best_PSNR_value:
             best_PSNR_value = PSNR_value
             best_model = copy.deepcopy(model)
             best_epoch = current_epoch

        sleep(0.05)
    
    print("Best Epoch: {}, Best Recon PSNR: {}".format(best_epoch, best_PSNR_value))
    print("Completed!")
    
    return losses, best_model

def eval_model(coords, phi, use_fourier, model, use_batching, scaling_factor, X):

  with torch.no_grad():
    if use_batching:
      coords_in_batches = torch.split(coords, coords.shape[0]//(scaling_factor**2), dim = 0)

    if use_batching:
      result_list = []
      for i in range(len(coords_in_batches)):   
        inp = coords_in_batches[i]
        if use_fourier:
          input_one_batch = phi(inp).cuda()
        else:
          input_one_batch = inp.cuda()   
        result = model.forward(input_one_batch)
        result_list.append(result)
      recon = torch.cat(result_list, dim = 0).T.reshape(X.shape)

    else:   
      if use_fourier:
        input = phi(coords).cuda()
      else:
        input = coords.cuda()
      result = model.forward(input)
      recon = result.T.reshape(X.shape)
  
  return recon


def eval_model_cpu(coords, phi, use_fourier, model, use_batching, scaling_factor, X):

  with torch.no_grad():
    if use_batching:
      coords_in_batches = torch.split(coords, coords.shape[0]//(scaling_factor**2), dim = 0)

    if use_batching:
      result_list = []
      for i in range(len(coords_in_batches)):   
        inp = coords_in_batches[i]
        if use_fourier:
          input_one_batch = phi(inp)
        else:
          input_one_batch = inp   
        result = model.forward(input_one_batch)
        result_list.append(result)
      recon = torch.cat(result_list, dim = 0).T.reshape(X.shape)

    else:   
      if use_fourier:
        input = phi(coords)
      else:
        input = coords
      result = model.forward(input)
      recon = result.T.reshape(X.shape)
  
  return recon


def process_one_img(
        is_training_mode, loss_choices, layer_sizes, use_sine_layer,
        lr, gamma, coords_base, phi, use_fourier, target_base, use_noise, 
        coord_noise_factor, target_noise_factor, X_lr, no_epochs, evaluation_epochs,
        reg_coeff, use_regularization, use_batching, scaling_factor, use_trimensional,
        use_lr_decay, X_hr, test_scaling_factor, coordinate_sys_option,
        hyper_para_dict, version, dataset_name, img_name
        ):
    
    print("Processing Image: {}".format(img_name))
    
    if is_training_mode:
      print("Training Models")
      losses_for_diff_loss_choices = {}
      recons_for_diff_loss_choices = {}
      network_for_diff_loss_choices = {}
      for name, alphas in loss_choices.items():
        temp_loss = []
        temp_recon = []
        temp_network = {}
        for alpha in alphas:
          print("--------------------------")
          print(name + " (Alpha = " + str(alpha) + ")")
          network = MLP(layer_sizes, use_sine_layer).cuda()
          opt = torch.optim.Adam(network.parameters(), lr=lr)
          scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
          loss, best_network = train_model(
                    coords_base, phi, use_fourier, target_base, use_noise, coord_noise_factor, target_noise_factor,
                    X_lr.size(1), X_lr.size(2), 
                    no_epochs, evaluation_epochs, 
                    network, opt, scheduler, reg_coeff, use_regularization, 
                    use_batching, scaling_factor, use_trimensional,
                    use_lr_decay, name, alpha,
                    X_hr, False, True, test_scaling_factor,
                    coordinate_sys_option,
                    use_sine_layer)
          recon = eval_model(coords_base, phi, use_fourier, network, use_batching, scaling_factor, X_lr)
          temp_loss.append(loss)
          temp_recon.append(recon) 
          temp_network[alpha] = copy.deepcopy(best_network)
          torch.save({
                'epoch': no_epochs,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss,
                'recon': recon,
                'hyper_para_dict': hyper_para_dict
                }, "/content/gdrive/My Drive/MSc DSML/UCL/Main/COMP0158 MSc DSML Project/coordinate_net_{}_{}_{}_{}_{}.ckpt".format(version, dataset_name, img_name, name, str(alpha)))
        losses_for_diff_loss_choices[name] = temp_loss
        recons_for_diff_loss_choices[name] = temp_recon
        network_for_diff_loss_choices[name] = temp_network
    
    else:
      print("Loading Saved Models")
      losses_for_diff_loss_choices = {}
      recons_for_diff_loss_choices = {}
      network_for_diff_loss_choices = {}
      for name, alphas in loss_choices.items():
        temp_loss = []
        temp_recon = []
        temp_network = {}
        for alpha in alphas:
          print("=======================================================")
          print(name + " (Alpha = " + str(alpha) + ")")
          network = MLP(layer_sizes, use_sine_layer).cuda()
          opt = torch.optim.Adam(network.parameters(), lr=lr)
          scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
          checkpoint = torch.load("/content/gdrive/My Drive/MSc DSML/UCL/Main/COMP0158 MSc DSML Project/coordinate_net_{}_{}_{}_{}_{}.ckpt".format(version, dataset_name, img_name, name, str(alpha)))
          network.load_state_dict(checkpoint['model_state_dict'])
          opt.load_state_dict(checkpoint['optimizer_state_dict'])
          loss = checkpoint['loss']
          recon = checkpoint['recon']
          temp_loss.append(loss)
          temp_recon.append(recon) 
          temp_network[alpha] = copy.deepcopy(network)
        losses_for_diff_loss_choices[name] = temp_loss
        recons_for_diff_loss_choices[name] = temp_recon
        network_for_diff_loss_choices[name] = temp_network

    return losses_for_diff_loss_choices, recons_for_diff_loss_choices, network_for_diff_loss_choices
  
    
def compare_recon_lr(loss_choices, recons_for_diff_loss_choices, X_lr, img_name):

    print("Compare Recon Against Ground Truth For Image: {}".format(img_name))    
    
    for name, alphas in loss_choices.items():
      for idx, alpha in enumerate(alphas):
        recon = recons_for_diff_loss_choices[name][idx]
        plt.figure(figsize=(28,6))
        plt.subplot(1,1,1)
        plt.imshow(recon.permute(1,2,0).cpu())
        plt.title(name + " (Alpha = " + str(alpha) + ")")
      plt.show()
    plt.figure(figsize=(28,6))
    plt.subplot(1,1,1)
    plt.imshow(X_lr.permute(1,2,0))
    plt.title('Ground Truth')
    plt.show()


def compare_output_gt(X, Y, X_name, Y_name, img_name):

    print("Compare {} Against {} For Image: {}".format(X_name, Y_name, img_name))    
    
    plt.figure(figsize=(28,6))
    plt.subplot(1,1,1)
    plt.imshow(X.permute(1,2,0).cpu())
    plt.title("{}".format(X_name))
    plt.show()
    
    plt.figure(figsize=(28,6))
    plt.subplot(1,1,1)
    plt.imshow(Y.permute(1,2,0).cpu())
    plt.title("{}".format(Y_name))
    plt.show()    


def print_training_logs(loss_choices, losses_for_diff_loss_choices, img_name):
    
    print("Printing Training Logs For Image: {}".format(img_name))  
    
    plt.figure(figsize=(25,10))
    for name, alphas in loss_choices.items():
        for idx, alpha in enumerate(alphas):
          loss = losses_for_diff_loss_choices[name][idx]
          plt.subplot(1,2,1)
          plt.plot(loss, label = name + " (Alpha = " + str(alpha) + ")")
    plt.xlabel('Epoch')
    plt.ylabel('Loss (w/o Regularizer)')
    plt.title('Losses of different loss functions')
    plt.legend()
    plt.show()
    
    
def print_training_logs_2(loss_choices, losses_for_diff_loss_choices, img_name):

    epochs = []
    x_labels = []
    for name, alphas in loss_choices.items():
      for idx, alpha in enumerate(alphas):
        epochs.append(len(losses_for_diff_loss_choices[name][idx]))
        x_labels.append(name + " (Alpha = " + str(alpha) + ")")
    epochs = pd.Series(epochs)
    
    plt.figure(figsize=(12, 8))
    ax = epochs.plot(kind='bar')
    ax.set_xlabel('Loss Functions')
    ax.set_ylabel('Epochs')
    ax.set_xticklabels(x_labels)
    ax.set_title('# of epochs for different loss functions')
    
    rects = ax.patches
    
    labels = epochs
    
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')


def print_recon(coords_base, phi, use_fourier, network, use_batching, scaling_factor, X_lr, to_image, img_name):
    
    print("Printing Recon For Image: {}".format(img_name)) 

    recon = eval_model(coords_base, phi, use_fourier, network, use_batching, scaling_factor, X_lr)
    compare_output_gt(recon, X_lr, "Recon", "Ground Truth", img_name)
    PSNR_value, SSIM_value = print_performance(X_lr.cuda(), recon)
    return PSNR_value, SSIM_value


def print_sr_interpolation(X_hr, use_sine_layer, coordinate_sys_option, phi, use_fourier, network, scaling_factor, to_image, img_name):
    
    # Note that this no longer fits on GPU in a single; Could be handled by minibatching over pixels.
    print("Printing SR Interpolation For Image: {}".format(img_name)) 
    
    coords_SR = generate_coordinates(X_hr.size(1), X_hr.size(2), use_sine_layer, coordinate_sys_option, use_trimensional = False)
    recon_SR = eval_model(coords_SR, phi, use_fourier, network, True, scaling_factor, X_hr)
    compare_output_gt(recon_SR, X_hr, "SR", "Ground Truth", img_name) 
    PSNR_value, SSIM_value = print_performance(X_hr.cuda(), recon_SR) 
    return PSNR_value, SSIM_value


def print_sr_interpolation_arbitrary(network, IM_lr, X_lr, upsample_scale, use_sine_layer, 
                                     phi, use_fourier, scaling_factor, to_image,
                                     coordinate_sys_option, img_name):
    
    print("Printing SR Interpolation For Arbitrary Scale For Image: {}".format(img_name)) 

    network_copy = copy.deepcopy(network).to("cpu")
    upsample_scale = 5
    X_upsample = torch.zeros((3, X_lr.size(1)*upsample_scale, X_lr.size(2)*upsample_scale))
    
    coords_upsample = generate_coordinates(X_lr.size(1)*upsample_scale, X_lr.size(2)*upsample_scale, use_sine_layer, coordinate_sys_option, use_trimensional = False)
    recon_upsample = eval_model_cpu(coords_upsample, phi, use_fourier, network_copy, True, scaling_factor, X_upsample)
    to_image(recon_upsample)


def print_bicubic_interpolation_arbitrary(X_lr, IM_lr, upsample_scale, img_name):
    
    print("Printing Bicubic SR For Arbitrary Scale For Image: {}".format(img_name)) 
    
    baseline_bicubic_upsample = IM_lr.resize((X_lr.size(2)*upsample_scale, X_lr.size(1)*upsample_scale), resample=Image.BICUBIC)
    baseline_bicubic_upsample
    

def print_baselines_nearest(IM_lr, IM_hr, img_name):
    
    print("Printing SR Baseline (Nearest) For Image: {}".format(img_name)) 
    
    baseline_nearest = IM_lr.resize(IM_hr.size, resample=Image.NEAREST)
    return baseline_nearest


def print_baselines_bilinear(IM_lr, IM_hr, img_name):
    
    print("Printing SR Baseline (Bilinear) For Image: {}".format(img_name)) 
    
    baseline_bilinear = IM_lr.resize(IM_hr.size, resample=Image.BILINEAR)
    return baseline_bilinear


def print_baselines_bicubic(IM_lr, IM_hr, img_name):
    
    print("Printing SR Baseline (Bicubic) For Image: {}".format(img_name)) 
    
    baseline_bicubic = IM_lr.resize(IM_hr.size, resample=Image.BICUBIC)
    return baseline_bicubic


def eval_performance_mean_sd(performance_dict, metric, purpose, img_name_list, loss_choices, patching, patch_count):
    
    if patching:
        for name, alphas in loss_choices.items():
            for alpha in alphas:
                for img_name in img_name_list:   
                    values = []
                    for i in range(patch_count):
                        value = performance_dict[img_name][i][name][alpha]
                        values.append(value)
                    m = np.mean(values)
                    sd = np.std(values)
                    print("Printing {} {} For Current Model Architecture/Hyperparameters ({} {}) For {}, Mean = {}, Standard Deviation = {}".format(purpose, metric, name, alpha, img_name, m, sd)) 
    else:
        for name, alphas in loss_choices.items():
            for alpha in alphas:
                values = []
                for img_name in img_name_list:        
                    value = performance_dict[img_name][name][alpha]
                    values.append(value)
                m = np.mean(values)
                sd = np.std(values)
                print("Printing {} {} For Current Model Architecture/Hyperparameters ({} {}), Mean = {}, Standard Deviation = {}".format(purpose, metric, name, alpha, m, sd)) 


def eval_performance_mean_sd_baseline(performance_dict, metric, purpose, img_name_list, patching, patch_count):
    
    if patching:
        for img_name in img_name_list:   
            values = []
            for i in range(patch_count):
                value = performance_dict[img_name][i]
                values.append(value)
            m = np.mean(values)
            sd = np.std(values)
            print("Printing {} {} For Current Model Architecture/Hyperparameters For {}, Mean = {}, Standard Deviation = {}".format(purpose, metric, img_name, m, sd)) 
    else:
        values = []
        for img_name in img_name_list:        
            value = performance_dict[img_name]
            values.append(value)
        m = np.mean(values)
        sd = np.std(values)
        print("Printing {} {} For Current Model Architecture/Hyperparameters, Mean = {}, Standard Deviation = {}".format(purpose, metric, m, sd)) 
