# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import os
import math
import tensorflow as tf
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
import lpips as lp
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.registration import phase_cross_correlation
import torch
import numpy as np
import abc
import logging

from models.utils import from_flattened_numpy, to_flattened_numpy
from scipy import integrate
import sde_lib
from models import utils as mutils
from utils import eprint

def vesde_discretize(x, t, sde):
    """SMLD(NCSN) discretization."""
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 sde.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G

def reverse_diffusion_discretize(x, t, model, sde):
  """Create discretized iteration rules for the reverse diffusion sampler."""
  f, G = vesde_discretize(x, t, sde)
  score = mutils.score_fn(model, sde, x, t, False)
  rev_f = f - G[:, None, None, None] ** 2 * score
  rev_G = G
  return rev_f, rev_G, score


def reverse_diffusion_update_fn(model, sde, x, t):
  f, G, score = reverse_diffusion_discretize(x, t, model, sde)
  z = torch.randn_like(x)
  x_mean = x - f
  noise2 = 1.00 * G[:, None, None, None] * z
  x = x_mean + noise2
  return x, x_mean, score

def langevin_update_fn(model, sde, x, t, target_snr, n_steps):
  alpha = torch.ones_like(t)

  for i in range(n_steps):
    grad = mutils.score_fn(model, sde, x, t, train=False)
    noise = torch.randn_like(x)
    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
    noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
    step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
    diff = step_size[:, None, None, None] * grad
    x_mean = x + diff
    noise2 = 1.00 * torch.sqrt(step_size * 2)[:, None, None, None] * noise
    x = x_mean + noise2

  return x, x_mean

def ve_sde(x, t):
  sigma_min = 0.01
  sigma_max = 50.0
  sigma = sigma_min * (sigma_max /sigma_min) ** t
  drift = torch.zeros_like(x)
  diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)),
                                              device=t.device))
  return drift, diffusion


def reverse_sde_old(model, sde, x, t, next_t, dt):
  drift, diffusion = ve_sde(x, t)
  score = mutils.score_fn(model, sde, x, t, False)
  drift = x - (drift - diffusion[:, None, None, None] ** 2 * score) * dt
  return score, drift, (diffusion * torch.sqrt(dt))[:, None, None, None]

def reverse_sde(model, sde, x, t, next_t):
  dt = (t - next_t)[:, None, None, None]
  score = mutils.score_fn(model, sde, x, t, False)
  sigma_min = 0.01
  sigma_max = 50.0

  # = diffusion^2 in old method ve_sde
  sigma_delta = (2 * (sigma_min * (sigma_max / sigma_min) ** t) ** 2 * math.log(sigma_max / sigma_min))[:, None, None, None]

  #sigma_i2 = (sigma_min * (sigma_max /sigma_min) ** t) ** 2
  #sigma_next_i2 = (sigma_min * (sigma_max /sigma_min) ** next_t) ** 2
  #sigma_delta = (sigma_i2 - sigma_next_i2)[:, None, None, None] / dt

  drift = x + sigma_delta * dt * score
  diffusion = torch.sqrt(sigma_delta * dt)
  return score, drift, diffusion


def euler_sampler(sample_dir, step, model, sde, shape, inverse_scaler, snr, n_steps, probability_flow, continuous, denoise, device, eps):
  # Initial sample
  x = sde.prior_sampling(shape).to(device)
  timesteps = torch.linspace(eps, sde.T, sde.N, device=device)


  for i in reversed(range(0, sde.N)):
    eprint("\rSampling: {}/{}".format(sde.N-i, sde.N), end='')

    x = x.requires_grad_()

    t = timesteps[i]
    vec_t = torch.ones(shape[0], device=t.device) * t
    dt = -1.0 / sde.N
    z = torch.randn_like(x)
    score, drift, diffusion = reverse_sde(model, sde, x, vec_t)
    x_mean = x + drift * dt
    new_x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z

    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas.to(t.device)[timestep]

    x_tweedie = x + sigma*sigma * score  

    x = new_x.detach()

    if i % 25 == 0:
      save_sample(sample_dir, step, sde.N-i, inverse_scaler(x.detach()))
        

  eprint()

  return inverse_scaler(x), sde.N * (n_steps + 1)


def euler_sampler_conditional(sample_dir, step, model, sde, shape, inverse_scaler, snr, n_steps, probability_flow, continuous, denoise, device, eps, measure_fn, anti_measure_fn, target_measurements, targets):
  # auf google colab bis 1024 batch size
  # wandb zum evaluieren
  # wandb: layken.mahrus@moosbay.com
  # mehr zahlen

  result = None
  total_results_measurement_error = None

  for run in range(1, 2):
    # STEPS nichtlinear?
    eprint("run " + str(run))
    steps = 1000
    iterations = 1
    eprint("steps=" + str(steps) + ", iters=" + str(iterations))
    timesteps = torch.linspace(eps, sde.T, steps, device=device)
    #timesteps[timesteps<0.5] = ((timesteps[timesteps<0.5] * 2.0) ** 0.5) / 2.0
    #timesteps[timesteps>0.5] = ((timesteps[timesteps>0.5] * 2.0 - 1.0) ** 2.0) / 2.0 + 0.5
    total_results = None
    for samplei in range(10, 10+iterations):
      # Initial sample
      x = sde.prior_sampling(shape).to(device)

      for i in reversed(range(1, steps)):
        eprint("\rSampling: {}/{}".format(steps-i, steps), end='')

        x = x.requires_grad_()

        # Time of the start of this step
        t = timesteps[i]
        next_t = timesteps[i-1]
        if i >= 2:
            nextnext_t = timesteps[i-2]
        else:
            nextnext_t = 0.0

        vec_t = torch.ones(shape[0], device=t.device) * t
        vec_next_t = torch.ones(shape[0], device=t.device) * next_t
        vec_nextnext_t = torch.ones(shape[0], device=t.device) * nextnext_t

        # Euler sampler
        z = torch.randn_like(x)
        # score, drift, diffusion = reverse_sde_old(model, sde, x, vec_t, vec_next_t, 1.0 / steps)
        score, drift, diffusion = reverse_sde(model, sde, x, vec_t, vec_next_t)
        new_x = drift + diffusion * z
        # PC sampler
        """
        x, x_mean = langevin_update_fn(model, sde, x, vec_t, snr, n_steps)
        new_x, x_mean, score = reverse_diffusion_update_fn(model, sde, x, vec_t)
        """

        # Gradient step
        sigma_min = 0.01
        sigma_max = 50.0
        sigma = sigma_min * (sigma_max /sigma_min) ** t
        actual_variance = x.var((0, 1)).sqrt()[0, :].mean()
        eprint("target variance=", sigma, " actual=", actual_variance, " error=", 1.0 - sigma / actual_variance)
        #new_x = new_x / actual_variance * sigma

        x_tweedie = x + sigma*sigma * score
        x_tweedie_measured = measure_fn(x_tweedie)
        diff = anti_measure_fn(x_tweedie, torch.abs(target_measurements - x_tweedie_measured))
        losses = torch.square(diff)
        losses = torch.sum(losses, (-1, -2), keepdim=True)
        lossessum = torch.sum(losses)
        x_grad = torch.autograd.grad(lossessum, x)[0]
        #sigma_delta = (2 * (sigma_min * (sigma_max / sigma_min) ** t) ** 2 * math.log(sigma_max / sigma_min))
        #dt = (t - next_t)
        new_x -= 0.05 * x_grad #/ losses.sqrt()

        # Manifold constraint
        # Take phase from new_x and amplitude from y_t
        # TODO: not every time, maybe every 10 iters?
        # if True or i < 60:
        if i % 10 == 0:
            # Renoising
            # score2 = mutils.score_fn(model, sde, new_x, vec_next_t, False)
            # sigma2 = sigma_min * (sigma_max /sigma_min) ** next_t
            # x_tweedie2 = new_x + sigma2*sigma2 * score2
            # new_x = anti_measure_fn(x_tweedie2, target_measurements)
            # z = torch.randn_like(x)
            # new_x = new_x + sigma2*z

            # Hypothetical constraint
            z = torch.randn_like(x)
            new_x = anti_measure_fn(new_x, measure_fn(sigma*z + targets))

        x = new_x.detach()

        if i == steps-1:
          save_sample(sample_dir, 0, steps-i, inverse_scaler(targets.detach()))
          save_sample(sample_dir, 1, steps-i, inverse_scaler(target_measurements.detach()))
        if i % 10 == 5:
          save_sample(sample_dir, 1000, steps-i, inverse_scaler(x_tweedie.detach()))
          # save_sample(sample_dir, 1000, steps-i, inverse_scaler(new_x.detach()))
          #save_sample(sample_dir, 1000, steps-i, inverse_scaler(x_tweedie2.detach()))
                    
        #elif (i-1) % 25 == 0:
        #  save_sample(sample_dir, step, steps-i, inverse_scaler(measure_fn(x_tweedie).detach()))
      result = x_tweedie
      result_measured = measure_fn(x_tweedie) # Use tweedie here so it's not constrained
      measurement_error = mse(result_measured, target_measurements)
      if total_results == None or total_results_measurement_error == None:
        total_results = result
        total_results_measurement_error = measurement_error
      else:
        comparisons = measurement_error < total_results_measurement_error
        total_results[comparisons] = result[comparisons]
        total_results_measurement_error[comparisons] = measurement_error[comparisons]
      #eprint("\n")
      registered_results = torch.tensor(register(targets.detach().cpu().numpy(), total_results.detach().cpu().numpy()))
      eprint("lpips: " + str(lpips(targets, registered_results).mean()))
      eprint("mse: " + str(mse(targets.cpu(), registered_results).mean().item()))
      eprint("mae: " + str(mae(targets.cpu(), registered_results).mean().item()))
      ssimresult = ssim(targets.detach().cpu().numpy(), registered_results.numpy())
      eprint("ssim: " + str(ssimresult.mean()))
      save_sample(sample_dir, run, samplei, inverse_scaler(total_results.detach()), str(ssimresult.mean()))


  return inverse_scaler(result), sde.N * (n_steps + 1)

def lpips(true, pred, mode='vgg'):
    assert mode in ['alex', 'vgg', 'squeeze']
    true = (true * 2 - 1).repeat(1, 1, 3, 1).cpu()
    pred = (pred * 2 - 1).repeat(1, 1, 3, 1).cpu()
    
    if true.shape[1] == 1:
        true = torch.tile(true, (1,3,1,1))
        pred = torch.tile(pred, (1,3,1,1))
    if true.shape[2] != 64:
        true = F.interpolate(true, size=64)
        pred = F.interpolate(pred, size=64)
    
    loss_fn = lp.LPIPS(net=mode, verbose=False)
    lps = loss_fn(true.float(), pred.float()).squeeze().detach().numpy()
    return lps

def mse(true, pred):
    mses = torch.mean((true - pred)**2, (1, 2, 3))
    return mses

def mae(true, pred):
    maes = torch.mean(torch.abs(true - pred), (1, 2, 3))
    return maes

def ssim(true, pred):
    # image should be in [0,1]
    ssims = np.zeros(len(pred))
    for i in range((len(pred))):
        ssims[i] = structural_similarity(true[i].transpose(1, 2, 0), 
                                         pred[i].transpose(1, 2, 0), 
                                         channel_axis=2,
                                         data_range=1)
    return ssims

def psnr(true, pred):
    psnrs = np.zeros(len(true))
    for i in range((len(true))):
        psnrs[i] = peak_signal_noise_ratio(true[i].transpose(1, 2, 0), 
                                           pred[i].transpose(1, 2, 0), 
                                           data_range=1)
    return psnrs

def register(true, pred):
    registered = np.zeros(pred.shape)

    for i in range(len(true)):
        t = true[i].transpose(1, 2, 0)
        p = pred[i].transpose(1, 2, 0)
        
        shifted_pred, err = cross_correlation(t, p)
        rotshifted_pred, rot_err = cross_correlation(
            t, np.rot90(p, k=2, axes=(0, 1))
        )
        
        if err <= rot_err:
            registered[i] = shifted_pred.transpose(2, 0, 1)  
        else:
            registered[i] = rotshifted_pred.transpose(2, 0, 1)

    return registered


def img2gray(rgb):
    if rgb.shape[2] == 1:
        return rgb.squeeze(2)

    return rgb[...,:3] @ np.array([0.2989, 0.5870, 0.1140])
def cross_correlation(true, pred):
    true_gray, pred_gray = img2gray(true), img2gray(pred)
    shifts, err, _ = phase_cross_correlation(
        true_gray, pred_gray, normalization=None
    )
    shifted = np.roll(pred, np.array(shifts).astype(int), axis=(0, 1))
    return shifted, err 

def pc_sampler(sample_dir, step, model, sde, shape, inverse_scaler, snr, n_steps, probability_flow, continuous, denoise, device, eps):
  with torch.no_grad():
    # Initial sample
    x = sde.prior_sampling(shape).to(device)
    timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

    for i in range(0, sde.N):
      eprint("\rSampling: {}/{}".format(i, sde.N), end='')
      t = timesteps[i]
      vec_t = torch.ones(shape[0], device=t.device) * t
      x, x_mean = langevin_update_fn(model, sde, x, vec_t, snr, n_steps)
      x, x_mean = reverse_diffusion_update_fn(model, sde, x, vec_t)
      if i % 50 == 0:
          save_sample(sample_dir, step, i, inverse_scaler(x_mean))

    eprint()
    return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)


def save_sample(sample_dir, step, iter, sample, ssim=None):
    #this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
    #tf.io.gfile.makedirs(this_sample_dir)
    nrow = int(np.sqrt(sample.shape[0]))
    image_grid = make_grid(sample, nrow, padding=2)
    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    #with tf.io.gfile.GFile(
    #    os.path.join(this_sample_dir, "sample_{}.np".format(iter)), "wb") as fout:
    #    np.save(fout, sample)

    if ssim==None:
        with tf.io.gfile.GFile(
            os.path.join(sample_dir, "sample_{}.png".format(step)), "wb") as fout:
            save_image(image_grid, fout)
    else:
        with tf.io.gfile.GFile(
            os.path.join(sample_dir, "sample_{}_ssim={}.png".format(step, ssim)), "wb") as fout:
            save_image(image_grid, fout)
        
