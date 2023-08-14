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
import tensorflow as tf
from torchvision.utils import make_grid, save_image
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
  rev_f = f - G[:, None, None, None] ** 2 * mutils.score_fn(model, sde, x, t, False)
  rev_G = G
  return rev_f, rev_G


def reverse_diffusion_update_fn(model, sde, x, t):
  f, G = reverse_diffusion_discretize(x, t, model, sde)
  z = torch.randn_like(x)
  x_mean = x - f
  noise2 = 1.00 * G[:, None, None, None] * z
  x = x_mean + noise2
  return x, x_mean

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


def reverse_sde(model, sde, x, t):
  drift, diffusion = ve_sde(x, t)
  score = mutils.score_fn(model, sde, x, t, False)
  drift = drift - diffusion[:, None, None, None] ** 2 * score
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

    target = torch.zeros_like(x[0])
    target[0,[0, 1, 30, 31],:] = 1.00
    target[1,[0, 1, 30, 31],:] = 1.00
    target[2,[0, 1, 30, 31],:] = 1.00
    mask = target != 0.0

    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas.to(t.device)[timestep]
    x_tweedie = x + sigma*sigma * score  
    losses = torch.square(mask * (target - x_tweedie))
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    x_grad = torch.autograd.grad(losses, x)[0]

    new_x -= 1.0 / torch.sqrt(losses) * x_grad

    # manifold constraint
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(target, t)
    perturbed_data = mean + std * z
    new_x = mask * perturbed_data + (~mask) * new_x

    x = new_x.detach()

    if i % 25 == 0:
      save_sample(sample_dir, step, sde.N-i, inverse_scaler(x_tweedie.detach()))

  eprint()

  return inverse_scaler(x), sde.N * (n_steps + 1)

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


def save_sample(sample_dir, step, iter, sample):
    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
    tf.io.gfile.makedirs(this_sample_dir)
    nrow = int(np.sqrt(sample.shape[0]))
    image_grid = make_grid(sample, nrow, padding=2)
    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    #with tf.io.gfile.GFile(
    #    os.path.join(this_sample_dir, "sample_{}.np".format(iter)), "wb") as fout:
    #    np.save(fout, sample)

    with tf.io.gfile.GFile(
        os.path.join(this_sample_dir, "sample_{}.png".format(iter)), "wb") as fout:
        save_image(image_grid, fout)
