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
"""Training and evaluation for score-based generative models. """

# import to get rid of this error:
#undefined symbol: cudaGraphDebugDotPrint, version libcudart.so.11.0
import torch

import gc
import io
import os
import time
import sys

import numpy as np
import tensorflow as tf
import torchinfo
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
#import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.transforms import GaussianBlur
from torchvision.utils import make_grid, save_image
from utils import eprint, save_checkpoint, restore_checkpoint

# import wandb

FLAGS = flags.FLAGS

def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # wandb
  try:
    wandb.init(
      # set the wandb project where this run will be logged
      project="uni",
    
      # track hyperparameters and run metadata
      config={
          "dataset": "mnist-smallermodel",
      }
    )
  except:
    eprint("wandb failed")
    

  sample_dir = os.path.join(workdir, "samples")
  sample_dir = os.path.join(sample_dir, str(int(time.time())))
  tf.io.gfile.makedirs(sample_dir)

  # Initialize model.
  model = mutils.create_model(config)
  ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, model.parameters())
  state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Always use VESDE
  sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, device=config.device)
  sampling_eps = 1e-5

  # Building sampling functions
  sampling_shape = (config.training.batch_size, config.data.num_channels,
                    config.data.image_size, config.data.image_size)

  num_train_steps = config.training.n_iters

  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    # Execute one training step
    loss = losses.step_fn(state, sde, batch, config, train)

    try:
      wandb.log({"loss": loss})
    except:
      pass

    if step % 100 == 0:
      logging.info("\nstep: %d, training_loss: %.5e" % (step, loss.item()))

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % 5000 == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    eprint("\rTraining: {}/{}".format(step, num_train_steps), end='')

    # Report the loss on an evaluation dataset periodically
    if step % 500 == 0:
      eprint();
      eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      eval_loss = losses.step_fn(state, sde, eval_batch, config, train)
      logging.info("\nstep: %d, eval_loss: %.5e" % (step, eval_loss.item()))

    # Save a checkpoint periodically and generate samples if needed
    if step != -1 and step % 100 == 0 or step == num_train_steps:
      eprint();
      # Save the checkpoint.
      save_step = step
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      def measure_fn(image):
        measurements = torch.abs(torch.fft.fft2(image))
        return measurements
      perturbed_data = measure_fn(eval_batch)
      t = torch.full((eval_batch.shape[0],), sde.T, device=eval_batch.device)
      result = mutils.score_fn(model, sde, perturbed_data, t, train)

      registered_results = torch.tensor(sampling.register(eval_batch.detach().cpu().numpy(), result.detach().cpu().numpy()))
      eprint("lpips: " + str(sampling.lpips(eval_batch, registered_results).mean()))
      eprint("mse: " + str(sampling.mse(eval_batch.cpu(), registered_results).mean().item()))
      eprint("mae: " + str(sampling.mae(eval_batch.cpu(), registered_results).mean().item()))
      ssimresult = sampling.ssim(eval_batch.detach().cpu().numpy(), registered_results.numpy())
      eprint("ssim: " + str(ssimresult.mean()))

      nrow = int(np.sqrt(eval_batch.shape[0]))
      # normal
      image_grid = make_grid(eval_batch, nrow, padding=2)
      with tf.io.gfile.GFile(
          os.path.join(sample_dir, "train_sample_{}_before.png".format(step)), "wb") as fout:
          save_image(image_grid, fout)
      # reconstructed
      image_grid = make_grid(result, nrow, padding=2)
      with tf.io.gfile.GFile(
          os.path.join(sample_dir, "train_sample_{}_after.png".format(step)), "wb") as fout:
          save_image(image_grid, fout)

      # Generate and save samples
      if False and config.training.snapshot_sampling:
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        sample, n = sampling.euler_sampler(sample_dir, step, model, sde, sampling_shape, inverse_scaler, config.sampling.snr, config.sampling.n_steps_each, config.sampling.probability_flow, config.training.continuous, config.sampling.noise_removal, config.device, sampling_eps)
        ema.restore(model.parameters())


def sample(config, workdir):
  sample_dir = os.path.join(workdir, "samples")
  sample_dir = os.path.join(sample_dir, str(int(time.time())))
  tf.io.gfile.makedirs(sample_dir)

  # Initialize model.
  model = mutils.create_model(config)

  # Building sampling functions
  sampling_shape = (config.training.batch_size, config.data.num_channels,
                    config.data.image_size, config.data.image_size)

  ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, model.parameters())
  state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Always use VESDE
  sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, device=config.device)
  sampling_eps = 1e-5

  eprint(torchinfo.summary(model, input_size=sampling_shape))

  for step in range(0, 1):
    # Generate and save samples
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    image = next(eval_iter)['image']._numpy()
    #image = tf.image.decode_image(tf.io.read_file("input.jpg"))._numpy().reshape(1, 32, 32, 3) / 255.0
    batch = torch.from_numpy(image).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    targets = batch

    #measure_fn = GaussianBlur(15, 2.0)
    #anti_measure_fn = lambda x_tweedie, image: image

    def measure_fn(image):
      measurements = torch.abs(torch.fft.fft2(image))
      return measurements

    def anti_measure_fn(x_tweedie, measured_diff):
      # Take phase from x_tweedie and amplitude from measured_diff
      x_tweedie_fft = torch.fft.fft2(x_tweedie)
      return torch.real(torch.fft.ifft2((measured_diff / (torch.abs(x_tweedie_fft)+0.001) * x_tweedie_fft)))

    measurements = measure_fn(targets)
    # Add noise
    z = torch.randn_like(measurements)
    #measurements += 0.1* z

    sample, n = sampling.euler_sampler_conditional(sample_dir, step, model, sde, sampling_shape, inverse_scaler, config.sampling.snr, config.sampling.n_steps_each, config.sampling.probability_flow, config.training.continuous, config.sampling.noise_removal, config.device, sampling_eps, measure_fn, anti_measure_fn, measurements, targets)
    ema.restore(model.parameters())


