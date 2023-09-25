#!/bin/bash

rsync -av --progress --exclude workdir --exclude workdir_mnist2/samples --exclude workdir_mnist3 --exclude workdir_mnist4 --exclude workdir_mnist5/checkpoints --exclude workdir_mnist5/samples --exclude venv . puffin:testing/
