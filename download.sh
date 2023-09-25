#!/bin/bash

rsync -av --progress --exclude workdir --exclude venv puffin:testing/workdir/samples/ workdir/samples
rsync -av --progress --exclude workdir_mnist2 --exclude venv puffin:testing/workdir_mnist2/samples/ workdir_mnist2/samples
rsync -av --progress --exclude workdir_mnist3 --exclude venv puffin:testing/workdir_mnist3/samples/ workdir_mnist3/samples
rsync -av --progress --exclude workdir_mnist5 --exclude venv puffin:testing/workdir_mnist5/samples/ workdir_mnist5/samples
rsync -av --progress --exclude workdir_mnist5 --exclude venv puffin:testing/workdir_mnist28/samples/ workdir_mnist28/samples
rsync -av --progress --exclude workdir_mnist5 --exclude venv puffin:testing/workdir_mnist29/samples/ workdir_mnist29/samples
rsync -av --progress --exclude workdir --exclude venv puffin:testing/workdir_mnist3/checkpoints-meta/ workdir_mnist3/checkpoints-meta
rsync -av --progress --exclude workdir --exclude venv puffin:testing/workdir_mnist5/checkpoints-meta/ workdir_mnist5/checkpoints-meta
rsync -av --progress --exclude workdir --exclude venv puffin:testing/workdir_mnist28/checkpoints-meta/ workdir_mnist28/checkpoints-meta
