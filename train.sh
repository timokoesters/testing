#!/bin/bash
source venv/bin/activate

#TORCH_CUDNN_V8_API_DISABLED=1 CUDA_VISIBLE_DEVICES="0" MIOPEN_LOG_LEVEL=4 python3 main.py --config configs/ve/cifar10_ncsnpp_continuous.py --mode train --workdir workdir_mnist3 > /dev/null
#TORCH_CUDNN_V8_API_DISABLED=1 CUDA_VISIBLE_DEVICES="0" MIOPEN_LOG_LEVEL=4 python3 main.py --config configs/ve/mnist_ncsnpp_continuous.py --mode train --workdir workdir_mnist5 > /dev/null
TORCH_CUDNN_V8_API_DISABLED=1 CUDA_VISIBLE_DEVICES="0" MIOPEN_LOG_LEVEL=4 python3 main.py --config configs/ve/mnist_ncsnpp_continuous.py --mode train --workdir workdir_mnist_e2e_fft > /dev/null
