#!/bin/zsh

# Python 3.10
py="/Users/erik/opt/miniconda3/envs/anyscale/bin/python"

# MacOS SDK stuff:
# xcrun --show-sdk-path --sdk macosx
# sudo xcode-select --reset
# sudo xcode-select --switch <path_to_Xcode.app>

# For PyTorch build
export MACOSX_DEPLOYMENT_TARGET=13.1
export CC=clang     # Apple clang version 13.1.6 (clang-1316.0.21.2.5)
export CXX=clang++  # Apple clang version 13.1.6 (clang-1316.0.21.2.5)
export USE_CUDA=0
export USE_ROCM=0
export USE_MPS=1
export USE_PYTORCH_METAL=1
export USE_DISTRIBUTED=1
#export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
#export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# USE_FFMPEG?
# USE_PYTORCH_METAL_EXPORT?
# USE_TENSORPIPE?

$py -m pip install typing-extensions
$py -m pip install wheel
$py -m pip install delocate
$py -m pip install pyyaml

$py setup.py bdist_wheel

echo "Remember to run delocate"
