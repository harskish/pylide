from create_wheel import make_editable_install
make_editable_install()
import halide as hl
import torch
import numpy as np
from shutil import rmtree
from pathlib import Path

import halide_pt_op as hlpt
from utilities import get_default_device, get_dataset_dir, random_seed
#from repos.anyres_gan.training.dataset import ImageFolderDataset
#import siv

from torch_utils.ops import upfirdn2d
from torch_ops import benchmark
import time
from tqdm import trange

from halide_pt_op import BUILD_TOP_DIR

@hlpt.hl_torch_op
def noop(input, target: hl.Target):
    x, y, c, n = hl.vars('x y c n')
    out = hl.Func('out')
    out[x, y, c, n] = input[x, y, c, n]
    
    if target.has_gpu_feature():
       xo, yo, xi, yi = hl.vars('xo yo xi yi')
       out.gpu_tile(x, y, xo, yo, xi, yi, 16, 16)
    
    return out

@hlpt.hl_torch_op
def pure(target: hl.Target):
    x, y, c, n = hl.vars('x y c n')
    out = hl.Func('out')
    out[x, y, c, n] = hl.f32(x*y)
    
    if target.has_gpu_feature():
       xo, yo, xi, yi = hl.vars('xo yo xi yi')
       out.gpu_tile(x, y, xo, yo, xi, yi, 16, 8)
    
    return out

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    if device.type == 'mps':
        torch.mps.synchronize()

# Simulate costly computation
def produce_input(img, N=128):
    A = torch.randn(N, N, device=img.device)
    B = A @ A.t()  # guaranteed invertible (positive semi-definite)
    return img + torch.randn_like(img) * B.inverse().mean()

if __name__ == '__main__':
    print('Deleting cached pt ops')
    rmtree(f'{BUILD_TOP_DIR}/torch_ops', ignore_errors=True)
    print('Deleting cached pipelines')
    rmtree(f'{BUILD_TOP_DIR}/halide_pipelines', ignore_errors=True)

    random_seed()
    dev = torch.device(get_default_device())

    dset = ImageFolderDataset(f'{get_dataset_dir()}/random')
    img_np = dset._load_raw_image(0, resize=False)[None] / 255.0
    img = torch.tensor(img_np, dtype=torch.float32, device=dev).contiguous()
    N, C, H, W = img.shape

    # Output sync:
    # - Static pre-allocted input
    # - Allocate output buffer on demand, check for sync
    print('Testing output sync')
    for _ in trange(1000):
        for allocator in [torch.ones_like, torch.zeros_like, torch.empty_like]:
            input = produce_input(img)
            sync(dev)

            # No sync after allocation
            out = allocator(input)
            output = noop(input, out=out, debug=False)
            assert torch.all(output == input), 'Output not properly synced'

    # Input sync:
    # - Pre-allocate output buffer
    # - Produce input on demand, check if HL-op sees correct data
    print('Testing input sync')
    for _ in trange(1000):
        for allocator in [torch.ones_like, torch.zeros_like, torch.empty_like]:
            out = allocator(img)
            sync(dev)
            
            # No sync after input creation
            input = produce_input(img)
            output = noop(input, out=out, debug=False)
            assert torch.all(output == input), 'Input not properly synced'

    # No explicit syncs at all
    print('Testing no sync')
    for _ in trange(1000):
        for allocator in [torch.ones_like, torch.zeros_like, torch.empty_like]:
            input = produce_input(img)
            out = allocator(img)
            output = noop(input, out=out, debug=False)
            assert torch.all(output == input), 'Tensors not properly synced'

    print('Done')
