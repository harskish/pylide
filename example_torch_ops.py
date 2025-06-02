import sys, os, shutil
from pathlib import Path
import platform
from create_wheel import make_editable_install
make_editable_install()
import halide as hl

from halide_pt_op import hl_torch_op
from scipy import signal
from PIL import Image
import numpy as np
import torch
import time
from typing import Callable
from utilities import get_default_device
from python_ops import run_CPU, run_GPU

def benchmark(func: Callable = None, len_s=5):
    # Initial JIT compile
    out = func()

    sync = lambda : None
    if out.device.type == 'cuda':
        sync = lambda : torch.cuda.synchronize()
    if out.device.type == 'mps':
        sync = lambda : torch.mps.synchronize()

    n_iter = 0
    t0 = time.time()
    while time.time() - t0 < len_s:
        _ = func()
        sync()
        n_iter += 1
    delta = time.time() - t0
    ms_per_it = 1000 * delta / n_iter
    return ms_per_it

# l0, u0, ..., ln, un
def rdom(*args) -> hl.RDom:
    return hl.RDom([hl.Range(l, u) for l,u in zip(args[0::2], args[1::2])])

@hl_torch_op
def gauss_down4(input: hl.Func, target: hl.Target) -> hl.Func:
    assert input.type() == hl.Float(32), 'Expected float32'
    k = hl.Buffer(type=hl.Float(32), sizes=(5, 5), name='gauss_down4_kernel')
    k.translate([-2, -2])

    x, y, n = hl.vars('x y n')
    #r = rdom(-2, 5, -2, 5)
    r = rdom(0, 5, 0, 5)

    # Gaussian kernel
    k1d = signal.gaussian(5, std=1.4).reshape(-1, 1)
    k2d = np.outer(k1d, k1d).astype(np.float32)

    k.fill(0)
    for i in range(5):
        for j in range(5):
            k[i - 2, j - 2] = k2d[i, j]

    # output with applied kernel and stride 4
    output = hl.Func('output')
    output[x, y, n] = hl.f32(hl.sum(hl.f32(input[4*x + r.x, 4*y + r.y, n] * k[r.x, r.y])))

    # Schedule
    if target.has_gpu_feature():
        raise NotImplementedError()
    else:
        output.compute_root().parallel(y).vectorize(x, 4) # 16?

    return output

@hl_torch_op
def box_down2(input: hl.Func, target: hl.Target) -> hl.Func:
    assert input.type() == hl.Float(32), 'Wrong input type, expected float32'

    x, y, c = hl.vars('x y c')
    output = hl.Func('output')
    r = rdom(0, 2, 0, 2)

    # output with box filter and stride 2
    output[x, y, c] = hl.sum(input[2*x + r.x, 2*y + r.y, c]) / 4.0

    if target.has_gpu_feature():
        raise NotImplementedError()
    else:
        output.compute_root().parallel(y).vectorize(x, 4)

    return output

@hl_torch_op
def vadd(a, b, target: hl.Target):
    i = hl.Var('i')
    out = hl.Func('output')
    out[i] = a[i] + b[i]

    # Schedule
    if target.has_gpu_feature():
       return out.gpu_tile(i, hl.Var('j'), 4)
    else:
       return out.compute_root().parallel(i)

@hl_torch_op
def test_fun_cpu():
    return run_CPU(res=4096, schedule=True)

@hl_torch_op
def test_fun_gpu():
    return run_GPU(res=4096, schedule=True)

if __name__ == '__main__':
    # Remove cahced libs
    #shutil.rmtree(Path(__file__).parent / '.cache', ignore_errors=True)
    print('Not removing cached HL-ops')

    # Box filter
    buffer_in = torch.ones((3, 512, 512), dtype=torch.float32, device='cpu')
    buffer_out = torch.zeros((3, 256, 256), dtype=torch.float32, device='cpu')
    print('box_down2:', benchmark(lambda: box_down2(buffer_in, out=buffer_out)))
    print('box_down2 output:\n', buffer_out)

    # Gaussian filter
    # buffer_in = torch.ones((3, 512, 512), dtype=torch.float32, device='cpu')
    # buffer_out = torch.zeros((3, 128, 128), dtype=torch.float32, device='cpu')
    # print('gauss_down4:', benchmark(lambda: gauss_down4(buffer_in, out=buffer_out)))
    # print('gauss_down4 output:\n', buffer_out)

    # HL Buffer types:
    # enum struct BufferDeviceOwnership : int {
    #     Allocated,               ///> halide_device_free will be called when device ref count goes to zero
    #     WrappedNative,           ///> halide_device_detach_native will be called when device ref count goes to zero
    #     Unmanaged,               ///> No free routine will be called when device ref count goes to zero
    #     AllocatedDeviceAndHost,  ///> Call device_and_host_free when DevRefCount goes to zero.
    #     Cropped,                 ///> Call halide_device_release_crop when DevRefCount goes to zero.
    # };

    # vadd
    dev = get_default_device()
    ta = torch.arange(0, 5, dtype=torch.float32, device=dev)
    tb = torch.arange(0, 5, dtype=torch.float32, device=dev)
    tc = torch.zeros_like(ta)
    vadd(ta, tb, out=tc, debug=True)
    print(ta, tb, tc)
    # print('vadd:', benchmark(lambda : vadd(ta, tb, out=tc)))
    # #print(f'   {ta}\n + {tb}\n = {tc}')

    # # Test different size (dynamic shape support)
    # ta = ta[1:3]
    # tb = tb[1:3]
    # tc = tc[1:3]
    # vadd(ta, tb, out=tc)
    # #print(f'   {ta}\n + {tb}\n = {tc}')

    # Testfun CPU
    # res_cpu = torch.empty((3, 4096, 4096), dtype=torch.float32, device='cpu')
    # print('test_fun_cpu (4k):', benchmark(lambda : test_fun_cpu(out=res_cpu)))
    # Image.fromarray((255*res_cpu).permute(1, 2, 0).cpu().byte().numpy()).show()

    # Files:
    # test_fun_gpu.c: contains metal kernel, test_fun_gpu_argv, test_fun_gpu
    # test_fun_gpu.a: static lib with above symbols
    
    # Stuff to check:
    # Halide::PyTorch::wrap<uint8_t>(_f): problem with Metal tensors?
    #   => https://github.com/halide/Halide/blob/main/src/runtime/HalidePyTorchHelpers.h#L96
    # hl.OutputFileType.pytorch_wrapper
    
    # halide_metal_run => src/runtime/metal.cpp
    
    # _halide_buffer_get_host(_f_buffer);  # *.c
    # halide_copy_to_device(_ucon, _f_buffer, _282);
    # _halide_buffer_set_device_dirty(_f_buffer, _319);
    
    # src/CodeGen_PyTorch.cpp: if (is_cuda) ...


    # Context setup:
    # https://github.com/grimoire/TorchMPSCustomOpsDemo/blob/fdb2c0251cc6fa46563926e9f3bf7d701ea28556/csrc/pytorch/mps/custom_add_mps.mm#LL10C5-L10C5
    # https://github.com/halide/Halide/blob/4a802519d6b5201b042f39a189f180b8211e5e4b/src/runtime/metal.cpp#L731

    # Halide:
    # typedef halide_metal_device mtl_device; == id<MTLDevice>
    # typedef halide_metal_command_queue mtl_command_queue;  <= msgSend("newCommandQueue")
    # Torch:
    # typedef id<MTLCommandQueue> MTLCommandQueue_t;
    # typedef id<MTLCommandBuffer> MTLCommandBuffer_t;
    # typedef id<MTLSharedEvent> MTLSharedEvent_t;
    # typedef id<MTLDevice> MTLDevice_t;
    # TORCH_API MPSStream* getCurrentMPSStream();

    # cmake --build build && cmake --install build --prefix halide-install

    # Testfun GPU
    #res_gpu = torch.zeros((3, 8*512, 8*512), dtype=torch.float32, device=get_default_device())
    #print('test_fun_gpu (4k):', benchmark(lambda : test_fun_gpu(out=res_gpu), len_s=15))
    #ret = test_fun_gpu(out=res_gpu)
    #Image.fromarray((255*res_gpu).permute(1, 2, 0).cpu().byte().numpy()).show()

    #assert res_gpu.count_nonzero() != 0, 'Failed?'
    #print('Done')


