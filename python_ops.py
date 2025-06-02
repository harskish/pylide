# https://github.com/halide/Halide/blob/main/python_bindings/src/halide/halide_/PyDerivative.cpp
# https://github.com/halide/Halide/blob/srj/python-tutorial/README_python.md

from os import makedirs
import numpy as np
from PIL import Image
from create_wheel import make_editable_install
make_editable_install()
import halide as hl
from halide.halide_ import HalideError
import time
from tqdm import trange
import siv

# https://github1s.com/halide/Halide/blob/main/python_bindings/tutorial/lesson_12_using_the_gpu.py

def find_gpu_target():
    features_to_try = [
        hl.TargetFeature.CUDA,         # Win & Linux primary
        hl.TargetFeature.Metal,        # Mac primary
        hl.TargetFeature.D3D12Compute, # Win backup
        hl.TargetFeature.OpenCL,       # Mac & Linux backup
    ]

    for f in features_to_try:
        target = hl.get_host_target().with_feature(f)
        try:
            if hl.host_supports_target_device(target):
                print('Using GPU backend', f)
                return target
        except HalideError:
            pass

    raise RuntimeError('No supported GPU backend found')

def get_graph(res):
    # Some constants
    k = 20.0 / float(res)

    # Simple formula
    x, y, c = hl.Var('x'), hl.Var('y'), hl.Var('c')
    f = hl.Func('f')
    e = hl.sin(x * ((c + 1.0) / 3.0) * k) * hl.cos(y * ((c + 1.0) / 3.0) * k)
    e = hl.max(0.0, e)
    f[x, y, c] = hl.cast(hl.UInt(8), e * 255.0)
    
    #f.trace_stores()

    return f, x, y, c

def run_CPU(res, schedule=True):
    f, x, y, _ = get_graph(res)
    if schedule:
        f.vectorize(x, 4).parallel(y)
    return f

def run_GPU(res, schedule=True):
    f, x, y, c = get_graph(res)

    # Introduce thread and block indices
    bx, by, tx, ty = hl.Var('bx'), hl.Var('by'), hl.Var('tx'), hl.Var('ty')
    
    if schedule:
        f.reorder(c, x, y).bound(c, 0, 3).unroll(c)
        f.gpu_tile(x, y, bx, by, tx, ty, x_size=8, y_size=8) # 2D 8x8 tiles

    return f

# https://dragly.org/2023/11/01/numlide/
# https://dragly.org/2024/09/08/interactive-halide/

if __name__ == '__main__':
    res = 4096
    
    # CPU
    f = run_CPU(res)
    target_cpu = hl.get_host_target()
    f.compile_jit(target_cpu)

    dts = []
    for _ in trange(30, ascii=True):
        t0 = time.time()
        buf = f.realize([res, res, 3], target_cpu)
        dt = time.time() - t0
        dts.append(dt)
    assert buf.type() == hl.UInt(8)
    print(f'CPU - avg dt: {1000*np.mean(dts):.2f}ms')
    f.print_loop_nest()
    print()

    # GPU
    f = run_GPU(res)
    target_gpu = find_gpu_target()

    dts = []
    for _ in trange(30, ascii=True):
        t0 = time.time()
        buf = f.realize([res, res, 3], target_gpu)
        dt = time.time() - t0
        dts.append(dt)
    assert buf.type() == hl.UInt(8)
    print(f'GPU - avg dt: {1000*np.mean(dts):.2f}ms')
    f.print_loop_nest()
    print()

    # Show GPU output
    siv.draw(img_chw=np.asanyarray(buf))
    print("Success!")
