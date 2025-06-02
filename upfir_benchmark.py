import torch
import numpy as np
from shutil import rmtree
from pathlib import Path

from halide_ops.create_wheel import make_editable_install
make_editable_install(llvm_ver=19)
import halide as hl # type: ignore

from halide_ops import halide_pt_op as hlpt
from utilities import get_default_device, get_dataset_dir, random_seed
from repos.anyres_gan.training.dataset import ImageFolderDataset

from torch_utils.ops import upfirdn2d
from halide_ops.torch_ops import benchmark
from halide_ops import halide_pt_op
from halide_ops import custom_ops_hl
custom_ops_hl.VERBOSITY = halide_pt_op.VERBOSITY = 'none'

print('Deleting cached pipelines and torch ops')
halide_pt_op.clear_disk_cache()

dev = get_default_device()

res = 256
dset = ImageFolderDataset(f'{get_dataset_dir()}/random', crop_image=True, resolution=res)
img_np = dset._load_raw_image(0, resize=True)[None] / 255.0
img = torch.tensor(img_np, dtype=torch.float32, device=dev).contiguous()
N, C, H, W = img.shape
assert H >= 16 and W >= 16 and C == 3, 'Invalid image dims'
#siv.draw(img_chw=img[0])

ran1d = torch.arange(0, 1024, device=dev, dtype=torch.float32)
test_img = torch.outer(ran1d, ran1d).unsqueeze(0).repeat(1, 3, 1, 1) # NCHW

print('TODO: minibatch and channel count!')
seed_offset = 0

timings_ms = np.zeros((50, 3)) # (ref, cuda, hl)
for test_idx in range(timings_ms.shape[0]):
    seed = seed_offset + test_idx
    random_seed(seed, silent=True)
    hlpt.DEBUG_IDX = seed
    scale_opts = [
        (1, 1), # (up, down)
        (2, 2),
        (4, 4),
        (1, 2), # 2x downscale
        (2, 4), # 2x downscale
        (4, 8), # 2x downscale
    ]
    UP, DOWN = scale_opts[np.random.randint(len(scale_opts))]
    pad = (
        np.random.randint(-9, 9),
        np.random.randint(-9, 9),
        np.random.randint(-9, 9),
        np.random.randint(-9, 9),
    )
    
    fW = np.random.randint(1, 15)
    separable = np.random.rand() < 0.5
    flip_filter = np.random.randn() < 0.5
    fshape = (fW,) if separable else (fW, fW)
    f = torch.randn(*fshape, dtype=torch.float32, device=dev)
    f /= f.sum()

    args = (img, f)
    kwargs = dict(up=UP, down=DOWN, padding=pad, flip_filter=flip_filter)
    tag = ''

    run_ref = lambda : upfirdn2d.upfirdn2d(*args, **kwargs, impl='ref')
    run_cuda = lambda : upfirdn2d.upfirdn2d(*args, **kwargs, impl='cuda')
    run_halide = lambda : upfirdn2d.upfirdn2d(*args, **kwargs, impl='halide')
    def comp_diff(*args, **kwargs):
        return (upfirdn2d.upfirdn2d(*args, **kwargs, impl='ref') - upfirdn2d.upfirdn2d(*args, **kwargs, impl='halide')).abs()[0]
    def comp_cat(*args, **kwargs):
        return torch.cat([upfirdn2d.upfirdn2d(*args, **kwargs, impl='ref'), upfirdn2d.upfirdn2d(*args, **kwargs, impl='halide')], dim=-1)[0]

    # import siv;
    # f = torch.ones(12, 12, dtype=torch.float32, device=dev)
    # f /= f.sum()
    # siv.draw(img_chw=comp_diff(img, f, up=UP, down=DOWN, padding=(-5, 6, -9, 8), flip_filter=flip_filter))
    # siv.draw(img_chw=comp_cat(img, f, up=UP, down=DOWN, padding=(-5, 6, -9, 8), flip_filter=flip_filter))
    
    # Check consistency/determinism
    for _ in range(100):
        out1 = run_halide()
        out2 = run_halide()
        if not torch.all(out1 == out2):
            import siv; siv.draw(img_chw=(out1[0] - out2[0]).abs())
            tag = f' <--- INCONSISTENT'

    # Check correctness
    out_ref = run_ref()
    out_hl = run_halide()
    assert out_ref.shape == out_hl.shape, 'Incorrect output shape'
    diff_rel = (out_ref - out_hl).abs().sum() / out_ref.abs().sum()
    if diff_rel > 1e-5:
        import siv; siv.draw(img_chw=torch.cat((out_ref[0], out_hl[0]), dim=2))
        tag += f' <--- ERR={diff_rel:.2e}'
        #raise RuntimeError(f'Results differ for seed {seed}')

    # Performance
    dur = 0.7
    perf_ref = benchmark(run_ref, len_s=dur)
    perf_cuda = 0 #benchmark(run_cuda, len_s=dur)
    perf_hl = benchmark(run_halide, len_s=dur)
    timings_ms[test_idx] = (perf_ref, perf_cuda, perf_hl)
    print(f'[{test_idx}, seed={seed}]: up{UP}, down{DOWN}, sep={int(separable)}, flip={int(flip_filter)} fw={fW}: ref={perf_ref:.2f}ms/it, cuda={perf_cuda:.2f}ms/it, halide={perf_hl:.2f}ms/it ({(perf_ref / perf_hl):.1f}x){tag}')

avg_speedup_ref = np.mean(timings_ms[:, 0] / timings_ms[:, 2])
avg_speedup_cuda = np.mean(timings_ms[:, 1] / timings_ms[:, 2])

print(f'Avg speedup: ref={avg_speedup_ref:.2f}x, cuda={avg_speedup_cuda:.2f}x')
print('Done')
