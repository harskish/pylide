from pathlib import Path
from .create_wheel import make_editable_install; make_editable_install()
import halide as hl
from .utilities import hl_init_autosched; hl_init_autosched()
import numpy as np
from . import halide_pt_op as hlpt

# l0, sz0, ..., ln, szn
# def rdom(*args) -> hl.RDom:
#     return hl.RDom([hl.Range(lower, sz) for lower,sz in zip(args[0::2], args[1::2])])

def get_output_shape(img, f, up, down, padding):
    fH, fW = (f.shape[0], f.shape[0]) if f.ndim == 1 else f.shape
    N, C, H, W = img.shape
    upx, upy = up
    downx, downy = down
    padx0, padx1, pady0, pady1 = padding
    
    H = (H * upy + (pady0 + pady1) - (fH - 1) + (downy - 1)) // downy
    W = (W * upx + (padx0 + padx1) - (fW - 1) + (downx - 1)) // downx

    return (N, C, H, W)

def to_np(hl_arr):
    return np.transpose(np.array(hl_arr.get()), list(range(hl_arr.dimensions()))[::-1])

# PY GENERATOR SYNTAX (hl.GeneratorContext): https://github.com/halide/Halide/issues/7669

@hlpt.hl_torch_op
def _upfir_hl_op(
    input,
    f,
    pad_left, # right and bottom pad handled by output size
    pad_top,
    gain, # hl scalar
    up,   # must be const for autosched
    down, # must be const for autosched
    \
    pad_mode='constant',
    flip_filter=False,  # param is 'backwards'
    separable=False,  # compile-time const
    target: hl.Target = None
):
    def rdom(*args) -> hl.RDom:
        return hl.RDom([hl.Range(lower, sz) for lower,sz in zip(args[0::2], args[1::2])])
    
    # Validate arguments.
    assert isinstance(input, hl.InputBuffer)
    assert isinstance(f, hl.InputBuffer)
    assert input.dimensions() == 4
    assert f.dimensions() in [1, 2]
    assert f.type() == hl.Float(32)

    # Variables
    x, y, c, n, fx, fy = hl.vars('x y c n fx fy')

    # Pad input with zeros
    assert pad_mode == 'constant'
    with_boundary = hl.BoundaryConditions.constant_exterior(input, 0.0) # zero pad

    # Upsample by inserting zeros.
    upsampled_x, upsampled_y = hl.funcs('upsampled_x upsampled_y')
    upsampled_x[x, y, c, n] = hl.select((x % up[0]) == 0, with_boundary[x / up[0], y, c, n], 0)
    upsampled_y[x, y, c, n] = hl.select((y % up[1]) == 0, upsampled_x[x, y / up[1], c, n], 0)

    # Padding and cropping handled by x/y offsets
    padded = hl.Func('padded')
    fW = f.dim(0).extent()
    fH = f.dim(1).extent() if f.dimensions() > 1 else fW
    
    # padding: padx0, padx1, pady0, pady1
    padx0 = hl.clamp(pad_left, 0, 1_000)
    pady0 = hl.clamp(pad_top, 0, 1_000)
    cropx0 = hl.clamp(pad_left, -1_000, 0) # negative value
    cropy0 = hl.clamp(pad_top, -1_000, 0) # negative value
    
    # End padding handled by output size?
    padded[x, y, c, n] = upsampled_y[x - padx0 - cropx0 + fW // 2, y - pady0 - cropy0 + fH // 2, c, n]
    f_flipped, f_scaled, convx, convxy = hl.funcs('f_flipped f_scaled convx convxy')

    if separable:
        r = rdom(0, fW)
        f_flipped[fx] = f[fW - fx - 1] if not flip_filter else f[fx]
        f_scaled[fx] = hl.cast(input.type(), f_flipped[fx] * hl.pow(gain, 0.5)) # in-place update buggy on CUDA...
        convx[x, y, c, n] = hl.sum(f_scaled[r] * padded[x - fW // 2 + r, y, c, n])
        convxy[x, y, c, n] = hl.sum(f_scaled[r] * convx[x, y - fW // 2 + r, c, n])
    else:
        r = rdom(0, fW, 0, fH)
        f_flipped[fx, fy] = f[fW - fx - 1, fH - fy - 1] if not flip_filter else f[fx, fy]
        f_scaled[fx, fy] = hl.cast(input.type(), f_flipped[fx, fy] * gain)
        convxy[x, y, c, n] = hl.sum(f_scaled[r.x, r.y] * padded[x - fW // 2 + r.x, y - fH // 2 + r.y, c, n])

    # Downsample by throwing away pixels.
    out = hl.Func('out')

    # With static up/down: no clamp necessary for autosched
    out[x, y, c, n] = convxy[x*down[0], y*down[1], c, n]

    if target.has_gpu_feature():
        xo, yo, xi, yi = hl.vars('xo yo xi yi')
        out.gpu_tile(x, y, xo, yo, xi, yi, 8, 8)

        # Set estimates for autosheduler
        # B = 1
        # sx = up[0] / down[0]
        # sy = up[1] / down[1]
        # input.set_estimates([(0, input.get().width()), (0, input.get().height()), (0, input.get().channels()), (0, B)])
        # out.set_estimates([(0, int(sx*input.get().width())), (0, int(sy*input.get().height())), (0, input.get().channels()), (0, B)])
        # f.set_estimates([(0, f.get().width())] if separable else [(0, f.get().width()), (0, f.get().height())])
        
        # # Params: Halide\src\autoschedulers\anderson2021\AutoSchedule.cpp
        # out = hl.Pipeline(out)
        # asp = hl.AutoschedulerParams('Anderson2021', {
        #     'parallelism': 16,
        #     #'beam_size': '',
        #     #'random_dropout': '',
        #     #'random_dropout_seed': '',
        #     #'weights_path': '',
        #     #'disable_subtiling': '',
        #     #'randomize_tilings': '',
        #     #'search_space_options': '',
        #     #'freeze_inline_compute_root': '',
        #     #'partial_schedule_path': '',
        #     #'num_passes': 20,
        #     #'stack_factor': '',
        #     #'shared_memory_limit_kb': '',
        #     #'shared_memory_sm_limit_kb': '',
        #     #'active_block_limit': '',
        #     #'active_warp_limit': '',
        # })
        # out.apply_autoscheduler(target, asp) #print(res.schedule_source)
    else:
        # Set estimates for autoscheduler
        B = 1
        sx = up[0] / down[0]
        sy = up[1] / down[1]
        input.set_estimates([(0, input.get().width()), (0, input.get().height()), (0, input.get().channels()), (0, B)])
        out.set_estimates([(0, int(sx*input.get().width())), (0, int(sy*input.get().height())), (0, input.get().channels()), (0, B)])
        f.set_estimates([(0, f.get().width())] if separable else [(0, f.get().width()), (0, f.get().height())])

        # Params: Halide\src\autoschedulers\adams2019\AutoSchedule.cpp
        out = hl.Pipeline(out)
        asp = hl.AutoschedulerParams('Adams2019', {
            'parallelism': 16,
            #'beam_size': ''
            #'random_dropout': ''
            #'random_dropout_seed': ''
            #'weights_path': ''
            #'disable_subtiling': ''
            #'disable_memoized_features': ''
            #'disable_memoized_blocks': ''
            #'memory_limit': ''
        })
        out.apply_autoscheduler(target, asp) #print(res.schedule_source)

    return out

@hlpt.hl_torch_op
def _upfir_hl_buggy(
    input,
    pad,
    \
    target: hl.Target = None
):  
    pad_top = pad if isinstance(pad, hl.Param) else pad[0] # handle scalar and buffer
    pady0 = hl.max(pad_top, 0)
    cropy0 = hl.min(pad_top, 0) # negative value

    padded = hl.Func('padded')
    x, y, c, n = hl.vars('x y c n')
    with_boundary = hl.BoundaryConditions.constant_exterior(input, 0) # zero pad
    padded[x, y, c, n] = with_boundary[x, y - pady0 - cropy0, c, n]
    out = padded

    if target.has_gpu_feature():
        xo, yo, xi, yi = hl.vars('xo yo xi yi')
        out.gpu_tile(x, y, xo, yo, xi, yi, 16, 16)

    return out

if __name__ == '__main__':
    from PIL import Image
    from halide_pt_op import _make_hl_imageparam, _make_hl_buffer, _make_hl_scalar
    from python_ops import find_gpu_target

    W, H, C = (256, 128, 4)
    dtype = 'float32'
    getshape = lambda b: f'w={b.width()} h={b.height()} c={b.channels()}'
    
    def get_img():
        in_hwc = np.array(Image.open('/Users/erik/datasets/random/image 11.png').resize((W, H)))
        in_chw = np.transpose(in_hwc, (2, 0, 1))
        in_nchw = in_chw[None]
        if C == 4:
            in_nchw = np.concat([in_nchw, 255 * np.ones_like(in_nchw[:, 0:1, :, :])], axis=1)
        if dtype == 'float32':
            in_nchw = in_nchw.astype(np.float32) / 255
        return in_nchw
    
    img = _make_hl_imageparam(get_img())
    target = find_gpu_target()
    #target = hl.get_host_target()
    #padding = _make_hl_scalar(0)
    #padding = _make_hl_imageparam(np.array([0], dtype=np.int32))
    padding = _make_hl_buffer(np.array([0], dtype=np.int32))
    f = _make_hl_imageparam(np.array([1, 3, 3, 1], dtype=np.float32) / 8)
    
    func = _upfir_hl_buggy.original(img, padding, target=target)
    #func = _upfir_hl_op.original(
    #    img,
    #    f,
    #    0, # pad left
    #    padding,
    #    1, # gain
    #    [1, 1], # up
    #    [1, 1], # down
    #    separable=True,
    #    target=target,
    #)

    for crop in [-10, -5, 0, 5, 10]:
        #padding.set(crop)
        padding.copy_from(hl.Buffer(np.array([crop], dtype=np.int32)))
        #np.asanyarray(padding.get())[0] = crop # corrupts
        ret = func.realize((W, H, C, 1), target)
        import siv; siv.draw(img_chw=np.array(ret)[0]) # TODO: non-copy draw with asanyarray mutates data!
        print(crop)

    # JIT, hl.scalar: works
    # JIT, imageparam + copy_from: works
    # JIT, imageparam + asanyarray modify: broken

    # hl.buffer: assumed constant?
    # hl.imageparam: assumed runtime varying
