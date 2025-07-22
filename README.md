## PyLide: Custom PyTorch operations using Halide

Supports CUDA and MPS (Apple Silicon) PyTorch backends.

### Usage
```python
from pylide import halide_pt_op as hlpt
from pylide.create_wheel import make_editable_install; make_editable_install()
import torch

@hlpt.hl_torch_op
def my_convolution(
    # Positional args: dynamic inputs
    input,
    filter,
    \
    # Keyword args: compiletime-constants, changes cause recompilation
    pad_mode='constant',
    target: hl.Target = None # supplied by pylide for scheduling purposes
):
    def rdom(*args) -> hl.RDom:
        return hl.RDom([hl.Range(lower, sz) for lower,sz in zip(args[0::2], args[1::2])])
    
    # Variables
    x, y, c, n = hl.vars('x y c n')
    out = hl.Func('out')

    # Reduction domain
    fW = filter.dim(0).extent()
    fH = filter.dim(1).extent()
    r = rdom(0, fW, 0, fH)
    
    # Convolve
    padded = hl.BoundaryConditions.constant_exterior(input, 0.0) # zero pad
    out[x, y, c, n] = hl.sum(filter[r.x, r.y] * padded[x - fW // 2 + r.x, y - fH // 2 + r.y, c, n])

    # Manual or automatic schedule
    if target.has_gpu_feature():
        xo, yo, xi, yi = hl.vars('xo yo xi yi')
        out.gpu_tile(x, y, xo, yo, xi, yi, 8, 8)
    else:
        # Set estimates for autoscheduler
        B = 1
        input.set_estimates([(0, input.get().width()), (0, input.get().height()), (0, input.get().channels()), (0, B)])
        out.set_estimates([(0, input.get().width()), (0, input.get().height()), (0, input.get().channels()), (0, B)])
        f.set_estimates([(0, f.get().width()), (0, f.get().height())])

        # Params: Halide\src\autoschedulers\adams2019\AutoSchedule.cpp
        out = hl.Pipeline(out)
        asp = hl.AutoschedulerParams('Adams2019', { 'parallelism': 16 })
        out.apply_autoscheduler(target, asp)
        print(res.schedule_source)
    
    return out

# Inputs are NCHW pytorch tensors
input_nchw = torch.random.randn((1, 3, 512, 512), device='cuda')
filter = torch.random.randn((3, 3), device='cuda')

# Compiled and cached on first call
# The indices are flipped (WHCN) in Halide, but storage is unchanged

# The Halide programming model requires the output shape to be explicitly provided
# => this is done through an explicit output tensor
output_nchw = torch.empty_like(input_nchw)
my_convolution(input_nchw, filter, out=output_nchw)
```