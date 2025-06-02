import torch
import torchvision
from create_wheel import make_editable_install; make_editable_install() # alternatively: python create_wheel.py && pip install ...
from utilities import get_default_device
from torch_utils.ops import upfirdn2d
import halide_pt_op

from pyviewer.toolbar_viewer import AutoUIViewer
from pyviewer.params import *
from pyviewer.utils import reshape_grid

dev = get_default_device()
res = 256
img_uint8 = torchvision.io.read_image('statues.jpeg').to(dev)
img = img_uint8.div(255.0).unsqueeze(0).contiguous() # float32 in [0, 1]
N, C, H, W = img.shape
assert H >= 16 and W >= 16 and C == 3, 'Invalid image dims'

@strict_dataclass
class State(ParamContainer):
    up: Param = EnumSliderParam('UP', 1, [1, 2, 4, 8])
    down: Param = EnumSliderParam('DOWN', 1, [1, 2, 4, 8])
    fw: Param = IntParam('fw', 1, 1, 15, buttons=True)
    separable: Param = BoolParam('Separable', True)
    auto_gain: Param = BoolParam('Auto gain', False)
    show_ref: Param = BoolParam('Show ref', False)
    show_hl: Param = BoolParam('Show HL', True)
    clip: Param = BoolParam('Clip', True)
    flip: Param = BoolParam('Flip filter', False)
    padx: Param = Int2Param('Pad x', (0, 0), -100, 100)
    pady: Param = Int2Param('Pad y', (0, 0), -100, 100)
    
class Viewer(AutoUIViewer):
    def setup_state(self):
        self.state = State()
        self.state_last = None
        self.cache = {}
        self.rel_diff = None # relative difference w.r.t reference

    def draw_toolbar(self):
        if self.rel_diff is not None:
            imgui.text(f'Relative diff: {self.rel_diff:.2f}')
        if imgui.button('Clear caches'):
            halide_pt_op.clear_caches()
    
    def compute(self):
        return self.process(**self.state.as_dict())
    
    def process(
        self,
        up,
        down,
        fw,
        separable,
        auto_gain,
        show_ref,
        show_hl,
        clip,
        flip,
        padx,
        pady,
    ):
        fshape = (fw,) if separable else (fw, fw)
        f = torch.ones(*fshape, dtype=torch.float32, device=dev)
        f /= f.sum()
        args = (img, f)
        gain = max(1, (up / down)) ** 2 if auto_gain else 1
        kwargs = dict(up=up, down=down, padding=(*padx, *pady), flip_filter=flip, gain=float(gain))
        imgs = []
        if show_hl:
            from upfir import _upfir_hl_buggy, get_output_shape
            imgs.append(upfirdn2d.upfirdn2d(*args, **kwargs, impl='halide'))
            #out = torch.zeros(*img.shape, device=dev, dtype=torch.float32)
            #pad_t = torch.tensor([pady[0]], device=dev, dtype=torch.int32)
            #res = _upfir_hl_buggy(img.float(), pad_t, out=out).to(img.dtype) # tensor, broken
            #res = _upfir_hl_buggy(img.float(), pady[0], out=out).to(img.dtype) # scalar, OK
            #imgs.append(res)
        if show_ref:
            imgs.append(upfirdn2d.upfirdn2d(*args, **kwargs, impl='ref'))
        if imgs:
            if len(imgs) == 2:
                self.rel_diff = (imgs[1]-imgs[0]).abs().sum() / imgs[1].sum().clip(min=1e-5)
            grid = reshape_grid(torch.cat(imgs, dim=0)) # hwc
            return grid.clip(0, 1) if clip else grid
        else:
            self.rel_diff = None
            return None

if __name__ == '__main__':
    viewer = Viewer('AutoUI example')
