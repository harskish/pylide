import numpy as np
import halide as hl
import siv
import time

# l0, sz0, ..., ln, szn
def rdom(*args) -> hl.RDom:
    return hl.RDom([hl.Range(lower, sz) for lower,sz in zip(args[0::2], args[1::2])])

@hl.alias(upfir_Adams2019={"autoscheduler": "Adams2019"})
@hl.generator(name="upfir")
class UpfirGenerator(hl.Generator):
    # Runtime
    input = hl.InputBuffer(hl.Float(32), 4)
    f = hl.InputBuffer(hl.Float(32), 2)
    padx0 = hl.InputScalar(hl.Int(32))
    padx1 = hl.InputScalar(hl.Int(32))
    pady0 = hl.InputScalar(hl.Int(32))
    pady1 = hl.InputScalar(hl.Int(32))
    gain = hl.InputScalar(hl.Float(32))
    result = hl.OutputBuffer(hl.Float(32), 4)

    # Compiletime
    flip_filter = hl.GeneratorParam(False)
    separable = hl.GeneratorParam(False)
    up_x = hl.GeneratorParam(2)
    up_y = hl.GeneratorParam(2)
    down_x = hl.GeneratorParam(2)
    down_y = hl.GeneratorParam(2)
    
    def generate(self):
        # Variables
        x, y, c, n, fx, fy = hl.vars('x y c n fx fy')

        # Pad input with zeros
        with_boundary = hl.BoundaryConditions.constant_exterior(self.input, 0.0) # zero pad

        # Upsample by inserting zeros.
        upsampled_x, upsampled_y = hl.funcs('upsampled_x upsampled_y')
        upsampled_x[x, y, c, n] = hl.select((x % self.up_x) == 0, with_boundary[x / self.up_x, y, c, n], 0)
        upsampled_y[x, y, c, n] = hl.select((y % self.up_y) == 0, upsampled_x[x, y / self.up_y, c, n], 0)

        # Padding and cropping handled by x/y offsets
        padded = hl.Func('padded')
        fW = self.f.dim(0).extent()
        fH = self.f.dim(1).extent() if self.f.dimensions() > 1 else fW
        
        padx0 = hl.clamp(self.padx0, 0, 1_000)
        pady0 = hl.clamp(self.pady0, 0, 1_000)
        cropx0 = hl.clamp(self.padx0, -1_000, 0) # negative value
        cropy0 = hl.clamp(self.pady0, -1_000, 0) # negative value
        
        padded[x, y, c, n] = upsampled_y[x - padx0 - cropx0 + fW // 2, y - pady0 - cropy0 + fH // 2, c, n]
        f_flipped, f_scaled, convx, convxy = hl.funcs('f_flipped f_scaled convx convxy')

        if self.separable:
            r = rdom(0, fW)
            f_flipped[fx] = self.f[fW - fx - 1] if not self.flip_filter else self.f[fx]
            f_scaled[fx] = hl.cast(self.input.type(), f_flipped[fx] * hl.pow(self.gain, 0.5)) # in-place update buggy on CUDA...
            convx[x, y, c, n] = hl.sum(f_scaled[r] * padded[x - fW // 2 + r, y, c, n])
            convxy[x, y, c, n] = hl.sum(f_scaled[r] * convx[x, y - fW // 2 + r, c, n])
        else:
            r = rdom(0, fW, 0, fH)
            f_flipped[fx, fy] = self.f[fW - fx - 1, fH - fy - 1] if not self.flip_filter else self.f[fx, fy]
            f_scaled[fx, fy] = hl.cast(self.input.type(), f_flipped[fx, fy] * self.gain)
            convxy[x, y, c, n] = hl.sum(f_scaled[r.x, r.y] * padded[x - fW // 2 + r.x, y - fH // 2 + r.y, c, n])

        # Downsample by throwing away pixels.
        # With static up/down: no clamp necessary for autosched
        self.result[x, y, c, n] = convxy[x*self.down_x, y*self.down_y, c, n]

        # if not self.using_autoscheduler():
        #     vec = 32

        #     xi, yi = hl.vars('xi yi')
        #     self.result \
        #         .compute_root() \
        #         .tile(x, y, xi, yi, 256, 64) \
        #         .vectorize(xi, vec) \
        #         .parallel(y)

if __name__ == "__main__":
    t = hl.Target("host")
    with hl.GeneratorContext(t):
        gen = UpfirGenerator()
        test_filter = gen.compile_to_callable()
        out = np.empty((1024, 1024, 3, 1), dtype=np.float32)
        timings = []
        for _ in range(50):
            t0 = time.time()
            test_filter(
                input=np.random.randn(1024, 1024, 3, 1).astype(np.float32),
                f=np.random.randn(13, 13).astype(np.float32),
                padx0=0,
                padx1=0,
                pady0=0,
                pady1=0,
                gain=1.0,
                result=out,
            )
            timings.append(time.time() - t0)
        min_time_ms = 1000 * min(timings)
        print('Min time ms:', min_time_ms)
        #siv.draw(img_hwc=out[:, :, :])
        print('Done')
