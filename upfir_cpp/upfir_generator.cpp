#include "Halide.h"

using namespace Halide;

class Upfir: public Generator<Upfir> {
public:
    // Runtime
    Input<Buffer<float, 4>> input{"input"};
    Input<Buffer<float, 2>> f{"f"};
    Input<int> padx0{"padx0"};
    Input<int> padx1{"padx1"};
    Input<int> pady0{"pady0"};
    Input<int> pady1{"pady1"};
    Input<float> gain{"gain"};
    Output<Buffer<float, 4>> result{"result"};

    // Compiletime
    GeneratorParam<bool> flip_filter{"flip_filter", false};
    GeneratorParam<bool> separable{"separable", false};
    GeneratorParam<uint8_t> up_x{"up_x", 2};
    GeneratorParam<uint8_t> up_y{"up_y", 2};
    GeneratorParam<uint8_t> down_x{"down_x", 2};
    GeneratorParam<uint8_t> down_y{"down_y", 2};

    void generate() {
        Var x, y, c, n, fx, fy;

        // Pad input with zeros
        Func with_boundary = BoundaryConditions::constant_exterior(input, 0.0f);

        // Upsample by inserting zeros.
        Func upsampled_x("upsampled_x"), upsampled_y("upsampled_y");
        upsampled_x(x, y, c, n) = select((x % up_x) == 0, with_boundary(x / up_x, y, c, n), 0);
        upsampled_y(x, y, c, n) = select((y % up_y) == 0, upsampled_x(x, y / up_y, c, n), 0);

        // Padding and cropping handled by x/y offsets
        Expr fW = f.dim(0).extent();
        Expr fH = separable ? fW : f.dim(1).extent();
        Expr pad_x0 = clamp(padx0, 0, 1000);
        Expr pad_y0 = clamp(pady0, 0, 1000);
        Expr crop_x0 = clamp(padx0, -1000, 0); // negative value
        Expr crop_y0 = clamp(pady0, -1000, 0); // negative value
        
        Func padded("padded");
        padded(x, y, c, n) = upsampled_y(x - pad_x0 - crop_x0 + fW / 2, y - pad_y0 - crop_y0 + fH / 2, c, n);
        
        Func f_flipped("f_flipped"), f_scaled("f_scaled"), convx("convx"), convxy("convxy");

        if (separable) {
            RDom r(0, fW);
            f_flipped(fx) = (flip_filter) ? f(fx, 0) : f(fW - fx - 1, 0); // inverted
            f_scaled(fx) = cast(input.type(), f_flipped(fx) * pow(gain, 0.5f)); // in-place update buggy on CUDA...
            convx(x, y, c, n) = sum(f_scaled(r) * padded(x - fW / 2 + r, y, c, n));
            convxy(x, y, c, n) = sum(f_scaled(r) * convx(x, y - fW / 2 + r, c, n));
        } else {
            RDom r(0, fW, 0, fH);
            f_flipped(fx, fy) = flip_filter ? f(fx, fy) : f(fW - fx - 1, fH - fy - 1);
            f_scaled(fx, fy) = cast(input.type(), f_flipped(fx, fy) * gain);
            convxy(x, y, c, n) = sum(f_scaled(r.x, r.y) * padded(x - fW / 2 + r.x, y - fH / 2 + r.y, c, n));
        }

        // Downsample by throwing away pixels.
        result(x, y, c, n) = convxy(x*down_x, y*down_y, c, n);

        // Estimates for autoschedulers
        input.set_estimates({{0, 1024}, {0, 1024}, {0, 3}, {0, 1}});
        result.set_estimates({{0, 1024}, {0, 1024}, {0, 3}, {0, 1}});
        f.set_estimates({{0, 63}, {0, 63}});
        padx0.set_estimate(0);
        pady0.set_estimate(0);
        padx1.set_estimate(0);
        pady1.set_estimate(0);
        gain.set_estimate(1.0f);

        if (!using_autoscheduler()) {
            // Throughput
            // Inline all: 2.61096
            // Compute root all: 0.853979
            // result and boundary: 2.65148
            // tiled: 2.34781
            // +vectorize: 10.7316
            // +parallel: 41.1112
            
            const int vec = 32;

            Var xi, yi;
            result
                .compute_root()
                .tile(x, y, xi, yi, 256, 64)
                .vectorize(xi, vec)
                .parallel(y);
        }
    }
};

HALIDE_REGISTER_GENERATOR(Upfir, upfir);