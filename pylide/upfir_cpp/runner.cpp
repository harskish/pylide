#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "bin/upfir.h"

#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: ./filter input.png output.png range_sigma timing_iterations\n"
               "e.g. ./filter input.png output.png 0.1 10\n");
        return 0;
    }

    float r_sigma = (float)atof(argv[3]);
    int timing_iterations = atoi(argv[4]);

    Buffer<float, 2> input = load_and_convert_image(argv[1]);
    Buffer<float, 2> output(input.width(), input.height());

    // struct halide_buffer_t *_input_buffer, struct halide_buffer_t *_f_buffer, int32_t _padx0, int32_t _padx1, int32_t _pady0, int32_t _pady1, float _gain, struct halide_buffer_t *_result_buffer
    //upfir(input, r_sigma, output);

    // Timing code. Timing doesn't include copying the input data to
    // the gpu or copying the output back.

    // Manually-tuned version
    // double min_t_manual = benchmark(timing_iterations, 10, [&]() {
    //     bilateral_grid(input, r_sigma, output);
    //     output.device_sync();
    // });
    // printf("Manually-tuned time: %gms\n", min_t_manual * 1e3);

    // convert_and_save_image(output, argv[2]);

    printf("Success!\n");
    return 0;
}
