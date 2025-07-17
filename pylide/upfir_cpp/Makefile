#CXX = ~/llvm15/bin/clang++
CXX = c++
HLDIR ?= ~/halide-install

# sudo apt-get install libpng-dev libjpeg-dev ffmpeg

# Genrator
bin/main: upfir_generator.cpp
	$(CXX) -g $(HLDIR)/share/tools/GenGen.cpp upfir_generator.cpp -I $(HLDIR)/include -L $(HLDIR)/lib -lHalide -Wl,-rpath $(HLDIR)/lib -o bin/main

bin/upfir.a: bin/main
	bin/main -g upfir -o bin -e static_library,registration,stmt,assembly,c_header target=x86-64-linux-avx2 \
		up_x=2 \
		up_y=2 \
		down_x=2 \
		down_y=2 \
		separable=False

bin/viz/upfir.a: bin/main
	bin/main -g upfir -o bin/viz target=host-trace_all

bin/upfir.rungen: bin/upfir.a
	$(CXX) $(HLDIR)/share/tools/RunGenMain.cpp bin/upfir.a bin/upfir.registration.cpp -I $(HLDIR)/include -o bin/upfir.rungen -lpng -ljpeg -lz -lpthread -ldl

bin/viz/upfir.rungen: bin/viz/upfir.a
	$(CXX) $(HLDIR)/share/tools/RunGenMain.cpp bin/viz/upfir.a bin/viz/upfir.registration.cpp -I $(HLDIR)/include -o bin/viz/upfir.rungen -lpng -ljpeg -lz -lpthread -ldl

bin/traceviz:
	$(CXX) ../Halide/util/HalideTraceViz.cpp -I $(HLDIR)/include -I $(HLDIR)/share/tools -o bin/traceviz

test: bin/upfir.rungen
	bin/upfir.rungen \
		input=random:0:[1024,1024,3,1] \
		f=constant:0.04:[13,13] \
		padx0=-10 \
		padx1=0 \
		pady0=-10 \
		pady1=0 \
		gain=1.0 \
		result=upfirry.png \
		--output_extents=[1024,1024,3,1] \
		--benchmarks=all --benchmark_min_time=1 --parsable_output

testbig: bin/upfir.rungen
	bin/upfir.rungen input=random:0:[4000,4000] result=upfirry.png --benchmarks=all --benchmark_min_time=1 --parsable_output | grep THROUGHPUT_MPIX_PER_SEC

# Usage example: Halide\tutorial\figures\generate_figures_8.sh
# 	=> using Halide\python_bindings\tutorial\lesson_08_scheduling_2.py
video: bin/viz/upfir.rungen bin/traceviz
	HL_TRACE_FILE=/dev/stdout bin/viz/upfir.rungen input=random:0:[64,64] result=upfirry.png | \
		bin/traceviz --size 1920 1080 --timestep 10000 | ffmpeg -f rawvideo -pix_fmt bgr32 -s 1920x1080 -i /dev/stdin -c:v h264 bin/viz/output.avi

clean:/
	rm -rf bin/*
	mkdir bin/viz