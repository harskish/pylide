cmake_minimum_required(VERSION 3.22)
project(upfir)

#enable_testing()

# Linux: export Halide_DIR=~/halide-install/lib/cmake/Halide/
# Run with: cmake . -B bin && cmake --build bin --config Release

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(upfir.generator
                     SOURCES upfir_generator.cpp
                     LINK_LIBRARIES Halide::Tools)

# HL library
add_halide_library(upfir FROM upfir.generator
                   STMT upfir_STMT
                   SCHEDULE upfir_SCHEDULE)

# Autoscheduler
add_halide_library(upfir_auto_schedule FROM upfir.generator
                   GENERATOR upfir
                   STMT upfir_auto_schedule_STMT
                   SCHEDULE upfir_auto_schedule_SCHEDULE
                   REGISTRATION upfir_REGISTRATION
                   AUTOSCHEDULER Halide::Adams2019)

# Runner
add_executable(upfir_runner runner.cpp)
target_link_libraries(upfir_runner
                      PRIVATE
                      Halide::ImageIO # For halide_image_io.h
                      Halide::Tools # For halide_benchmark.h
                      upfir
                      upfir_auto_schedule)

# Benchmark
add_executable(bench ${upfir_REGISTRATION})
target_link_libraries(bench
                      PRIVATE
                      Halide::ImageIO # For halide_image_io.h
                      Halide::Tools # For halide_benchmark.h
                      Halide::RunGenMain
                      upfir
                      upfir_auto_schedule)

