cmake_minimum_required(VERSION 3.10)
project(datagen C)

set(CMAKE_C_STANDARD 11)

add_executable(datagen main.c data/data.h)