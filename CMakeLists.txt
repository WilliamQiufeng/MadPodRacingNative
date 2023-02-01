cmake_minimum_required(VERSION 3.24)
project(MadPodRacing)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-fms-extensions)
add_library(MadPodRacing SHARED ANN.h GameSimulator.cpp GameSimulator.h Utils.cpp Utils.h Exports.cpp)
set_target_properties(MadPodRacing PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY /Users/mac/UnityProjects/MadPodRacing/Assets/Plugins/)