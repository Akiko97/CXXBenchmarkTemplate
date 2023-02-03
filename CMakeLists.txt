cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -mfma")

project(
        cpp_benchmark_template
        VERSION 1.0
        LANGUAGES CXX
)

include(FetchContent)
FetchContent_Declare(
        nanobench
        GIT_REPOSITORY https://github.com/martinus/nanobench.git
        GIT_TAG v4.1.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nanobench)

add_executable(cpp_benchmark_template main.cpp)
target_link_libraries(cpp_benchmark_template PRIVATE nanobench)
