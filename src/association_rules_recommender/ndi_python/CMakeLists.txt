cmake_minimum_required(VERSION 3.17)
project(ndi)

set(CMAKE_CXX_STANDARD 20)

file(GLOB Lib "bf/*.h" "bf/*.cpp")

add_executable(ndi ${Lib})