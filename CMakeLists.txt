cmake_minimum_required(VERSION 3.14)

# Make gtest visible to clangd vscode extension
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(GravNetHLS CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(googletest)

add_subdirectory(test/hls)
