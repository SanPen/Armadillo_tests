cmake_minimum_required(VERSION 3.5)

project(Armadillo_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# add includes
include_directories(lib/armadillo-9.860.1/include)
include_directories(lib/openblas/include)
include_directories(src)

# directory where the .so are
link_directories(lib/openblas/lib)

# create the target executable, AKA the name of this project
add_executable(Armadillo_tests src/main.cpp)

# link the .so files located at "link_directories" to the executable.
target_link_libraries(Armadillo_tests libopenblas.so)  # link to OpenBlas



