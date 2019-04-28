# CMakeLists.txt
cmake_minimum_required(VERSION 3.7)
find_package(TensorflowCC REQUIRED)
add_executable(tf_ex tensorflow.cpp)

# With tensorflow_cc
target_link_libraries(tf_ex TensorflowCC::Shared)

# For shared library setting, you may also link cuda if it is available.
# find_package(CUDA)
# if(CUDA_FOUND)
#   target_link_libraries(example ${CUDA_LIBRARIES})
# endif()