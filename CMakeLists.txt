cmake_minimum_required(VERSION 3.0)
project(benchmark_project)

# 设置编译标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找并链接Google Benchmark库
find_package(benchmark REQUIRED)

# 添加可执行文件
add_executable(lstm LSTM.cpp benchmark.cpp)

# 链接Google Benchmark库
target_link_libraries(lstm PRIVATE benchmark::benchmark)
