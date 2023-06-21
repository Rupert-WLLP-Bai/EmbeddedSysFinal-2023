// file: benchmark.cpp

#include <benchmark/benchmark.h>

#include "LSTM.h"

// 创建一个全局变量作为 LSTM 类的实例
const int inputSize  = 10;
const int hiddenSize = 20;
LSTM      lstm(inputSize, hiddenSize);

// 定义输入
std::vector<double> input(inputSize, 0.5);

// 定义一个基准测试函数
// 在这个函数中，我们将多次调用 lstm.forward()、lstm.backward() 和
// lstm.updateWeights() 函数的运行时间将被 Google Benchmark 库记录和报告
static void LSTM_Benchmark(benchmark::State& state) {
    // 这个循环会运行多次，直到时间足够长以获取精确的基准测试结果
    for (auto _ : state) {
        lstm.forward(input);
        lstm.backward(input);
        lstm.updateWeights(0.01);
    }
}

// 注册刚才定义的基准测试函数，BENCHMARK 是 Google Benchmark 库的一个宏
// 这行代码的意思是将 LSTM_Benchmark 函数注册为一个基准测试
BENCHMARK(LSTM_Benchmark);

// 程序的主函数，它将运行所有注册的基准测试
BENCHMARK_MAIN();
