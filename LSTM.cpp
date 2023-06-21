// file: LSTM.cpp

#include "LSTM.h"

#include <cmath>
#include <random>

LSTM::LSTM(int inputSize, int hiddenSize)
    : inputSize(inputSize), hiddenSize(hiddenSize) {
  // 初始化权重和偏置
  std::random_device rd;
  std::mt19937 generator(rd());
  std::normal_distribution<double> distribution(0.0, 0.1);

  inputGateWeights.resize(inputSize * hiddenSize);
  forgetGateWeights.resize(inputSize * hiddenSize);
  outputGateWeights.resize(inputSize * hiddenSize);
  cellStateWeights.resize(inputSize * hiddenSize);

  inputGateBiases.resize(hiddenSize);
  forgetGateBiases.resize(hiddenSize);
  outputGateBiases.resize(hiddenSize);
  cellStateBiases.resize(hiddenSize);

  dInputGateWeights.resize(inputSize * hiddenSize);
  dForgetGateWeights.resize(inputSize * hiddenSize);
  dOutputGateWeights.resize(inputSize * hiddenSize);
  dCellStateWeights.resize(inputSize * hiddenSize);

  dInputGateBiases.resize(hiddenSize);
  dForgetGateBiases.resize(hiddenSize);
  dOutputGateBiases.resize(hiddenSize);
  dCellStateBiases.resize(hiddenSize);

  for (int i = 0; i < inputSize * hiddenSize; i++) {
    inputGateWeights[i] = distribution(generator);
    forgetGateWeights[i] = distribution(generator);
    outputGateWeights[i] = distribution(generator);
    cellStateWeights[i] = distribution(generator);

    dInputGateWeights[i] = 0.0;
    dForgetGateWeights[i] = 0.0;
    dOutputGateWeights[i] = 0.0;
    dCellStateWeights[i] = 0.0;
  }

  for (int i = 0; i < hiddenSize; i++) {
    inputGateBiases[i] = distribution(generator);
    forgetGateBiases[i] = distribution(generator);
    outputGateBiases[i] = distribution(generator);
    cellStateBiases[i] = distribution(generator);

    dInputGateBiases[i] = 0.0;
    dForgetGateBiases[i] = 0.0;
    dOutputGateBiases[i] = 0.0;
    dCellStateBiases[i] = 0.0;
  }

  input.resize(inputSize);
  output.resize(hiddenSize);
  cellState.resize(hiddenSize);
  hiddenState.resize(hiddenSize);

  dHiddenState.resize(hiddenSize);
  dCellState.resize(hiddenSize);

  inputGate.resize(hiddenSize);
  forgetGate.resize(hiddenSize);
  outputGate.resize(hiddenSize);
  cellStateUpdate.resize(hiddenSize);
}

void LSTM::sigmoid(std::vector<double> &x) {
  for (double &val : x) {
    val = 1.0 / (1.0 + std::exp(-val));
  }
}

void LSTM::tanh(std::vector<double> &x) {
  for (double &val : x) {
    val = std::tanh(val);
  }
}

double LSTM::dotProduct(const std::vector<double> &a,
                        const std::vector<double> &b) {
  double result = 0.0;
  for (int i = 0; i < a.size(); i++) {
    result += a[i] * b[i];
  }
  return result;
}

void LSTM::forward(const std::vector<double> &input) {
  this->input = input;

  // 计算输入门
  // 在 forward 函数中
  // 计算输入门
  // 不再需要这行：std::vector<double> inputGate(hiddenSize);
  std::copy(input.begin(), input.end(), hiddenState.begin());
  for (int i = 0; i < hiddenSize; i++) {
    hiddenState[i] =
        dotProduct(input, {&inputGateWeights[i * inputSize],
                           &inputGateWeights[(i + 1) * inputSize]}) +
        inputGateBiases[i];
  }
  sigmoid(hiddenState);

  // 计算遗忘门
  std::vector<double> forgetGate(hiddenSize);
  std::copy(input.begin(), input.end(), hiddenState.begin());
  for (int i = 0; i < hiddenSize; i++) {
    hiddenState[i] =
        dotProduct(input, {&forgetGateWeights[i * inputSize],
                           &forgetGateWeights[(i + 1) * inputSize]}) +
        forgetGateBiases[i];
  }
  sigmoid(hiddenState);

  // 计算输出门
  std::vector<double> outputGate(hiddenSize);
  std::copy(input.begin(), input.end(), hiddenState.begin());
  for (int i = 0; i < hiddenSize; i++) {
    hiddenState[i] =
        dotProduct(input, {&outputGateWeights[i * inputSize],
                           &outputGateWeights[(i + 1) * inputSize]}) +
        outputGateBiases[i];
  }
  sigmoid(hiddenState);

  // 计算细胞状态更新
  std::vector<double> cellStateUpdate(hiddenSize);
  std::copy(input.begin(), input.end(), hiddenState.begin());
  for (int i = 0; i < hiddenSize; i++) {
    hiddenState[i] =
        dotProduct(input, {&cellStateWeights[i * inputSize],
                           &cellStateWeights[(i + 1) * inputSize]}) +
        cellStateBiases[i];
  }
  tanh(hiddenState);

  // 更新细胞状态
  for (int i = 0; i < hiddenSize; i++) {
    cellState[i] = inputGate[i] * hiddenState[i] + forgetGate[i] * cellState[i];
  }

  // 更新输出
  tanh(cellState);
  for (int i = 0; i < hiddenSize; i++) {
    output[i] = outputGate[i] * cellState[i];
  }
}

void LSTM::backward(const std::vector<double> &dOutput) {
  // 计算对输出门、细胞状态和输入门的梯度
  for (int i = 0; i < hiddenSize; i++) {
    dOutputGateBiases[i] = dOutput[i] * cellState[i];
    dCellState[i] += dOutput[i] * outputGate[i];
    dCellState[i] *= (1 - cellState[i] * cellState[i]);
    dInputGateBiases[i] = dCellState[i] * hiddenState[i];
    dForgetGateBiases[i] = dCellState[i] * cellState[i];
    dCellStateBiases[i] = dCellState[i] * forgetGate[i];
  }

  // 计算对输入权重的梯度
  for (int i = 0; i < hiddenSize; i++) {
    for (int j = 0; j < inputSize; j++) {
      dOutputGateWeights[i * inputSize + j] = dOutputGateBiases[i] * input[j];
      dCellStateWeights[i * inputSize + j] = dCellStateBiases[i] * input[j];
      dInputGateWeights[i * inputSize + j] = dInputGateBiases[i] * input[j];
      dForgetGateWeights[i * inputSize + j] = dForgetGateBiases[i] * input[j];
    }
  }

  // 计算对输入的梯度
  for (int i = 0; i < inputSize; i++) {
    dHiddenState[i] = 0.0;
    for (int j = 0; j < hiddenSize; j++) {
      dHiddenState[i] +=
          dOutputGateBiases[j] * outputGateWeights[j * inputSize + i];
      dHiddenState[i] +=
          dInputGateBiases[j] * inputGateWeights[j * inputSize + i];
      dHiddenState[i] +=
          dForgetGateBiases[j] * forgetGateWeights[j * inputSize + i];
      dHiddenState[i] +=
          dCellStateBiases[j] * cellStateWeights[j * inputSize + i];
    }
  }
}

void LSTM::updateWeights(double learningRate) {
  for (int i = 0; i < inputSize * hiddenSize; i++) {
    inputGateWeights[i] -= learningRate * dInputGateWeights[i];
    forgetGateWeights[i] -= learningRate * dForgetGateWeights[i];
    outputGateWeights[i] -= learningRate * dOutputGateWeights[i];
    cellStateWeights[i] -= learningRate * dCellStateWeights[i];
  }

  for (int i = 0; i < hiddenSize; i++) {
    inputGateBiases[i] -= learningRate * dInputGateBiases[i];
    forgetGateBiases[i] -= learningRate * dForgetGateBiases[i];
    outputGateBiases[i] -= learningRate * dOutputGateBiases[i];
    cellStateBiases[i] -= learningRate * dCellStateBiases[i];
  }
}

std::vector<double> LSTM::getOutput() const { return output; }
