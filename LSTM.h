// file: LSTM.h

#ifndef LSTM_H
#define LSTM_H

#include <vector>

class LSTM {
public:
    LSTM(int inputSize, int hiddenSize);

    // Serial computation functions
    void                forward(const std::vector<double>& input);
    void                backward(const std::vector<double>& dOutput);
    void                updateWeights(double learningRate);
    std::vector<double> getOutput() const;

    // Parallel computation functions
    void forwardParallel(const std::vector<double>& input);
    void backwardParallel(const std::vector<double>& dOutput);
    void updateWeightsParallel(double learningRate);

private:
    int                 inputSize;
    int                 hiddenSize;
    std::vector<double> input;
    std::vector<double> output;
    std::vector<double> cellState;
    std::vector<double> hiddenState;

    std::vector<double> inputGateWeights;
    std::vector<double> forgetGateWeights;
    std::vector<double> outputGateWeights;
    std::vector<double> cellStateWeights;

    std::vector<double> inputGateBiases;
    std::vector<double> forgetGateBiases;
    std::vector<double> outputGateBiases;
    std::vector<double> cellStateBiases;

    std::vector<double> dInputGateWeights;
    std::vector<double> dForgetGateWeights;
    std::vector<double> dOutputGateWeights;
    std::vector<double> dCellStateWeights;

    std::vector<double> dInputGateBiases;
    std::vector<double> dForgetGateBiases;
    std::vector<double> dOutputGateBiases;
    std::vector<double> dCellStateBiases;

    std::vector<double> dHiddenState;
    std::vector<double> dCellState;

    std::vector<double> inputGate;
    std::vector<double> forgetGate;
    std::vector<double> outputGate;
    std::vector<double> cellStateUpdate;



    void   sigmoid(std::vector<double>& x);
    void   tanh(std::vector<double>& x);
    double dotProduct(const std::vector<double>& a, const std::vector<double>& b);
};

#endif
