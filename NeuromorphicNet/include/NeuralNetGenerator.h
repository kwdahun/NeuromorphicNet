#pragma once
#include "IFNeuron.h"
#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

class NeuralNetGenerator {
public:
    static std::pair<std::vector<IFNeuron*>, std::vector<IFNeuron*>>
        generate(const std::string& json_path, float threshold = 1.0f, float leakage = 0.95f) {

        std::ifstream file(json_path);
        json weights;
        file >> weights;

        const size_t INPUT_SIZE = 784;  // 28 * 28
        const size_t HIDDEN1_SIZE = 512;
        const size_t HIDDEN2_SIZE = 256;
        const size_t HIDDEN3_SIZE = 128;
        const size_t OUTPUT_SIZE = 10;

        std::vector<IFNeuron*> input_layer;
        std::vector<IFNeuron*> hidden1_layer;
        std::vector<IFNeuron*> hidden2_layer;
        std::vector<IFNeuron*> hidden3_layer;
        std::vector<IFNeuron*> output_layer;
        std::vector<IFNeuron*> all_neurons;

        size_t neuron_count = 0;

        for (size_t i = 0; i < INPUT_SIZE; i++) {
            input_layer.push_back(new IFNeuron(neuron_count, threshold, leakage));
            neuron_count++;
        }
        for (size_t i = 0; i < HIDDEN1_SIZE; i++) {
            hidden1_layer.push_back(new IFNeuron(neuron_count, threshold, leakage));
            neuron_count++;
        }
        for (size_t i = 0; i < HIDDEN2_SIZE; i++) {
            hidden2_layer.push_back(new IFNeuron(neuron_count, threshold, leakage));
            neuron_count++;
        }
        for (size_t i = 0; i < HIDDEN3_SIZE; i++) {
            hidden3_layer.push_back(new IFNeuron(neuron_count, threshold, leakage));
            neuron_count++;
        }
        for (size_t i = 0; i < OUTPUT_SIZE; i++) {
            output_layer.push_back(new IFNeuron(neuron_count, threshold, leakage));
            neuron_count++;
        }

        all_neurons.insert(all_neurons.end(), input_layer.begin(), input_layer.end());
        all_neurons.insert(all_neurons.end(), hidden1_layer.begin(), hidden1_layer.end());
        all_neurons.insert(all_neurons.end(), hidden2_layer.begin(), hidden2_layer.end());
        all_neurons.insert(all_neurons.end(), hidden3_layer.begin(), hidden3_layer.end());
        all_neurons.insert(all_neurons.end(), output_layer.begin(), output_layer.end());

        // Connect input to hidden1
        auto& w1 = weights["layers.0.weight"];
        for (size_t out = 0; out < HIDDEN1_SIZE; out++)
            for (size_t in = 0; in < INPUT_SIZE; in++)
                input_layer[in]->connectTo(hidden1_layer[out], w1[out][in].get<float>());

        // Connect hidden1 to hidden2
        auto& w2 = weights["layers.2.weight"];
        for (size_t out = 0; out < HIDDEN2_SIZE; out++)
            for (size_t in = 0; in < HIDDEN1_SIZE; in++)
                hidden1_layer[in]->connectTo(hidden2_layer[out], w2[out][in].get<float>());

        // Connect hidden2 to hidden3
        auto& w3 = weights["layers.4.weight"];
        for (size_t out = 0; out < HIDDEN3_SIZE; out++)
            for (size_t in = 0; in < HIDDEN2_SIZE; in++)
                hidden2_layer[in]->connectTo(hidden3_layer[out], w3[out][in].get<float>());

        // Connect hidden3 to output
        auto& w4 = weights["layers.6.weight"];
        for (size_t out = 0; out < OUTPUT_SIZE; out++)
            for (size_t in = 0; in < HIDDEN3_SIZE; in++)
                hidden3_layer[in]->connectTo(output_layer[out], w4[out][in].get<float>());

        return { input_layer, all_neurons };
    }
};