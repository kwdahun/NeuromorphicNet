#include <iostream>
#include "IFNeuron.h"
#include "MNISTSpikeGenerator.h"
#include "NeuralNetGenerator.h"
#include "Conv2d.h"
#include "Matrix.h"

using namespace std;

int main() {
    MNISTSpikeGenerator generator(0.1, 0.001, 200.0);
    if (!generator.readImages(MNIST_DATA_DIR "/train-images-idx3-ubyte") ||
        !generator.readLabels(MNIST_DATA_DIR "/train-labels-idx1-ubyte")) {
        cerr << "Failed to read MNIST data\n";
        return 1;
    }

    size_t SAMPLE_INDEX = 180;

    auto spikes = generator.generateSpikes(generator.getImages()[SAMPLE_INDEX]);
    std::cout << "Image Label: " << static_cast<int>(generator.getLabels()[SAMPLE_INDEX]) << "\n\n";
    auto [input_neurons, all_neurons] = NeuralNetGenerator::generate(MNIST_DATA_DIR "/mnist_classifier.json", 0.6f, 0.95f);

    // Output neurons are the last 10 neurons
    vector<IFNeuron*> output_neurons(all_neurons.end() - 10, all_neurons.end());
    vector<int> spike_counts(10, 0);

    // Process spikes through network
    for (int t = 0; t < generator.getTimeSteps(); ++t) {
        // Input spikes
        for (size_t i = 0; i < input_neurons.size(); ++i) {
            if (spikes[i][t]) {
                input_neurons[i]->integrate(1.0f);
            }
        }

        // Update neurons
        for (auto neuron : all_neurons) {
            float prev_potential = neuron->getMembranePotential();
            neuron->fire();
            // Count spikes for output neurons
            if (find(output_neurons.begin(), output_neurons.end(), neuron) != output_neurons.end()) {
                if (prev_potential > neuron->getThreshold()) {
                    size_t idx = neuron->getId() - output_neurons[0]->getId();
                    spike_counts[idx]++;
                    cout << "Spike occured at output neuron " << idx << endl;
                }
            }
        }
        if (t % 50 == 0) {
            cout << "\nTimestep " << t << " Output potentials: ";
            for (auto neuron : output_neurons) {
                cout << neuron->getMembranePotential() << " ";
            }
            cout << endl;
        }
    }

    // Find winner neuron
    auto max_it = std::max_element(spike_counts.begin(), spike_counts.end());
    int predicted_digit = std::distance(spike_counts.begin(), max_it);

    cout << "Predicted digit: " << predicted_digit << endl;
    cout << "Spike counts: ";
    for (int count : spike_counts) {
        cout << count << " ";
    }
    cout << endl;

    // Cleanup
    for (auto neuron : all_neurons) {
        delete neuron;
    }

    return 0;
}