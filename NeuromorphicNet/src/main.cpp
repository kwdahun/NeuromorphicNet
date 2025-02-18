#include <iostream>
#include "IFNeuron.h"
#include "SpikeGenerator.h"
#include "NeuralNetGenerator.h"

using namespace std;

//int main() {
//    MNISTSpikeGenerator generator(0.3, 0.001, 200.0);
//
//    if (!generator.readImages(MNIST_DATA_DIR "/train-images-idx3-ubyte") ||
//        !generator.readLabels(MNIST_DATA_DIR "/train-labels-idx1-ubyte")) {
//        std::cerr << "Failed to read MNIST data\n";
//        return 1;
//    }
//
//    auto spikes = generator.generateSpikes(generator.getImages()[0]);
//
//    std::cout << "Image Label: " << static_cast<int>(generator.getLabels()[0]) << "\n\n";
//
//    std::cout << "Mean Firing Rates:\n";
//    for (size_t i = 0; i < 284; ++i) {
//        double spike_count = std::accumulate(spikes[i].begin(), spikes[i].end(), 0.0);
//        double mean_rate = (spike_count / generator.getTimeSteps()) * 1000.0;
//
//        std::cout << "Neuron " << std::setw(2) << i
//            << ": Mean firing rate = " << std::fixed << std::setprecision(2)
//            << mean_rate << " Hz\n";
//    }
//
//    std::cout << "\nSpike Trains (- = spike, space = no spike):\n";
//    for (size_t i = 0; i < 284; ++i) {
//        std::cout << "Neuron " << std::setw(2) << i << ": ";
//        for (int t = 0; t < generator.getTimeSteps(); ++t) {
//            std::cout << (spikes[i][t] ? "-" : " ");
//        }
//        std::cout << "\n";
//    }
//
//    std::cout << "\nInput Image:\n";
//    const auto& image = generator.getImages()[0];
//    for (size_t i = 0; i < 28; ++i) {
//        for (size_t j = 0; j < 28; ++j) {
//            uint8_t pixel = image[i * 28 + j];
//            char c = ' ';
//            if (pixel > 200) c = '#';
//            else if (pixel > 150) c = '+';
//            else if (pixel > 100) c = '.';
//            std::cout << c << c;
//        }
//        std::cout << "\n";
//    }
//
//    auto input_neurons = NeuralNetGenerator::generate(MNIST_DATA_DIR "/mnist_classifier.json", 30.0f, 0.95f);
//
//
//
//
//    return 0;
//}

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
    auto [input_neurons, all_neurons] = NeuralNetGenerator::generate(MNIST_DATA_DIR "/mnist_classifier.json", 0.2f, 0.95f);

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