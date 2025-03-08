//#include <iostream>
//#include "IFNeuron.h"
//#include "SpikeGenerator.h"
//#include "NeuralNetGenerator.h"
//
//using namespace std;
//
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