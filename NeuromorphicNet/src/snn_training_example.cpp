#include <iostream>
#include <vector>
#include <random>
#include "IFNeuron.h"
#include "SNNTrainer.h"
#include "MNISTSpikeGenerator.h"
#include "NeuralNetGenerator.h"

using namespace std;

int main() {
    // Initialize MNIST spike generator
    MNISTSpikeGenerator generator(0.1, 0.001, 200.0);
    if (!generator.readImages(MNIST_DATA_DIR "/train-images-idx3-ubyte") ||
        !generator.readLabels(MNIST_DATA_DIR "/train-labels-idx1-ubyte")) {
        cerr << "Failed to read MNIST data\n";
        return 1;
    }

    // Generate neural network
    auto [input_neurons, all_neurons] = NeuralNetGenerator::generate(
        MNIST_DATA_DIR "/mnist_classifier.json", 0.7f, 0.5f);
    
    // Get output neurons (last 10 neurons)
    vector<IFNeuron*> output_neurons(all_neurons.end() - 10, all_neurons.end());
    
    // Create trainer with Adam optimizer
    SNNTrainer trainer(all_neurons, input_neurons, output_neurons, 
                      generator.getTimeSteps(), 0.001f, "adam");
    
    cout << "Starting SNN training with surrogate gradients and Adam optimizer..." << endl;
    cout << "Network size: " << all_neurons.size() << " neurons" << endl;
    cout << "Input neurons: " << input_neurons.size() << endl;
    cout << "Output neurons: " << output_neurons.size() << endl;
    
    // Training parameters
    int num_epochs = 10;
    int batch_size = 32;
    int num_samples = min(1000, static_cast<int>(generator.getImages().size()));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int correct_predictions = 0;
        
        // Shuffle training indices
        vector<int> indices(num_samples);
        iota(indices.begin(), indices.end(), 0);
        random_device rd;
        mt19937 g(rd());
        shuffle(indices.begin(), indices.end(), g);
        
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int batch_end = min(batch_start + batch_size, num_samples);
            float batch_loss = 0.0f;
            
            for (int i = batch_start; i < batch_end; i++) {
                int sample_idx = indices[i];
                auto spikes = generator.generateSpikes(generator.getImages()[sample_idx]);
                int target_label = static_cast<int>(generator.getLabels()[sample_idx]);
                
                // Training step
                float loss = trainer.trainStep(spikes, target_label, "crossentropy");
                batch_loss += loss;
                
                // Check prediction
                int predicted = trainer.predict(spikes);
                if (predicted == target_label) {
                    correct_predictions++;
                }
                
                // Progress output
                if (i % 100 == 0) {
                    cout << "Epoch " << epoch + 1 << ", Sample " << i + 1 
                         << "/" << num_samples << ", Loss: " << loss 
                         << ", Predicted: " << predicted << ", Target: " << target_label << endl;
                }
            }
            
            epoch_loss += batch_loss;
        }
        
        float avg_loss = epoch_loss / num_samples;
        float accuracy = static_cast<float>(correct_predictions) / num_samples * 100.0f;
        
        cout << "Epoch " << epoch + 1 << "/" << num_epochs 
             << " - Loss: " << avg_loss 
             << ", Accuracy: " << accuracy << "%" << endl;
        
        // Reduce learning rate every few epochs
        if ((epoch + 1) % 3 == 0) {
            float new_lr = trainer.learning_rate * 0.9f;
            trainer.setLearningRate(new_lr);
            cout << "Learning rate reduced to: " << new_lr << endl;
        }
    }
    
    // Test on a few samples
    cout << "\nTesting on sample predictions:" << endl;
    for (int i = 0; i < 10; i++) {
        auto spikes = generator.generateSpikes(generator.getImages()[i]);
        int target_label = static_cast<int>(generator.getLabels()[i]);
        int predicted = trainer.predict(spikes);
        auto spike_counts = trainer.getSpikeCountsLastRun();
        
        cout << "Sample " << i << " - Target: " << target_label 
             << ", Predicted: " << predicted << ", Spike counts: ";
        for (int count : spike_counts) {
            cout << count << " ";
        }
        cout << endl;
    }
    
    // Cleanup
    for (auto neuron : all_neurons) {
        delete neuron;
    }
    
    cout << "Training completed!" << endl;
    return 0;
}