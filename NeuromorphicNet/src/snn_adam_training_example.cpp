#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include "IFNeuron.h"
#include "SNNTrainer.h"
#include "MNISTSpikeGenerator.h"
#include "NeuralNetGenerator.h"

using namespace std;

// Function to load best model for inference using exact path
bool loadBestModel(SNNTrainer& trainer) {
    string best_model_path = MODEL_CHECKPOINT_PATH;
    int epoch;
    float loss, accuracy;

    if (trainer.loadCheckpoint(best_model_path, epoch, loss, accuracy)) {
        cout << "Loaded best model from: " << best_model_path << endl;
        cout << "Best model stats - Epoch: " << epoch << ", Loss: " << loss
            << ", Accuracy: " << accuracy << "%" << endl;
        return true;
    }
    else {
        cout << "Failed to load best model from: " << best_model_path << endl;
        return false;
    }
}

// Function to train with a specific optimizer and checkpoint support
void trainWithOptimizer(SNNTrainer& trainer, MNISTSpikeGenerator& generator,
    const string& optimizer_name, int num_epochs = 5, int num_samples = 500,
    const string& checkpoint_dir = CHECKPOINT_DIR "/MNIST", bool resume = false) {

    cout << "\n=== Training with " << optimizer_name << " optimizer ===" << endl;

    // Initialize variables for resume functionality
    int start_epoch = 0;
    float best_accuracy = 0.0f;
    float prev_loss = 0.0f;

    // Try to resume from checkpoint if requested
    if (resume) {
        cout << "Searching for checkpoints in: " << checkpoint_dir << endl;
        
        // Search for the highest available checkpoint epoch (efficient file existence check)
        int highest_found_epoch = -1;
        for (int e = 0; e < 100; e++) {  // Search up to epoch 100
            string checkpoint_path = trainer.generateCheckpointPath(checkpoint_dir, e);
            
            // Just check if file exists, don't load it
            ifstream check_file(checkpoint_path);
            if (check_file.good()) {
                highest_found_epoch = e;
                cout << "Found checkpoint file at epoch " << e << ": " << checkpoint_path << endl;
                check_file.close();
            }
        }
        
        // Load the highest epoch found
        if (highest_found_epoch >= 0) {
            string checkpoint_path = trainer.generateCheckpointPath(checkpoint_dir, highest_found_epoch);
            int loaded_epoch;
            float loaded_loss, loaded_accuracy;
            
            if (trainer.loadCheckpoint(checkpoint_path, loaded_epoch, loaded_loss, loaded_accuracy)) {
                start_epoch = loaded_epoch + 1;
                best_accuracy = loaded_accuracy;
                prev_loss = loaded_loss;
                cout << "Resumed from checkpoint: epoch " << loaded_epoch
                    << ", loss: " << loaded_loss << ", accuracy: " << loaded_accuracy << "%" << endl;
            }
        }

        if (start_epoch == 0) {
            cout << "No valid checkpoint found, starting from scratch." << endl;
            cout << "Note: If checkpoints exist but are corrupted/incompatible, they will be ignored." << endl;
        }
    }

    auto start_time = chrono::high_resolution_clock::now();

    for (int epoch = start_epoch; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int correct_predictions = 0;

        // Shuffle training indices
        vector<int> indices(num_samples);
        iota(indices.begin(), indices.end(), 0);
        random_device rd;
        mt19937 g(rd());
        shuffle(indices.begin(), indices.end(), g);

        for (int i = 0; i < num_samples; i++) {
            int sample_idx = indices[i];
            auto spikes = generator.generateSpikes(generator.getImages()[sample_idx]);
            int target_label = static_cast<int>(generator.getLabels()[sample_idx]);

            // Training step
            float loss = trainer.trainStep(spikes, target_label, "crossentropy");
            epoch_loss += loss;

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

        float avg_loss = epoch_loss / num_samples;
        float accuracy = static_cast<float>(correct_predictions) / num_samples * 100.0f;

        cout << "Epoch " << epoch + 1 << "/" << num_epochs
            << " - Loss: " << avg_loss
            << ", Accuracy: " << accuracy << "%" << endl;

        // Save checkpoint every epoch
        string checkpoint_path = trainer.generateCheckpointPath(checkpoint_dir, epoch);
        if (trainer.saveCheckpoint(checkpoint_path, epoch, avg_loss, accuracy)) {
            cout << "Checkpoint saved: " << checkpoint_path << endl;
        }
        else {
            cout << "Failed to save checkpoint: " << checkpoint_path << endl;
        }

        // Save best model if accuracy improved
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            string best_model_path = trainer.generateBestModelPath(checkpoint_dir);
            if (trainer.saveCheckpoint(best_model_path, epoch, avg_loss, accuracy)) {
                cout << "Best model saved: " << best_model_path << " (accuracy: " << accuracy << "%)" << endl;
            }
        }

        // Adaptive learning rate for Adam
        if (optimizer_name == "Adam" && (epoch + 1) % 2 == 0) {
            float new_lr = trainer.learning_rate * 0.95f;
            trainer.setLearningRate(new_lr);
            cout << "Learning rate adjusted to: " << new_lr << endl;
        }
        // More aggressive decay for SGD
        else if (optimizer_name == "SGD" && (epoch + 1) % 2 == 0) {
            float new_lr = trainer.learning_rate * 0.8f;
            trainer.setLearningRate(new_lr);
            cout << "Learning rate adjusted to: " << new_lr << endl;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << optimizer_name << " training completed in " << duration.count() << " ms" << endl;
}

// Function to test the model
void testModel(SNNTrainer& trainer, MNISTSpikeGenerator& generator,
    const string& optimizer_name, int num_test_samples = 100) {

    cout << "\n=== Testing " << optimizer_name << " trained model ===" << endl;

    int correct = 0;
    for (int i = 0; i < num_test_samples; i++) {
        auto spikes = generator.generateSpikes(generator.getImages()[i]);
        int target_label = static_cast<int>(generator.getLabels()[i]);
        int predicted = trainer.predict(spikes);

        if (predicted == target_label) {
            correct++;
        }

        if (i < 10) {  // Show first 10 predictions
            auto spike_counts = trainer.getSpikeCountsLastRun();
            cout << "Sample " << i << " - Target: " << target_label
                << ", Predicted: " << predicted << ", Spike counts: ";
            for (int count : spike_counts) {
                cout << count << " ";
            }
            cout << endl;
        }
    }

    float test_accuracy = static_cast<float>(correct) / num_test_samples * 100.0f;
    cout << "Test Accuracy (" << optimizer_name << "): " << test_accuracy << "%" << endl;
}

int main() {
    // Initialize MNIST spike generator
    MNISTSpikeGenerator generator(0.1, 0.001, 200.0);
    if (!generator.readImages(MNIST_DATA_DIR "/train-images-idx3-ubyte") ||
        !generator.readLabels(MNIST_DATA_DIR "/train-labels-idx1-ubyte")) {
        cerr << "Failed to read MNIST data\n";
        return 1;
    }

    cout << "=== SNN Training Comparison: SGD vs Adam ===" << endl;
    cout << "Dataset size: " << generator.getImages().size() << " samples" << endl;
    cout << "Time steps: " << generator.getTimeSteps() << endl;

    // Training parameters
    int num_epochs = 15;
    int num_samples = 500;
    int num_test_samples = 100;

    // ========== SGD Training ==========
    //{
    //    cout << "\n" << string(50, '=') << endl;
    //    cout << "TRAINING WITH SGD OPTIMIZER" << endl;
    //    cout << string(50, '=') << endl;
    //    
    //    // Generate neural network for SGD
    //    auto [input_neurons_sgd, all_neurons_sgd] = NeuralNetGenerator::generate(
    //        MNIST_DATA_DIR "/mnist_classifier.json", 0.7f, 0.5f);
    //    
    //    vector<IFNeuron*> output_neurons_sgd(all_neurons_sgd.end() - 10, all_neurons_sgd.end());
    //    
    //    // Create SGD trainer
    //    SNNTrainer sgd_trainer(all_neurons_sgd, input_neurons_sgd, output_neurons_sgd, 
    //                          generator.getTimeSteps(), 0.01f, "sgd");  // Higher LR for SGD
    //    
    //    // Train with SGD
    //    trainWithOptimizer(sgd_trainer, generator, "SGD", num_epochs, num_samples);
    //    
    //    // Test SGD model
    //    testModel(sgd_trainer, generator, "SGD", num_test_samples);
    //    
    //    // Cleanup
    //    for (auto neuron : all_neurons_sgd) {
    //        delete neuron;
    //    }
    //}

    // ========== Adam Training ==========
    {
        cout << "\n" << string(50, '=') << endl;
        cout << "TRAINING WITH ADAM OPTIMIZER" << endl;
        cout << string(50, '=') << endl;

        // Set random seed for consistent network generation across runs
        srand(42);  // Fixed seed for reproducible network structure
        mt19937 gen(42);  // Also set C++ random generator
        
        // Generate neural network for Adam
        auto [input_neurons_adam, all_neurons_adam] = NeuralNetGenerator::generate(
            MNIST_DATA_DIR "/mnist_classifier.json", 0.7f, 0.5f);

        vector<IFNeuron*> output_neurons_adam(all_neurons_adam.end() - 10, all_neurons_adam.end());

        // Create Adam trainer
        SNNTrainer adam_trainer(all_neurons_adam, input_neurons_adam, output_neurons_adam,
            generator.getTimeSteps(), 0.001f, "adam");  // Lower LR for Adam

        // Configure Adam parameters (optional - using defaults)
        adam_trainer.setAdamParameters(0.9f, 0.999f, 1e-8f);

        // Train with Adam (with checkpoint support)
        // Set resume=true to resume from existing checkpoint, false to start fresh
        bool resume_training = true;  // Change to true to resume from checkpoint
        trainWithOptimizer(adam_trainer, generator, "Adam", num_epochs, num_samples,
            CHECKPOINT_DIR "/MNIST", resume_training);

        // Test Adam model (with best model if available)
        cout << "\n=== Testing with best saved model ===" << endl;
        if (loadBestModel(adam_trainer)) {
            testModel(adam_trainer, generator, "Adam (Best Model)", num_test_samples);
        }
        else {
            cout << "Using current model for testing..." << endl;
            testModel(adam_trainer, generator, "Adam", num_test_samples);
        }

        // Cleanup
        for (auto neuron : all_neurons_adam) {
            delete neuron;
        }
    }

    cout << "\n=== Training Comparison Complete ===" << endl;
    cout << "Adam optimizer typically converges faster and with more stable gradients." << endl;
    cout << "SGD may require more careful learning rate tuning but can be more robust." << endl;

    return 0;
}