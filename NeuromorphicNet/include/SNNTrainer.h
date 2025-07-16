#pragma once
#include "IFNeuron.h"
#include "LossFunction.h"
#include "AdamOptimizer.h"
#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <numeric>
#include <string>
#include <fstream>
#include <filesystem>

using namespace std;

class SNNTrainer {
private:
    vector<IFNeuron*> all_neurons;
    vector<IFNeuron*> input_neurons;
    vector<IFNeuron*> output_neurons;
    vector<vector<float>> membrane_traces;
    vector<vector<bool>> spike_traces;
    int time_steps;
    string optimizer_type;
    AdamOptimizer adam_optimizer;

public:
    float learning_rate;
    SNNTrainer(vector<IFNeuron*>& all_neurons, vector<IFNeuron*>& input_neurons,
        vector<IFNeuron*>& output_neurons, int time_steps = 100, float lr = 0.001f,
        const string& opt_type = "sgd")
        : all_neurons(all_neurons), input_neurons(input_neurons), output_neurons(output_neurons),
        time_steps(time_steps), learning_rate(lr), optimizer_type(opt_type), adam_optimizer(lr) {

        // Set learning rate for all neurons
        for (auto neuron : all_neurons) {
            neuron->setLearningRate(learning_rate);
        }

        // Initialize Adam optimizer states if using Adam
        if (optimizer_type == "adam") {
            for (auto neuron : all_neurons) {
                neuron->resetAdamState();
            }
        }

        // Initialize traces
        membrane_traces.resize(all_neurons.size());
        spike_traces.resize(all_neurons.size());
        for (size_t i = 0; i < all_neurons.size(); i++) {
            membrane_traces[i].resize(time_steps);
            spike_traces[i].resize(time_steps);
        }
    }

    // Forward pass through the network
    void forward(const vector<vector<int>>& input_spikes) {
        // Reset all neurons
        for (auto neuron : all_neurons) {
            neuron->resetState();
        }

        // Simulate time steps
        for (int t = 0; t < time_steps; t++) {
            // Apply input spikes
            for (size_t i = 0; i < input_neurons.size() && i < input_spikes.size(); i++) {
                if (t < input_spikes[i].size() && input_spikes[i][t] > 0) {
                    input_neurons[i]->integrate(1.0f);
                }
            }

            // Update all neurons
            for (auto neuron : all_neurons) {
                neuron->fire();
            }

            // Record traces
            for (size_t i = 0; i < all_neurons.size(); i++) {
                membrane_traces[i][t] = all_neurons[i]->getMembranePotential();
                spike_traces[i][t] = all_neurons[i]->getSpikeState();
            }
        }
    }

    // Backward pass using surrogate gradients
    void backward(int target_label, const string& loss_type = "crossentropy") {
        // Clear gradients
        for (auto neuron : all_neurons) {
            neuron->clearGradients();
        }

        // Compute loss gradients at each timestep
        for (int t = 0; t < time_steps; t++) {
            // Get output membrane potentials at time t
            vector<float> output_potentials;
            for (auto neuron : output_neurons) {
                int neuron_idx = find(all_neurons.begin(), all_neurons.end(), neuron) - all_neurons.begin();
                output_potentials.push_back(membrane_traces[neuron_idx][t]);
            }

            // Compute loss gradients
            vector<float> loss_gradients;
            if (loss_type == "crossentropy") {
                loss_gradients = LossFunction::timestepCrossEntropyGradient(output_potentials, target_label);
            }
            else if (loss_type == "potential") {
                loss_gradients = LossFunction::timestepPotentialGradient(output_potentials, target_label);
            }

            // Accumulate gradients for output neurons
            for (size_t i = 0; i < output_neurons.size(); i++) {
                output_neurons[i]->accumulateGradient(loss_gradients[i]);
            }
        }

        // Backward pass through network in reverse topological order
        topologicalBackward();
    }

    // Topological backward pass for graph networks
    void topologicalBackward() {
        // Compute in-degree for each neuron
        unordered_map<IFNeuron*, int> in_degree;
        for (auto neuron : all_neurons) {
            in_degree[neuron] = neuron->getPresynapticNeurons().size();
        }

        // Start from output neurons (assuming they have no postsynaptic connections for loss)
        queue<IFNeuron*> queue;
        for (auto neuron : output_neurons) {
            queue.push(neuron);
        }

        // Process neurons in reverse topological order
        while (!queue.empty()) {
            IFNeuron* current = queue.front();
            queue.pop();

            // Perform backward pass for current neuron
            current->backwardPass();

            // Add presynaptic neurons to queue if all their postsynaptic neurons are processed
            for (auto pre_neuron : current->getPresynapticNeurons()) {
                in_degree[pre_neuron]--;
                if (in_degree[pre_neuron] == 0) {
                    queue.push(pre_neuron);
                }
            }
        }
    }

    // Update weights using accumulated gradients
    void updateWeights() {
        if (optimizer_type == "adam") {
            for (auto neuron : all_neurons) {
                neuron->updateWeightsAdam();
            }
        }
        else {
            for (auto neuron : all_neurons) {
                neuron->updateWeights();
            }
        }
    }

    // Complete training step
    float trainStep(const vector<vector<int>>& input_spikes, int target_label,
        const string& loss_type = "crossentropy") {
        // Forward pass
        forward(input_spikes);

        // Compute loss
        float loss = computeLoss(target_label, loss_type);

        // Backward pass
        backward(target_label, loss_type);

        // Update weights
        updateWeights();

        return loss;
    }

    // Compute loss over all timesteps
    float computeLoss(int target_label, const string& loss_type = "crossentropy") {
        float total_loss = 0.0f;

        for (int t = 0; t < time_steps; t++) {
            vector<float> output_potentials;
            for (auto neuron : output_neurons) {
                int neuron_idx = find(all_neurons.begin(), all_neurons.end(), neuron) - all_neurons.begin();
                output_potentials.push_back(membrane_traces[neuron_idx][t]);
            }

            if (loss_type == "crossentropy") {
                total_loss += LossFunction::timestepCrossEntropyLoss(output_potentials, target_label);
            }
            else if (loss_type == "potential") {
                total_loss += LossFunction::timestepPotentialLoss(output_potentials, target_label);
            }
        }

        return total_loss / time_steps;
    }

    // Get prediction from spike counts
    int predict(const vector<vector<int>>& input_spikes) {
        forward(input_spikes);

        vector<int> spike_counts(output_neurons.size(), 0);
        for (size_t i = 0; i < output_neurons.size(); i++) {
            int neuron_idx = find(all_neurons.begin(), all_neurons.end(), output_neurons[i]) - all_neurons.begin();
            for (int t = 0; t < time_steps; t++) {
                if (spike_traces[neuron_idx][t]) {
                    spike_counts[i]++;
                }
            }
        }

        return max_element(spike_counts.begin(), spike_counts.end()) - spike_counts.begin();
    }

    // Get spike counts for analysis
    vector<int> getSpikeCountsLastRun() {
        vector<int> spike_counts(output_neurons.size(), 0);
        for (size_t i = 0; i < output_neurons.size(); i++) {
            int neuron_idx = find(all_neurons.begin(), all_neurons.end(), output_neurons[i]) - all_neurons.begin();
            for (int t = 0; t < time_steps; t++) {
                if (spike_traces[neuron_idx][t]) {
                    spike_counts[i]++;
                }
            }
        }
        return spike_counts;
    }

    void setLearningRate(float lr) {
        learning_rate = lr;
        adam_optimizer.setLearningRate(lr);
        for (auto neuron : all_neurons) {
            neuron->setLearningRate(learning_rate);
        }
    }

    void setOptimizer(const string& opt_type) {
        optimizer_type = opt_type;
        if (optimizer_type == "adam") {
            for (auto neuron : all_neurons) {
                neuron->resetAdamState();
            }
        }
    }

    string getOptimizer() const {
        return optimizer_type;
    }

    void setAdamParameters(float beta1, float beta2, float epsilon) {
        adam_optimizer.setBeta1(beta1);
        adam_optimizer.setBeta2(beta2);
        adam_optimizer.setEpsilon(epsilon);
    }

    // Checkpoint functionality
    bool saveCheckpoint(const string& checkpoint_path, int epoch, float loss, float accuracy) {
        try {
            // Create directory if it doesn't exist
            filesystem::create_directories(filesystem::path(checkpoint_path).parent_path());

            ofstream file(checkpoint_path, ios::binary);
            if (!file.is_open()) {
                return false;
            }

            // Save metadata
            file.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
            file.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
            file.write(reinterpret_cast<const char*>(&accuracy), sizeof(accuracy));
            file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));

            // Save optimizer type - use uint64_t for consistency
            uint64_t opt_len = static_cast<uint64_t>(optimizer_type.length());
            file.write(reinterpret_cast<const char*>(&opt_len), sizeof(opt_len));
            file.write(optimizer_type.c_str(), opt_len);

            // Save Adam parameters if using Adam
            if (optimizer_type == "adam") {
                float beta1 = adam_optimizer.getBeta1();
                float beta2 = adam_optimizer.getBeta2();
                float epsilon = adam_optimizer.getEpsilon();
                file.write(reinterpret_cast<const char*>(&beta1), sizeof(beta1));
                file.write(reinterpret_cast<const char*>(&beta2), sizeof(beta2));
                file.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
            }

            // Save network structure info - use uint64_t for consistency
            uint64_t num_neurons = static_cast<uint64_t>(all_neurons.size());
            uint64_t num_inputs = static_cast<uint64_t>(input_neurons.size());
            uint64_t num_outputs = static_cast<uint64_t>(output_neurons.size());
            uint64_t time_steps_64 = static_cast<uint64_t>(time_steps);
            file.write(reinterpret_cast<const char*>(&num_neurons), sizeof(num_neurons));
            file.write(reinterpret_cast<const char*>(&num_inputs), sizeof(num_inputs));
            file.write(reinterpret_cast<const char*>(&num_outputs), sizeof(num_outputs));
            file.write(reinterpret_cast<const char*>(&time_steps_64), sizeof(time_steps_64));

            // Save neuron states and weights
            for (auto neuron : all_neurons) {
                neuron->saveState(file);
            }

            file.close();
            return true;
        }
        catch (const exception& e) {
            return false;
        }
    }

    bool loadCheckpoint(const string& checkpoint_path, int& epoch, float& loss, float& accuracy) {
        try {
            ifstream file(checkpoint_path, ios::binary);
            if (!file.is_open()) {
                return false;
            }

            // Load metadata
            file.read(reinterpret_cast<char*>(&epoch), sizeof(epoch));
            if (!file.good()) { cout << "Failed to read epoch" << endl; return false; }
            file.read(reinterpret_cast<char*>(&loss), sizeof(loss));
            if (!file.good()) { cout << "Failed to read loss" << endl; return false; }
            file.read(reinterpret_cast<char*>(&accuracy), sizeof(accuracy));
            if (!file.good()) { cout << "Failed to read accuracy" << endl; return false; }
            file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
            if (!file.good()) { cout << "Failed to read learning_rate" << endl; return false; }
            cout << "Loaded metadata: epoch=" << epoch << ", loss=" << loss << ", accuracy=" << accuracy << endl;

            // Load optimizer type - use uint64_t for consistency
            uint64_t opt_len;
            file.read(reinterpret_cast<char*>(&opt_len), sizeof(opt_len));
            if (!file.good()) { cout << "Failed to read optimizer length" << endl; return false; }
            optimizer_type.resize(opt_len);
            file.read(&optimizer_type[0], opt_len);
            if (!file.good()) { cout << "Failed to read optimizer type" << endl; return false; }
            cout << "Loaded optimizer type: " << optimizer_type << endl;

            // Load Adam parameters if using Adam
            if (optimizer_type == "adam") {
                float beta1, beta2, epsilon;
                file.read(reinterpret_cast<char*>(&beta1), sizeof(beta1));
                if (!file.good()) { cout << "Failed to read beta1" << endl; return false; }
                file.read(reinterpret_cast<char*>(&beta2), sizeof(beta2));
                if (!file.good()) { cout << "Failed to read beta2" << endl; return false; }
                file.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));
                if (!file.good()) { cout << "Failed to read epsilon" << endl; return false; }
                adam_optimizer.setBeta1(beta1);
                adam_optimizer.setBeta2(beta2);
                adam_optimizer.setEpsilon(epsilon);
                cout << "Loaded Adam parameters: beta1=" << beta1 << ", beta2=" << beta2 << ", epsilon=" << epsilon << endl;
            }

            // Load and verify network structure - use uint64_t for consistency
            uint64_t num_neurons, num_inputs, num_outputs, loaded_time_steps;
            file.read(reinterpret_cast<char*>(&num_neurons), sizeof(num_neurons));
            if (!file.good()) { cout << "Failed to read num_neurons" << endl; return false; }
            file.read(reinterpret_cast<char*>(&num_inputs), sizeof(num_inputs));
            if (!file.good()) { cout << "Failed to read num_inputs" << endl; return false; }
            file.read(reinterpret_cast<char*>(&num_outputs), sizeof(num_outputs));
            if (!file.good()) { cout << "Failed to read num_outputs" << endl; return false; }
            file.read(reinterpret_cast<char*>(&loaded_time_steps), sizeof(loaded_time_steps));
            if (!file.good()) { cout << "Failed to read time_steps" << endl; return false; }
            cout << "Loaded network structure: neurons=" << num_neurons << ", inputs=" << num_inputs 
                 << ", outputs=" << num_outputs << ", time_steps=" << loaded_time_steps << endl;

            // Verify network structure matches
            cout << "Current network: neurons=" << all_neurons.size() << ", inputs=" << input_neurons.size() 
                 << ", outputs=" << output_neurons.size() << ", time_steps=" << time_steps << endl;
            if (num_neurons != static_cast<uint64_t>(all_neurons.size()) ||
                num_inputs != static_cast<uint64_t>(input_neurons.size()) ||
                num_outputs != static_cast<uint64_t>(output_neurons.size()) ||
                loaded_time_steps != static_cast<uint64_t>(time_steps)) {
                cout << "Network structure mismatch!" << endl;
                file.close();
                return false;
            }
            cout << "Network structure verified!" << endl;

            // Load neuron states and weights
            cout << "Loading neuron states for " << all_neurons.size() << " neurons..." << endl;
            for (size_t i = 0; i < all_neurons.size(); i++) {
                if (!all_neurons[i]->loadState(file)) {
                    cout << "Failed to load state for neuron " << i << endl;
                    file.close();
                    return false;
                }
                if (i % 100 == 0) {
                    cout << "Loaded " << i << " neurons..." << endl;
                }
            }
            cout << "All neuron states loaded successfully!" << endl;

            // Update learning rate for all neurons
            setLearningRate(learning_rate);

            file.close();
            return true;
        }
        catch (const exception& e) {
            return false;
        }
    }

    string generateCheckpointPath(const string& base_dir, int epoch) const {
        return base_dir + "/checkpoint_epoch_" + to_string(epoch) + ".bin";
    }

    string generateBestModelPath(const string& base_dir) const {
        return base_dir + "/best_model.bin";
    }
};