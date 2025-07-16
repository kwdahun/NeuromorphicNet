#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

using namespace std;

class IFNeuron {
private:
    int id;
    float membrane_potential;
    float threshold;
    float leakage_ratio; // lambda that is multiplied to membrane potential at t-1
    vector<IFNeuron*> presynaptic_neurons;
    vector<IFNeuron*> postsynaptic_neurons;
    vector<float> postsynaptic_weights;

    // Surrogate gradient training variables
    float gradient_potential;
    float gradient_threshold;
    vector<float> gradient_weights;
    float learning_rate;
    bool spike_state;

    // Adam optimizer variables
    vector<float> m_weights;  // First moment (momentum)
    vector<float> v_weights;  // Second moment (RMSprop)
    float m_threshold;
    float v_threshold;
    int time_step;  // For bias correction

public:
    IFNeuron(int id, float membrane_potential, float threshold, float leakage_ratio) {
        this->id = id;
        this->membrane_potential = membrane_potential;
        this->threshold = threshold;
        this->leakage_ratio = leakage_ratio;
        this->gradient_potential = 0.0f;
        this->gradient_threshold = 0.0f;
        this->learning_rate = 0.001f;
        this->spike_state = false;
        this->m_threshold = 0.0f;
        this->v_threshold = 0.0f;
        this->time_step = 0;
    }
    IFNeuron(int id, float threshold, float leakage_ratio) {
        this->id = id;
        this->membrane_potential = 0.0f;
        this->threshold = threshold;
        this->leakage_ratio = leakage_ratio;
        this->gradient_potential = 0.0f;
        this->gradient_threshold = 0.0f;
        this->learning_rate = 0.001f;
        this->spike_state = false;
        this->m_threshold = 0.0f;
        this->v_threshold = 0.0f;
        this->time_step = 0;
    }
    IFNeuron(int id) {
        this->id = id;
        this->membrane_potential = 0;
        this->threshold = 1.0;
        this->leakage_ratio = 0.95;
        this->gradient_potential = 0.0f;
        this->gradient_threshold = 0.0f;
        this->learning_rate = 0.001f;
        this->spike_state = false;
        this->m_threshold = 0.0f;
        this->v_threshold = 0.0f;
        this->time_step = 0;
    }
    ~IFNeuron() {
        presynaptic_neurons.clear();
        postsynaptic_neurons.clear();
        postsynaptic_weights.clear();
    }

    void connectTo(IFNeuron* postsynaptic_neuron, float weight) {
        if (postsynaptic_neuron == this) {
            return;
        }

        for (auto neuron : postsynaptic_neurons) {
            if (neuron == postsynaptic_neuron) {
                return;
            }
        }

        this->postsynaptic_neurons.push_back(postsynaptic_neuron);
        if (isfinite(weight)) {
            this->postsynaptic_weights.push_back(weight);
            this->gradient_weights.push_back(0.0f);
            this->m_weights.push_back(0.0f);
            this->v_weights.push_back(0.0f);
        }
        postsynaptic_neuron->presynaptic_neurons.push_back(this);
    }

    // applied every one time step to all neurons consisting spiking neural net
    void fire() {
        bool fired = membrane_potential > threshold;
        spike_state = fired;

        if (fired) {
            for (size_t i = 0; i < postsynaptic_weights.size(); i++) {
                postsynaptic_neurons[i]->integrate(postsynaptic_weights[i]);
            }

            membrane_potential = 0;
        }
        else {
            membrane_potential = leakage_ratio * membrane_potential;
        }
    }

    void integrate(float stimulus) {
        if (isfinite(stimulus)) {
            membrane_potential = membrane_potential + stimulus;
        }
    }

    const std::vector<IFNeuron*>& getPresynapticNeurons() const { return this->presynaptic_neurons; }
    const vector<IFNeuron*>& getPostSynapticNeurons() const { return this->postsynaptic_neurons; }
    const vector<float>& getPostsynapticWeights() const { return this->postsynaptic_weights; }
    int getId() const { return id; }
    float getMembranePotential() const { return membrane_potential; }
    float getThreshold() const { return threshold; }
    float getLeakageRatio() const { return leakage_ratio; }
    bool getSpikeState() const { return spike_state; }
    void setMembranePotential(float value) { membrane_potential = value; }
    void setThreshold(float value) { threshold = value; }
    void setLeakageRatio(float value) { leakage_ratio = value; }
    void setLearningRate(float value) { learning_rate = value; }

    // Surrogate gradient methods
    float surrogateGradient(float membrane_potential, float threshold, float beta = 1.0f) {
        // Fast sigmoid surrogate function
        float diff = membrane_potential - threshold;
        return beta / (1.0f + abs(beta * diff));
    }

    void clearGradients() {
        gradient_potential = 0.0f;
        gradient_threshold = 0.0f;
        fill(gradient_weights.begin(), gradient_weights.end(), 0.0f);
    }

    void accumulateGradient(float grad) {
        gradient_potential += grad;
    }

    void backwardPass() {
        // Compute surrogate gradient
        float surrogate_grad = surrogateGradient(membrane_potential, threshold);

        // Propagate gradients to presynaptic neurons
        for (size_t i = 0; i < presynaptic_neurons.size(); i++) {
            IFNeuron* pre_neuron = presynaptic_neurons[i];

            // Find weight connecting pre_neuron to this neuron
            for (size_t j = 0; j < pre_neuron->postsynaptic_neurons.size(); j++) {
                if (pre_neuron->postsynaptic_neurons[j] == this) {
                    // Gradient w.r.t. weight
                    float weight_grad = gradient_potential * surrogate_grad * (pre_neuron->spike_state ? 1.0f : 0.0f);
                    pre_neuron->gradient_weights[j] += weight_grad;

                    // Gradient w.r.t. presynaptic membrane potential
                    float pre_grad = gradient_potential * surrogate_grad * pre_neuron->postsynaptic_weights[j];
                    pre_neuron->accumulateGradient(pre_grad);
                    break;
                }
            }
        }
    }

    void updateWeights() {
        // Update synaptic weights using accumulated gradients (SGD)
        for (size_t i = 0; i < postsynaptic_weights.size(); i++) {
            postsynaptic_weights[i] -= learning_rate * gradient_weights[i];
        }
    }

    void updateWeightsAdam(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f) {
        time_step++;

        // Update weights using Adam optimizer
        for (size_t i = 0; i < postsynaptic_weights.size(); i++) {
            // Update biased first moment estimate
            m_weights[i] = beta1 * m_weights[i] + (1.0f - beta1) * gradient_weights[i];

            // Update biased second moment estimate
            v_weights[i] = beta2 * v_weights[i] + (1.0f - beta2) * gradient_weights[i] * gradient_weights[i];

            // Compute bias-corrected first moment estimate
            float m_hat = m_weights[i] / (1.0f - pow(beta1, time_step));

            // Compute bias-corrected second moment estimate
            float v_hat = v_weights[i] / (1.0f - pow(beta2, time_step));

            // Update weights
            postsynaptic_weights[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }

    void resetState() {
        membrane_potential = 0.0f;
        spike_state = false;
        clearGradients();
    }

    void resetAdamState() {
        fill(m_weights.begin(), m_weights.end(), 0.0f);
        fill(v_weights.begin(), v_weights.end(), 0.0f);
        m_threshold = 0.0f;
        v_threshold = 0.0f;
        time_step = 0;
    }

    // Checkpoint functionality
    bool saveState(ofstream& file) const {
        try {
            // Save basic parameters
            file.write(reinterpret_cast<const char*>(&id), sizeof(id));
            if (!file.good()) return false;
            
            file.write(reinterpret_cast<const char*>(&membrane_potential), sizeof(membrane_potential));
            if (!file.good()) return false;
            
            file.write(reinterpret_cast<const char*>(&threshold), sizeof(threshold));
            if (!file.good()) return false;
            
            file.write(reinterpret_cast<const char*>(&leakage_ratio), sizeof(leakage_ratio));
            if (!file.good()) return false;
            
            file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
            if (!file.good()) return false;
            
            file.write(reinterpret_cast<const char*>(&time_step), sizeof(time_step));
            if (!file.good()) return false;

            // Save weights - use explicit uint64_t for cross-platform consistency
            uint64_t num_weights = static_cast<uint64_t>(postsynaptic_weights.size());
            file.write(reinterpret_cast<const char*>(&num_weights), sizeof(num_weights));
            if (!file.good()) return false;
            
            if (num_weights > 0) {
                file.write(reinterpret_cast<const char*>(postsynaptic_weights.data()),
                    num_weights * sizeof(float));
                if (!file.good()) return false;
            }

            // Save Adam optimizer states
            if (num_weights > 0) {
                file.write(reinterpret_cast<const char*>(m_weights.data()),
                    num_weights * sizeof(float));
                if (!file.good()) return false;
                
                file.write(reinterpret_cast<const char*>(v_weights.data()),
                    num_weights * sizeof(float));
                if (!file.good()) return false;
            }
            
            file.write(reinterpret_cast<const char*>(&m_threshold), sizeof(m_threshold));
            if (!file.good()) return false;
            
            file.write(reinterpret_cast<const char*>(&v_threshold), sizeof(v_threshold));
            if (!file.good()) return false;

            return true;
        }
        catch (const exception& e) {
            return false;
        }
    }

    bool loadState(ifstream& file) {
        try {
            // Load basic parameters
            int loaded_id;
            file.read(reinterpret_cast<char*>(&loaded_id), sizeof(loaded_id));
            if (!file.good()) {
                cout << "Failed to read neuron ID for neuron " << id << endl;
                return false;
            }
            if (loaded_id != id) {
                cout << "ID mismatch for neuron " << id << ": expected " << id << ", got " << loaded_id << endl;
                return false; // ID mismatch
            }

            file.read(reinterpret_cast<char*>(&membrane_potential), sizeof(membrane_potential));
            if (!file.good()) { cout << "Failed to read membrane_potential for neuron " << id << endl; return false; }
            file.read(reinterpret_cast<char*>(&threshold), sizeof(threshold));
            if (!file.good()) { cout << "Failed to read threshold for neuron " << id << endl; return false; }
            file.read(reinterpret_cast<char*>(&leakage_ratio), sizeof(leakage_ratio));
            if (!file.good()) { cout << "Failed to read leakage_ratio for neuron " << id << endl; return false; }
            file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
            if (!file.good()) { cout << "Failed to read learning_rate for neuron " << id << endl; return false; }
            file.read(reinterpret_cast<char*>(&time_step), sizeof(time_step));
            if (!file.good()) { cout << "Failed to read time_step for neuron " << id << endl; return false; }

            // Load weights - use explicit uint64_t for cross-platform consistency
            uint64_t num_weights;
            file.read(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
            if (!file.good()) { cout << "Failed to read num_weights for neuron " << id << endl; return false; }

            // Check for corrupted data (unreasonably large weight counts indicate corruption)
            if (num_weights > 100000) {  // Reasonable upper bound for weight count
                cout << "Corrupted checkpoint detected for neuron " << id << ": num_weights=" << num_weights << endl;
                return false;
            }
            
            if (num_weights != static_cast<uint64_t>(postsynaptic_weights.size())) {
                cout << "Weight count mismatch for neuron " << id << ": expected " << postsynaptic_weights.size() 
                     << ", got " << num_weights << endl;
                return false; // Weight count mismatch
            }

            if (num_weights > 0) {
                file.read(reinterpret_cast<char*>(postsynaptic_weights.data()),
                    num_weights * sizeof(float));
                if (!file.good()) { cout << "Failed to read postsynaptic_weights for neuron " << id << endl; return false; }

                // Load Adam optimizer states
                file.read(reinterpret_cast<char*>(m_weights.data()),
                    num_weights * sizeof(float));
                if (!file.good()) { cout << "Failed to read m_weights for neuron " << id << endl; return false; }
                file.read(reinterpret_cast<char*>(v_weights.data()),
                    num_weights * sizeof(float));
                if (!file.good()) { cout << "Failed to read v_weights for neuron " << id << endl; return false; }
            }

            file.read(reinterpret_cast<char*>(&m_threshold), sizeof(m_threshold));
            if (!file.good()) { cout << "Failed to read m_threshold for neuron " << id << endl; return false; }
            file.read(reinterpret_cast<char*>(&v_threshold), sizeof(v_threshold));
            if (!file.good()) { cout << "Failed to read v_threshold for neuron " << id << endl; return false; }

            return true;
        }
        catch (const exception& e) {
            return false;
        }
    }
};