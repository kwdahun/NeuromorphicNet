#pragma once
#include <iostream>
#include <vector>

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

public:
    IFNeuron(int id, float membrane_potential, float threshold, float leakage_ratio) {
        this->id = id;
        this->membrane_potential = membrane_potential;
        this->threshold = threshold;
        this->leakage_ratio = leakage_ratio;
    }
    IFNeuron(int id, float threshold, float leakage_ratio) {
        this->id = id;
        this->membrane_potential = 0.0f;
        this->threshold = threshold;
        this->leakage_ratio = leakage_ratio;
    }
    IFNeuron(int id) {
        this->id = id;
        this->membrane_potential = 0;
        this->threshold = 1.0;
        this->leakage_ratio = 0.95;
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
        }
        postsynaptic_neuron->presynaptic_neurons.push_back(this);
    }

    // applied every one time step to all neurons consisting spiking neural net
    void fire() {
        bool fired = membrane_potential > threshold;

        if (fired) {
            // DEBUG Àü¿ë
            /*if (id < 784) cout << "Input[" << id << "] ";
            else if (id < 1296) cout << "Hidden1[" << id << "] ";
            else if (id < 1552) cout << "Hidden2[" << id << "] ";
            else if (id < 1680) cout << "Hidden3[" << id << "] ";
            else cout << "Output[" << id << "] ";*/

            for (size_t i = 0; i < postsynaptic_weights.size(); i++) {
                postsynaptic_neurons[i]->integrate(postsynaptic_weights[i]);
            }
        }

        membrane_potential = leakage_ratio * membrane_potential - fired * threshold;
    }

    void integrate(float stimulus) {
        if (isfinite(stimulus)) {
            membrane_potential = membrane_potential + stimulus;
        }
    }

    const std::vector<IFNeuron*>& getPresynapticNeurons() const { return this->presynaptic_neurons; }
    const vector<IFNeuron*>& getPostSynapticNeurons() const { return this->postsynaptic_neurons; }
    int getId() const { return id; }
    float getMembranePotential() const { return membrane_potential; }
    float getThreshold() const { return threshold; }
    float getLeakageRatio() const { return leakage_ratio; }
    void setMembranePotential(float value) { membrane_potential = value; }
    void setThreshold(float value) { threshold = value; }
    void setLeakageRatio(float value) { leakage_ratio = value; }
};