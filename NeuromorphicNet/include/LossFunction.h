#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class LossFunction {
public:
    // Per-timestep cross-entropy loss for membrane potentials
    static float timestepCrossEntropyLoss(const vector<float>& membrane_potentials, int target_label, float temperature = 1.0f) {
        // Convert membrane potentials to probabilities using softmax
        vector<float> probabilities = softmaxFromPotentials(membrane_potentials, temperature);

        // Compute cross-entropy loss: -log(p_target)
        float loss = -log(max(probabilities[target_label], 1e-7f)); // Avoid log(0)
        return loss;
    }

    // Per-timestep spike loss - encourages target neuron to spike at specific timestep
    static float timestepSpikeLoss(const vector<bool>& spikes, int target_label, float target_spike_prob = 0.8f) {
        float loss = 0.0f;
        for (size_t i = 0; i < spikes.size(); ++i) {
            if (i == target_label) {
                // Target neuron should spike
                loss += spikes[i] ? 0.0f : -log(1.0f - target_spike_prob);
            }
            else {
                // Non-target neurons should not spike
                loss += spikes[i] ? -log(target_spike_prob) : 0.0f;
            }
        }
        return loss;
    }

    // Per-timestep membrane potential loss
    static float timestepPotentialLoss(const vector<float>& membrane_potentials, int target_label, float target_potential = 1.2f) {
        float loss = 0.0f;
        for (size_t i = 0; i < membrane_potentials.size(); ++i) {
            if (i == target_label) {
                // Target neuron should have high membrane potential
                float diff = membrane_potentials[i] - target_potential;
                loss += diff * diff;
            }
            else {
                // Non-target neurons should have low membrane potential
                loss += membrane_potentials[i] * membrane_potentials[i];
            }
        }
        return loss / static_cast<float>(membrane_potentials.size());
    }

    // Gradient of per-timestep cross-entropy loss w.r.t. membrane potentials
    static vector<float> timestepCrossEntropyGradient(const vector<float>& membrane_potentials, int target_label, float temperature = 1.0f) {
        vector<float> probabilities = softmaxFromPotentials(membrane_potentials, temperature);
        vector<float> gradients(membrane_potentials.size());

        for (size_t i = 0; i < membrane_potentials.size(); ++i) {
            if (i == target_label) {
                gradients[i] = (probabilities[i] - 1.0f) / temperature;
            }
            else {
                gradients[i] = probabilities[i] / temperature;
            }
        }
        return gradients;
    }

    // Gradient of per-timestep potential loss w.r.t. membrane potentials
    static vector<float> timestepPotentialGradient(const vector<float>& membrane_potentials, int target_label, float target_potential = 1.2f) {
        vector<float> gradients(membrane_potentials.size());
        float n = static_cast<float>(membrane_potentials.size());

        for (size_t i = 0; i < membrane_potentials.size(); ++i) {
            if (i == target_label) {
                gradients[i] = 2.0f * (membrane_potentials[i] - target_potential) / n;
            }
            else {
                gradients[i] = 2.0f * membrane_potentials[i] / n;
            }
        }
        return gradients;
    }

    // Cross-entropy loss for spike count classification
    static float crossEntropyLoss(const vector<int>& spike_counts, int target_label) {
        // Convert spike counts to probabilities using softmax
        vector<float> probabilities = softmax(spike_counts);

        // Compute cross-entropy loss: -log(p_target)
        float loss = -log(max(probabilities[target_label], 1e-7f)); // Avoid log(0)
        return loss;
    }

    // Mean squared error loss for spike count regression
    static float mseLoss(const vector<int>& spike_counts, const vector<float>& targets) {
        float loss = 0.0f;
        for (size_t i = 0; i < spike_counts.size(); ++i) {
            float diff = static_cast<float>(spike_counts[i]) - targets[i];
            loss += diff * diff;
        }
        return loss / static_cast<float>(spike_counts.size());
    }

    // Spike count loss - encourages target neuron to spike more
    static float spikeCountLoss(const vector<int>& spike_counts, int target_label, float target_rate = 10.0f) {
        float loss = 0.0f;
        for (size_t i = 0; i < spike_counts.size(); ++i) {
            if (i == target_label) {
                // Target neuron should spike at target_rate
                float diff = static_cast<float>(spike_counts[i]) - target_rate;
                loss += diff * diff;
            }
            else {
                // Non-target neurons should have low spike counts
                loss += static_cast<float>(spike_counts[i] * spike_counts[i]);
            }
        }
        return loss / static_cast<float>(spike_counts.size());
    }

    // Temporal coding loss - considers spike timing
    static float temporalLoss(const vector<vector<bool>>& spike_trains, int target_label,
        const vector<float>& time_weights) {
        float loss = 0.0f;
        int time_steps = spike_trains[0].size();

        for (size_t neuron_idx = 0; neuron_idx < spike_trains.size(); ++neuron_idx) {
            for (int t = 0; t < time_steps; ++t) {
                float weight = time_weights[t];
                if (neuron_idx == target_label) {
                    // Target neuron should spike early (higher weight for early times)
                    if (!spike_trains[neuron_idx][t]) {
                        loss += weight; // Penalty for not spiking when expected
                    }
                }
                else {
                    // Non-target neurons should not spike, especially early
                    if (spike_trains[neuron_idx][t]) {
                        loss += weight; // Penalty for spiking when not expected
                    }
                }
            }
        }
        return loss;
    }

    // Gradient of cross-entropy loss w.r.t. spike counts
    static vector<float> crossEntropyGradient(const vector<int>& spike_counts, int target_label) {
        vector<float> probabilities = softmax(spike_counts);
        vector<float> gradients(spike_counts.size());

        for (size_t i = 0; i < spike_counts.size(); ++i) {
            if (i == target_label) {
                gradients[i] = probabilities[i] - 1.0f;
            }
            else {
                gradients[i] = probabilities[i];
            }
        }
        return gradients;
    }

    // Gradient of spike count loss w.r.t. spike counts
    static vector<float> spikeCountGradient(const vector<int>& spike_counts, int target_label,
        float target_rate = 10.0f) {
        vector<float> gradients(spike_counts.size());
        float n = static_cast<float>(spike_counts.size());

        for (size_t i = 0; i < spike_counts.size(); ++i) {
            if (i == target_label) {
                gradients[i] = 2.0f * (static_cast<float>(spike_counts[i]) - target_rate) / n;
            }
            else {
                gradients[i] = 2.0f * static_cast<float>(spike_counts[i]) / n;
            }
        }
        return gradients;
    }

private:
    // Softmax function to convert membrane potentials to probabilities
    static vector<float> softmaxFromPotentials(const vector<float>& potentials, float temperature = 1.0f) {
        vector<float> probabilities(potentials.size());
        float max_potential = *max_element(potentials.begin(), potentials.end());

        float sum = 0.0f;
        for (size_t i = 0; i < potentials.size(); ++i) {
            probabilities[i] = exp((potentials[i] - max_potential) / temperature);
            sum += probabilities[i];
        }

        for (size_t i = 0; i < probabilities.size(); ++i) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    // Softmax function to convert spike counts to probabilities
    static vector<float> softmax(const vector<int>& spike_counts) {
        vector<float> probabilities(spike_counts.size());
        float max_count = *max_element(spike_counts.begin(), spike_counts.end());

        float sum = 0.0f;
        for (size_t i = 0; i < spike_counts.size(); ++i) {
            probabilities[i] = exp(static_cast<float>(spike_counts[i]) - max_count);
            sum += probabilities[i];
        }

        for (size_t i = 0; i < probabilities.size(); ++i) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }
};