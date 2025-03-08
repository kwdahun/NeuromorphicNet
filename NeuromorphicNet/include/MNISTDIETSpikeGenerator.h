#pragma once

#include "SpikeGenerator.h"
#include "MNISTReader.h"
#include <vector>
#include <cstdint>
#include <algorithm>

class MNISTDIETSpikeGenerator : public SpikeGenerator {
private:
    MNISTReader mnist_reader_;
public:
    MNISTDIETSpikeGenerator(int time_steps_)
        : SpikeGenerator(time_steps_), mnist_reader_() {}

    bool readImages(const std::string& filename) {
        return mnist_reader_.readImages(filename);
    }

    bool readLabels(const std::string& filename) {
        return mnist_reader_.readLabels(filename);
    }

    // Implementation of the pure virtual method from base class
    std::vector<std::vector<int>> generateSpikes(const std::vector<uint8_t>& image) override {
        size_t image_size = image.size();
        std::vector<std::vector<int>> spikes(image_size, std::vector<int>(time_steps_));

        // For MNIST, we know the pixel values are 0-255
        const double p_min = 0.0;
        const double p_max = 255.0;
        const double norm_factor = p_max - p_min;

        // Generate spikes for each pixel
        for (size_t i = 0; i < image_size; ++i) {
            // Normalize pixel value to [0,1]
            double normalized_input = static_cast<double>(image[i]) / norm_factor;
            double firing_prob = normalized_input * f_max_ * dt_;

            // Ensure probability is in valid range [0,1]
            firing_prob = std::min(1.0, std::max(0.0, firing_prob));

            for (int t = 0; t < time_steps_; ++t) {
                spikes[i][t] = (dist_(rng_) < firing_prob) ? 1 : 0;
            }
        }

        return spikes;
    }

    // MNIST-specific getters
    size_t getNumImages() const { return mnist_reader_.getNumImages(); }
    size_t getImageSize() const { return mnist_reader_.getImageSize(); }
    const std::vector<uint8_t>& getLabels() const { return mnist_reader_.getLabels(); }
    const std::vector<std::vector<uint8_t>>& getImages() const { return mnist_reader_.getImages(); }
};