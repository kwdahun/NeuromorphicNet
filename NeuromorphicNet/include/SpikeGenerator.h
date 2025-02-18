#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>

class MNISTSpikeGenerator {
public:
    inline MNISTSpikeGenerator(double duration = 0.1, double dt = 0.001, double f_max = 100.0)
        : duration_(duration)
        , dt_(dt)
        , f_max_(f_max)
        , time_steps_(static_cast<int>(duration / dt))
        , num_images_(0)
        , image_size_(0) {
        // Initialize random number generator with time-based seed
        rng_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }

    inline uint32_t readBigEndian(const uint8_t* data) {
        return (static_cast<uint32_t>(data[0]) << 24) |
            (static_cast<uint32_t>(data[1]) << 16) |
            (static_cast<uint32_t>(data[2]) << 8) |
            static_cast<uint32_t>(data[3]);
    }

    inline bool readImages(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open image file: " << filename << std::endl;
            return false;
        }

        uint8_t header[16];
        file.read(reinterpret_cast<char*>(header), 16);

        uint32_t magic = readBigEndian(header);
        if (magic != 0x803) {
            std::cerr << "Invalid image file format" << std::endl;
            return false;
        }

        num_images_ = readBigEndian(header + 4);
        uint32_t rows = readBigEndian(header + 8);
        uint32_t cols = readBigEndian(header + 12);
        image_size_ = rows * cols;

        // Read all images
        images_.resize(num_images_);
        for (auto& image : images_) {
            image.resize(image_size_);
            file.read(reinterpret_cast<char*>(image.data()), image_size_);
        }

        return true;
    }

    inline bool readLabels(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open label file: " << filename << std::endl;
            return false;
        }

        uint8_t header[8];
        file.read(reinterpret_cast<char*>(header), 8);

        uint32_t magic = readBigEndian(header);
        if (magic != 0x801) {
            std::cerr << "Invalid label file format" << std::endl;
            return false;
        }

        uint32_t num_labels = readBigEndian(header + 4);
        if (num_labels != num_images_) {
            std::cerr << "Number of labels doesn't match number of images" << std::endl;
            return false;
        }

        // Read all labels
        labels_.resize(num_labels);
        file.read(reinterpret_cast<char*>(labels_.data()), num_labels);

        return true;
    }

    inline std::vector<std::vector<int>> generateSpikes(const std::vector<uint8_t>& image) {
        std::vector<std::vector<int>> spikes(image_size_, std::vector<int>(time_steps_));

        // For MNIST, we know the pixel values are 0-255
        const double p_min = 0.0;
        const double p_max = 255.0;
        const double norm_factor = p_max - p_min;

        // Generate spikes for each pixel
        for (size_t i = 0; i < image_size_; ++i) {
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

    // Getters
    size_t getNumImages() const { return num_images_; }
    size_t getImageSize() const { return image_size_; }
    int getTimeSteps() const { return time_steps_; }
    const std::vector<uint8_t>& getLabels() const { return labels_; }
    const std::vector<std::vector<uint8_t>>& getImages() const { return images_; }

private:
    // Spike generation parameters
    double duration_;
    double dt_;
    double f_max_;
    int time_steps_;

    // MNIST dataset properties
    size_t num_images_;
    size_t image_size_;
    std::vector<std::vector<uint8_t>> images_;
    std::vector<uint8_t> labels_;

    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
};