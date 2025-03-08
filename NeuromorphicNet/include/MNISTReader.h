#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <algorithm>

class MNISTReader {
public:
    MNISTReader() : num_images_(0), image_size_(0) {}

    uint32_t readBigEndian(const uint8_t* data) {
        return (static_cast<uint32_t>(data[0]) << 24) |
            (static_cast<uint32_t>(data[1]) << 16) |
            (static_cast<uint32_t>(data[2]) << 8) |
            static_cast<uint32_t>(data[3]);
    }

    bool readImages(const std::string& filename) {
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

    bool readLabels(const std::string& filename) {
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

    // Getters
    size_t getNumImages() const { return num_images_; }
    size_t getImageSize() const { return image_size_; }
    const std::vector<uint8_t>& getLabels() const { return labels_; }
    const std::vector<std::vector<uint8_t>>& getImages() const { return images_; }

private:
    // MNIST dataset properties
    size_t num_images_;
    size_t image_size_;
    std::vector<std::vector<uint8_t>> images_;
    std::vector<uint8_t> labels_;
};