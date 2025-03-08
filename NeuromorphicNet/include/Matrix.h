#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <omp.h>
#include <cblas.h>

using namespace std;

class Matrix {
private:
    vector<float> data;
    size_t batch_size_;
    size_t channels_;
    size_t height_;
    size_t width_;

    size_t get_index(size_t b, size_t c, size_t h, size_t w) const {
        return ((b * channels_ + c) * height_ + h) * width_ + w;
    }

public:
    // Constructor
    Matrix(size_t batch_size, size_t channels, size_t height, size_t width)
        : batch_size_(batch_size), channels_(channels), height_(height), width_(width) {
        size_t total_size = batch_size * channels * height * width;
        data.resize(total_size, 0.0f);
    }

    Matrix(size_t channels, size_t height, size_t width)
        : Matrix(1, channels, height, width) {}

    Matrix(const Matrix& other)
        : batch_size_(other.batch_size_)
        , channels_(other.channels_)
        , height_(other.height_)
        , width_(other.width_)
        , data(other.data) {
    }

    Matrix() : batch_size_(0), channels_(0), height_(0), width_(0) {}

    // Indexing
    float& at(size_t b, size_t c, size_t h, size_t w) {
        return data[get_index(b, c, h, w)];
    }

    const float& at(size_t b, size_t c, size_t h, size_t w) const {
        return data[get_index(b, c, h, w)];
    }

    // Operator Overloading
    Matrix operator*(const Matrix& other) const {
        if (width_ != other.height_) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }

        if (batch_size_ != other.batch_size_ || channels_ != other.channels_) {
            throw std::invalid_argument("Batch size or channels don't match");
        }

        Matrix result(batch_size_, channels_, height_, other.width_);

#pragma omp parallel for collapse(2)
        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t c = 0; c < channels_; ++c) {
                const float* A = &data[get_index(b, c, 0, 0)];
                const float* B = &other.data[other.get_index(b, c, 0, 0)];
                float* C = &result.data[result.get_index(b, c, 0, 0)];

                cblas_sgemm(
                    CblasRowMajor,    // Order
                    CblasNoTrans,     // TransA
                    CblasNoTrans,     // TransB
                    height_,          // M (rows of A)
                    other.width_,     // N (cols of B)
                    width_,           // K (cols of A = rows of B)
                    1.0f,             // alpha
                    A,                // A matrix
                    width_,           // lda (leading dimension of A)
                    B,                // B matrix
                    other.width_,     // ldb (leading dimension of B)
                    0.0f,             // beta
                    C,                // C matrix
                    other.width_      // ldc (leading dimension of C)
                );
            }
        }

        return result;
    }

    Matrix operator+(const Matrix& other) const {
        check_dimensions(other);
        Matrix result(batch_size_, channels_, height_, width_);
        size_t total_size = data.size();

#pragma omp parallel for simd
        for (size_t i = 0; i < total_size; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    void check_dimensions(const Matrix& other) const {
        if (batch_size_ != other.batch_size_ ||
            channels_ != other.channels_ ||
            height_ != other.height_ ||
            width_ != other.width_) {
            throw std::invalid_argument("Matrix dimensions don't match");
        }
    }

    // Debugging
    void print() const {
        for (size_t b = 0; b < batch_size_; ++b) {
            std::cout << "Batch " << b << ":\n";
            for (size_t c = 0; c < channels_; ++c) {
                std::cout << " Channel " << c << ":\n";
                for (size_t h = 0; h < height_; ++h) {
                    std::cout << "  ";
                    for (size_t w = 0; w < width_; ++w) {
                        printf("%6.2f ", at(b, c, h, w));
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    void print_color() const {
        std::string buffer;
        buffer.reserve(batch_size_ * channels_ * height_ * width_ * 20);

        for (size_t b = 0; b < batch_size_; ++b) {
            buffer += "Batch " + std::to_string(b) + ":\n";

            if (channels_ == 3) {
                for (size_t h = 0; h < height_; ++h) {
                    buffer += "  ";
                    for (size_t w = 0; w < width_; ++w) {
                        char temp[32];
                        int r = static_cast<int>(at(b, 0, h, w) * 255);
                        int g = static_cast<int>(at(b, 1, h, w) * 255);
                        int b_val = static_cast<int>(at(b, 2, h, w) * 255);

                        snprintf(temp, sizeof(temp), "\033[48;2;%d;%d;%dm  \033[0m", r, g, b_val);
                        buffer += temp;
                    }
                    buffer += '\n';
                }
            }
            else {
                for (size_t c = 0; c < channels_; ++c) {
                    for (size_t h = 0; h < height_; ++h) {
                        buffer += "  ";
                        for (size_t w = 0; w < width_; ++w) {
                            char temp[32];
                            int intensity = static_cast<int>(at(b, c, h, w) * 255);
                            snprintf(temp, sizeof(temp), "\033[48;2;%d;%d;%dm  \033[0m",
                                intensity, intensity, intensity);
                            buffer += temp;
                        }
                        buffer += '\n';
                    }
                    if (channels_ > 1) buffer += '\n';
                }
            }
            buffer += '\n';
        }

        std::cout << buffer;
    }

    // Getters
    size_t batch_size() const { return batch_size_; }
    size_t channels() const { return channels_; }
    size_t height() const { return height_; }
    size_t width() const { return width_; }
};