#pragma once
#include <cmath>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <cblas.h>
#include "Matrix.h"

using namespace std;

class Conv2d {
private:
    Matrix kernel;
    Matrix bias;
    Matrix kernel_gradient;
    Matrix bias_gradient;
    Matrix padded_input_cache;

    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;

public:
    Conv2d(size_t in_channels, size_t out_channels,
        size_t kernel_size, size_t stride = 1,
        size_t padding = 0)
        : in_channels(in_channels)
        , out_channels(out_channels)
        , kernel_size(kernel_size)
        , stride(stride)
        , padding(padding)
        , kernel(out_channels, in_channels, kernel_size, kernel_size)
        , bias(out_channels, 1, 1, 1)
        , padded_input_cache(1, 1, 1, 1)
    {
        float limit = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
        limit *= sqrt(2.0f);
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t h = 0; h < kernel_size; ++h) {
                    for (size_t w = 0; w < kernel_size; ++w) {
                        float random_val = ((float)rand() / RAND_MAX) * 2 * limit - limit;
                        kernel.at(oc, ic, h, w) = random_val;
                    }
                }
            }
            bias.at(oc, 0, 0, 0) = 0.0f;
        }
    }

    pair<size_t, size_t> get_output_dims(size_t H, size_t W) const {
        size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
        size_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
        return { H_out, W_out };
    }

    vector<float> im2col(const Matrix& padded_input, size_t batch) const {
        size_t C = padded_input.channels();
        size_t H = padded_input.height() - 2 * padding;
        size_t W = padded_input.width() - 2 * padding;
        size_t KH = kernel_size;
        size_t KW = kernel_size;

        auto [H_out, W_out] = get_output_dims(H, W);

        size_t rows = KH * KW * C;
        size_t cols = H_out * W_out;
        vector<float> data(rows * cols);

#pragma omp parallel for collapse(4)
        for (size_t c = 0; c < C; c++) {
            for (size_t kh = 0; kh < KH; kh++) {
                for (size_t kw = 0; kw < KW; kw++) {
                    for (size_t oh = 0; oh < H_out; oh++) {
#pragma omp simd
                        for (size_t ow = 0; ow < W_out; ow++) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            size_t idx = (((c * KH + kh) * KW + kw) * H_out + oh) * W_out + ow;
                            data[idx] = padded_input.at(batch, c, ih, iw);
                        }
                    }
                }
            }
        }

        return data;
    }

    vector<float> reshape_kernel() const {
        size_t K = in_channels * kernel_size * kernel_size;
        vector<float> reshaped_kernel(out_channels * K);

#pragma omp parallel for collapse(2)
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        size_t k_idx = ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        size_t reshaped_idx = oc * K + k_idx;

                        reshaped_kernel[reshaped_idx] = kernel.at(oc, ic, kh, kw);
                    }
                }
            }
        }

        return reshaped_kernel;
    }

    Matrix forward(const Matrix& input) {
        size_t B = input.batch_size();
        size_t C = input.channels();
        size_t H = input.height();
        size_t W = input.width();

        if (B == 0 || C != in_channels || H == 0 || W == 0) {
            throw invalid_argument("Invalid input dimensions");
        }
        auto [H_out, W_out] = get_output_dims(H, W);
        if (H_out == 0 || W_out == 0) {
            throw invalid_argument("Invalid output dimensions");
        }

        Matrix padded_input(B, C, H + 2 * padding, W + 2 * padding);
#pragma omp parallel for collapse(4)
        for (size_t b = 0; b < B; b++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
#pragma omp simd
                    for (size_t w = 0; w < W; w++) {
                        padded_input.at(b, c, h + padding, w + padding) = input.at(b, c, h, w);
                    }
                }
            }
        }

        padded_input_cache = padded_input;

        vector<float> reshaped_kernel = reshape_kernel();

        Matrix output(B, out_channels, H_out, W_out);

#pragma omp parallel for
        for (size_t b = 0; b < B; b++) {

            vector<float> cols = im2col(padded_input, b);

            size_t M = out_channels;
            size_t K = in_channels * kernel_size * kernel_size;
            size_t N = H_out * W_out;

            vector<float> output_data(M * N);

            cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                K,
                1.0f,
                reshaped_kernel.data(),
                K,
                cols.data(),
                N,
                0.0f,
                output_data.data(),
                N);

            for (size_t m = 0; m < M; m++) {
#pragma omp simd
                for (size_t n = 0; n < N; n++) {
                    size_t oh = n / W_out;
                    size_t ow = n % W_out;
                    output.at(b, m, oh, ow) = output_data[m * N + n] + bias.at(m, 0, 0, 0);
                }
            }
        }

        return output;
    }

    Matrix backward(const Matrix& gradient) {
        size_t B = padded_input_cache.batch_size();
        size_t H = padded_input_cache.height();
        size_t W = padded_input_cache.width();
        size_t origH = H - 2 * padding;
        size_t origW = W - 2 * padding;
        auto [H_out, W_out] = get_output_dims(origH, origW);

        // Compute bias gradients
#pragma omp parallel for
        for (size_t oc = 0; oc < out_channels; oc++) {
            float bias_grad = 0.0f;
            for (size_t b = 0; b < B; b++) {
                for (size_t oh = 0; oh < H_out; oh++) {
#pragma omp simd reduction(+:bias_grad)
                    for (size_t ow = 0; ow < W_out; ow++) {
                        bias_grad += gradient.at(b, oc, oh, ow);
                    }
                }
            }
            // Directly accumulate into bias_gradient (outside parallel region over b, so no atomic needed)
            bias_gradient.at(oc, 0, 0, 0) += bias_grad;
        }

        Matrix tmp_input_gradients(B, in_channels, H, W);

#pragma omp parallel for
        for (size_t b = 0; b < B; b++) {
            size_t M = out_channels;
            size_t N = H_out * W_out;
            vector<float> reshaped_grad(M * N);

            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    size_t oh = n / W_out;
                    size_t ow = n % W_out;
                    reshaped_grad[m * N + n] = gradient.at(b, m, oh, ow);
                }
            }

            vector<float> cols = im2col(padded_input_cache, b);
            size_t K = in_channels * kernel_size * kernel_size;

            vector<float> kernel_grad(M * K);
            cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                M,
                K,
                N,
                1.0f,
                reshaped_grad.data(),
                N,
                cols.data(),
                N,
                1.0f,
                kernel_grad.data(),
                K);

            // Accumulate kernel gradients
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t ic = 0; ic < in_channels; ic++) {
                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            size_t k_idx = ic * kernel_size * kernel_size + kh * kernel_size + kw;
                            float grad = kernel_grad[oc * K + k_idx];
#pragma omp atomic
                            kernel_gradient.at(oc, ic, kh, kw) += grad;
                        }
                    }
                }
            }

            vector<float> reshaped_kernel = reshape_kernel();
            vector<float> input_grad(K * N);
            cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                K,
                N,
                M,
                1.0f,
                reshaped_kernel.data(),
                K,
                reshaped_grad.data(),
                N,
                0.0f,
                input_grad.data(),
                N);

            // Compute input gradients
            for (size_t ic = 0; ic < in_channels; ic++) {
                for (size_t kh = 0; kh < kernel_size; kh++) {
                    for (size_t kw = 0; kw < kernel_size; kw++) {
                        for (size_t oh = 0; oh < H_out; oh++) {
                            for (size_t ow = 0; ow < W_out; ow++) {
                                size_t k_idx = ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                size_t n_idx = oh * W_out + ow;
                                size_t ih = oh * stride + kh;
                                size_t iw = ow * stride + kw;
                                // No atomic needed; each b is handled by a separate thread
                                tmp_input_gradients.at(b, ic, ih, iw) += input_grad[k_idx * N + n_idx];
                            }
                        }
                    }
                }
            }
        }

        Matrix input_gradients(B, in_channels, origH, origW);
#pragma omp parallel for collapse(3)
        for (size_t b = 0; b < B; b++) {
            for (size_t c = 0; c < in_channels; c++) {
                for (size_t h = 0; h < origH; h++) {
#pragma omp simd
                    for (size_t w = 0; w < origW; w++) {
                        input_gradients.at(b, c, h, w) =
                            tmp_input_gradients.at(b, c, h + padding, w + padding);
                    }
                }
            }
        }

        return input_gradients;
    }

    void update_parameters(float learning_rate) {
        // Update kernel
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        kernel.at(oc, ic, kh, kw) -= learning_rate * kernel_gradient.at(oc, ic, kh, kw);
                    }
                }
            }
            // Update bias
            bias.at(oc, 0, 0, 0) -= learning_rate * bias_gradient.at(oc, 0, 0, 0);
        }
        // Zero gradients after update
        zero_gradients();
    }

    void zero_gradients() {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        kernel_gradient.at(oc, ic, kh, kw) = 0.0f;
                    }
                }
            }
            bias_gradient.at(oc, 0, 0, 0) = 0.0f;
        }
    }

    void saveParameters(ofstream& file) const {
        file.write(reinterpret_cast<const char*>(&in_channels), sizeof(in_channels));
        file.write(reinterpret_cast<const char*>(&out_channels), sizeof(out_channels));
        file.write(reinterpret_cast<const char*>(&kernel_size), sizeof(kernel_size));
        file.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
        file.write(reinterpret_cast<const char*>(&padding), sizeof(padding));

        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        float weight = kernel.at(oc, ic, kh, kw);
                        file.write(reinterpret_cast<const char*>(&weight), sizeof(float));
                    }
                }
            }
        }

        for (size_t oc = 0; oc < out_channels; ++oc) {
            float bias_val = bias.at(oc, 0, 0, 0);
            file.write(reinterpret_cast<const char*>(&bias_val), sizeof(float));
        }
    }

    void loadParameters(ifstream& file) {
        file.read(reinterpret_cast<char*>(&in_channels), sizeof(in_channels));
        file.read(reinterpret_cast<char*>(&out_channels), sizeof(out_channels));
        file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
        file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
        file.read(reinterpret_cast<char*>(&padding), sizeof(padding));

        kernel = Matrix(out_channels, in_channels, kernel_size, kernel_size);
        bias = Matrix(out_channels, 1, 1, 1);

        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        float weight;
                        file.read(reinterpret_cast<char*>(&weight), sizeof(float));
                        kernel.at(oc, ic, kh, kw) = weight;
                    }
                }
            }
        }

        for (size_t oc = 0; oc < out_channels; ++oc) {
            float bias_val;
            file.read(reinterpret_cast<char*>(&bias_val), sizeof(float));
            bias.at(oc, 0, 0, 0) = bias_val;
        }
    }

    size_t get_in_channels() const { return in_channels; }
    size_t get_out_channels() const { return out_channels; }
    size_t get_kernel_size() const { return kernel_size; }
    size_t get_stride() const { return stride; }
    size_t get_padding() const { return padding; }
    const Matrix& get_kernel() const { return kernel; }
    const Matrix& get_bias() const { return bias; }
    void set_kernel(const Matrix& k) { kernel = k; }
    void set_bias(const Matrix& b) { bias = b; }
};