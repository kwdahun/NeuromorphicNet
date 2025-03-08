#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <cstdint>

// Abstract base class for spike generators
class SpikeGenerator {
public:
    SpikeGenerator(double duration = 0.1, double dt = 0.001, double f_max = 100.0)
        : duration_(duration)
        , dt_(dt)
        , f_max_(f_max)
        , time_steps_(static_cast<int>(duration / dt)) {
        // Initialize random number generator with time-based seed
        rng_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }
    SpikeGenerator(int time_steps)
        : time_steps_(time_steps)
        , duration_(0)
        , dt_(0)
        , f_max_(0)
    {}

    virtual ~SpikeGenerator() = default;

    // Pure virtual method that must be implemented by derived classes
    virtual std::vector<std::vector<int>> generateSpikes(const std::vector<uint8_t>& input) = 0;

    // Getter
    int getTimeSteps() const { return time_steps_; }
    double getDuration() const { return duration_; }
    double getDt() const { return dt_; }
    double getFMax() const { return f_max_; }

protected:
    // Spike generation parameters
    double duration_;
    double dt_;
    double f_max_;
    int time_steps_;

    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
};