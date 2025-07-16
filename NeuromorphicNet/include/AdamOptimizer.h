#pragma once
#include <vector>
#include <cmath>

using namespace std;

class AdamOptimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    
public:
    AdamOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f) 
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps) {}
    
    // Update parameters using Adam algorithm
    void update(vector<float>& params, const vector<float>& gradients, 
                vector<float>& m, vector<float>& v, int time_step) {
        
        for (size_t i = 0; i < params.size(); i++) {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0f - beta1) * gradients[i];
            
            // Update biased second moment estimate  
            v[i] = beta2 * v[i] + (1.0f - beta2) * gradients[i] * gradients[i];
            
            // Compute bias-corrected first moment estimate
            float m_hat = m[i] / (1.0f - pow(beta1, time_step));
            
            // Compute bias-corrected second moment estimate
            float v_hat = v[i] / (1.0f - pow(beta2, time_step));
            
            // Update parameters
            params[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
    
    // Single parameter update
    void update(float& param, float gradient, float& m, float& v, int time_step) {
        // Update biased first moment estimate
        m = beta1 * m + (1.0f - beta1) * gradient;
        
        // Update biased second moment estimate
        v = beta2 * v + (1.0f - beta2) * gradient * gradient;
        
        // Compute bias-corrected first moment estimate
        float m_hat = m / (1.0f - pow(beta1, time_step));
        
        // Compute bias-corrected second moment estimate
        float v_hat = v / (1.0f - pow(beta2, time_step));
        
        // Update parameter
        param -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
    
    // Getters and setters
    float getLearningRate() const { return learning_rate; }
    void setLearningRate(float lr) { learning_rate = lr; }
    
    float getBeta1() const { return beta1; }
    void setBeta1(float b1) { beta1 = b1; }
    
    float getBeta2() const { return beta2; }
    void setBeta2(float b2) { beta2 = b2; }
    
    float getEpsilon() const { return epsilon; }
    void setEpsilon(float eps) { epsilon = eps; }
};