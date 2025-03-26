#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
template <typename T>
void check_cuda_error(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

struct GPUSample {
    char* dna;       // Device pointer
    char* qual;      // Device pointer
    int* phred;      // Device pointer
    int integrity_hash;
    int length;
};

struct GPUSignature {
    char* dna;       // Device pointer
    int length;
};

struct MatchResultDevice {
    int sample_idx;
    int sig_idx;
    double confidence;
    int integrity_hash;
};

__global__ void preprocess_samples(GPUSample* d_samples, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    GPUSample sample = d_samples[idx];
    int sum = 0;

    for (int i = 0; i < sample.length; ++i) {
        int phred = sample.qual[i] - 33;  // Access device memory
        sample.phred[i] = phred;
        sum += phred;
    }

    d_samples[idx].integrity_hash = sum % 97;
}

__device__ bool is_match(const char* sample_dna, const char* sig_dna, int sig_len, int pos) {
    for (int j = 0; j < sig_len; ++j) {
        char s_char = sample_dna[pos + j];
        char v_char = sig_dna[j];
        if (s_char != 'N' && v_char != 'N' && s_char != v_char) {
            return false;
        }
    }
    return true;
}

__global__ void process_pairs(
    GPUSample* d_samples, GPUSignature* d_sigs, 
    int num_samples, int num_sigs, 
    MatchResultDevice* d_results, int* d_count
) {
    extern __shared__ double s_data[];
    double* s_conf = (double*)s_data;
    int* s_pos = (int*)&s_conf[blockDim.x];

    // Each block handles 1 pair
    int pair_idx = blockIdx.x;
    int sample_idx = pair_idx / num_sigs;
    int sig_idx = pair_idx % num_sigs;

    if (sample_idx >= num_samples || sig_idx >= num_sigs) return;

    GPUSample sample = d_samples[sample_idx];
    GPUSignature sig = d_sigs[sig_idx];
    int sig_len = sig.length;
    int max_pos = sample.length - sig_len;
    if (max_pos < 0) return;

    double best_conf = -1.0;
    int best_pos = -1;

    // Each thread checks positions in a strided loop
    for (int pos = threadIdx.x; pos <= max_pos; pos += blockDim.x) {
        if (is_match(sample.dna, sig.dna, sig_len, pos)) {
            double sum = 0.0;
            for (int j = 0; j < sig_len; ++j) {
                sum += sample.phred[pos + j];
            }
            double conf = sum / sig_len;

            if (conf > best_conf || (conf == best_conf && pos > best_pos)) {
                best_conf = conf;
                best_pos = pos;
            }
        }
    }

    s_conf[threadIdx.x] = best_conf;
    s_pos[threadIdx.x] = best_pos;
    __syncthreads();

    // Reduction within the block for this pair
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_conf[threadIdx.x] < s_conf[threadIdx.x + stride] ||
                (s_conf[threadIdx.x] == s_conf[threadIdx.x + stride] && 
                 s_pos[threadIdx.x] < s_pos[threadIdx.x + stride])) {
                s_conf[threadIdx.x] = s_conf[threadIdx.x + stride];
                s_pos[threadIdx.x] = s_pos[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && s_pos[0] != -1) {
        int idx = atomicAdd(d_count, 1);
        d_results[idx] = {sample_idx, sig_idx, s_conf[0], sample.integrity_hash};
    }
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, 
                const std::vector<klibpp::KSeq>& signatures, 
                std::vector<MatchResult>& matches) 
{
    std::vector<GPUSample> h_samples(samples.size());
    std::vector<GPUSignature> h_sigs(signatures.size());

    // Allocate device memory for DNA, qual, and phred scores
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        CHECK_CUDA_ERROR(cudaMalloc(&h_samples[i].dna, s.seq.size() + 1));
        CHECK_CUDA_ERROR(cudaMemcpy(h_samples[i].dna, s.seq.c_str(), s.seq.size() + 1, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMalloc(&h_samples[i].qual, s.qual.size() + 1));
        CHECK_CUDA_ERROR(cudaMemcpy(h_samples[i].qual, s.qual.c_str(), s.qual.size() + 1, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMalloc(&h_samples[i].phred, s.seq.size() * sizeof(int)));
        h_samples[i].length = static_cast<int>(s.seq.size());
    }

    for (size_t i = 0; i < signatures.size(); ++i) {
        const auto& sig = signatures[i];
        CHECK_CUDA_ERROR(cudaMalloc(&h_sigs[i].dna, sig.seq.size() + 1));
        CHECK_CUDA_ERROR(cudaMemcpy(h_sigs[i].dna, sig.seq.c_str(), sig.seq.size() + 1, cudaMemcpyHostToDevice));
        h_sigs[i].length = static_cast<int>(sig.seq.size());
    }

    // Now allocate GPU arrays for GPUSample and GPUSignature structs
    GPUSample* d_samples = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_samples, samples.size() * sizeof(GPUSample)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_samples, h_samples.data(), 
                                samples.size() * sizeof(GPUSample), 
                                cudaMemcpyHostToDevice));

    GPUSignature* d_sigs = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_sigs, signatures.size() * sizeof(GPUSignature)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sigs, h_sigs.data(), 
                                signatures.size() * sizeof(GPUSignature), 
                                cudaMemcpyHostToDevice));

    // Preprocess samples to compute Phred scores and integrity hash
    preprocess_samples<<<(samples.size() + 255) / 256, 256>>>(d_samples, samples.size());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Allocate memory for results on device
    MatchResultDevice* d_results = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, samples.size() * signatures.size() * sizeof(MatchResultDevice)));

    int* d_count = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_count, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_count, 0, sizeof(int)));

    // Calculate kernel dimensions
    int block_size = 256;
    int num_pairs = static_cast<int>(samples.size() * signatures.size());
    int grid_size = num_pairs;
    size_t shared_mem_size = block_size * (sizeof(double) + sizeof(int));

    // Launch the pairing kernel
    process_pairs<<<grid_size, block_size, shared_mem_size>>>(
        d_samples,
        d_sigs,
        static_cast<int>(samples.size()),
        static_cast<int>(signatures.size()),
        d_results,
        d_count
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the count back to host
    int count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    // Copy results back to host
    std::vector<MatchResultDevice> h_results(count);
    CHECK_CUDA_ERROR(cudaMemcpy(h_results.data(), d_results, 
                                count * sizeof(MatchResultDevice), 
                                cudaMemcpyDeviceToHost));

    // Populate matches vector
    for (const auto& res : h_results) {
        matches.push_back({
            samples[res.sample_idx].name,
            signatures[res.sig_idx].name,
            res.confidence,
            res.integrity_hash
        });
    }

    // Free device memory
    for (auto& sample : h_samples) {
        cudaFree(sample.dna);
        cudaFree(sample.qual);
        cudaFree(sample.phred);
    }
    for (auto& sig : h_sigs) {
        cudaFree(sig.dna);
    }

    cudaFree(d_samples);
    cudaFree(d_sigs);
    cudaFree(d_results);
    cudaFree(d_count);
}
