#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

struct Sample {
    char* d_d;
    char* d_q;
    int* d_p;
    int hash;
    int len;
};

struct Sig {
    char* d_d;
    int len;
};

struct ResDev {
    int s_idx;
    int sg_idx;
    double conf;
    int hash;
};

__global__ void preproc(Sample* d_s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Sample s = d_s[i];
    int sum = 0;

    for (int j = 0; j < s.len; ++j) {
        int p = s.d_q[j] - 33;
        s.d_p[j] = p;
        sum += p;
    }

    d_s[i].hash = sum % 97;
}

__device__ bool match(const char* s_d, const char* sg_d, int l, int p) {
    for (int j = 0; j < l; ++j) {
        char sc = s_d[p + j];
        char vc = sg_d[j];
        bool bothNotN = (sc != 'N' && vc != 'N');
        bool charsDiffer = (sc != vc);
    
        if (bothNotN && charsDiffer) {
            return false;
        }
    }
    return true;
}

__global__ void process(Sample* d_s, Sig* d_sg, int ns, int n_sg,
                        ResDev* res, int* cnt) {
    extern __shared__ double sc[];
    int* sp = (int*)&sc[blockDim.x];

    int p_idx = blockIdx.x;
    int s_idx = p_idx / n_sg;
    int sg_idx = p_idx % n_sg;

    if (s_idx >= ns || sg_idx >= n_sg) return;

    Sample s = d_s[s_idx];
    Sig sg = d_sg[sg_idx];
    int l = sg.len;
    int max_p = s.len - l;
    if (max_p < 0) return;

    double bc = -1.0;
    int bp = -1;

    for (int p = threadIdx.x; p <= max_p; p += blockDim.x) {
        if (match(s.d_d, sg.d_d, l, p)) {
            double sum = 0.0;
            for (int j = 0; j < l; ++j) sum += s.d_p[p + j];
            double c = sum / l;

            bool betterCorrelation = (c > bc);
            bool tieWithBetterPosition = ((c == bc) && (p > bp));

            if (betterCorrelation || tieWithBetterPosition) {
                bc = c;
                bp = p;
            }
        }
    }

    sc[threadIdx.x] = bc;
    sp[threadIdx.x] = bp;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            bool scCondition = (sc[threadIdx.x] < sc[threadIdx.x + step]);
            bool spCondition = (sc[threadIdx.x] == sc[threadIdx.x + step]) &&
                               (sp[threadIdx.x] < sp[threadIdx.x + step]);
    
            if (scCondition || spCondition) {
                sc[threadIdx.x] = sc[threadIdx.x + step];
                sp[threadIdx.x] = sp[threadIdx.x + step];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && sp[0] != -1) {
        int idx = atomicAdd(cnt, 1);
        res[idx] = {s_idx, sg_idx, sc[0], s.hash};
    }
}

void runMatcher(const std::vector<klibpp::KSeq>& s,
                const std::vector<klibpp::KSeq>& sg,
                std::vector<MatchResult>& m) {

    size_t total_seq_len_s  = 0;
    size_t total_qual_len_s = 0;
    size_t total_p_len_s    = 0;

    for (auto& sample : s) {
        total_seq_len_s  += (sample.seq.size()  + 1);
        total_qual_len_s += (sample.qual.size() + 1);
        total_p_len_s    += sample.seq.size();
    }

    // Allocate those big buffers on device
    char* d_allSeq_s  = nullptr;
    char* d_allQual_s = nullptr;
    int*  d_allP_s    = nullptr;

    CHECK_CUDA(cudaMalloc(&d_allSeq_s,  total_seq_len_s  * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&d_allQual_s, total_qual_len_s * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&d_allP_s,    total_p_len_s    * sizeof(int)));

    size_t seq_offset_s  = 0;
    size_t qual_offset_s = 0;
    size_t p_offset_s    = 0;

    // Prepare array of Sample structs on host
    std::vector<Sample> h_s(s.size());

    for (size_t i = 0; i < s.size(); ++i) {
        const auto& seqStr  = s[i].seq;
        const auto& qualStr = s[i].qual;
        size_t lenSeq  = seqStr.size();
        size_t lenQual = qualStr.size();

        // "Pointers" in the big device buffers for this sample
        h_s[i].d_d = d_allSeq_s  + seq_offset_s;
        h_s[i].d_q = d_allQual_s + qual_offset_s;
        h_s[i].d_p = d_allP_s    + p_offset_s;

        // Copy the sequence
        CHECK_CUDA(cudaMemcpy(h_s[i].d_d, seqStr.c_str(), lenSeq + 1, cudaMemcpyHostToDevice));

        // Copy the quality
        CHECK_CUDA(cudaMemcpy(h_s[i].d_q, qualStr.c_str(), lenQual + 1, cudaMemcpyHostToDevice));

        // We do NOT copy p[] yet – that will be computed in preproc kernel.
        // Just advance offsets
        seq_offset_s  += (lenSeq  + 1);
        qual_offset_s += (lenQual + 1);
        p_offset_s    += lenSeq;

        h_s[i].len = static_cast<int>(lenSeq);
        h_s[i].hash = 0;
    }


    size_t total_seq_len_sg = 0;
    for (auto& signature : sg) {
        total_seq_len_sg += (signature.seq.size() + 1);
    }

    char* d_allSeq_sg = nullptr;
    CHECK_CUDA(cudaMalloc(&d_allSeq_sg, total_seq_len_sg * sizeof(char)));

    size_t seq_offset_sg = 0;
    std::vector<Sig> h_sg(sg.size());
    for (size_t i = 0; i < sg.size(); ++i) {
        const auto& seqStr = sg[i].seq;
        size_t lenSeq = seqStr.size();

        // This Sig's pointer goes into the big signature buffer
        h_sg[i].d_d = d_allSeq_sg + seq_offset_sg;
        CHECK_CUDA(cudaMemcpy(h_sg[i].d_d, seqStr.c_str(), lenSeq + 1, cudaMemcpyHostToDevice));

        seq_offset_sg += (lenSeq + 1);
        h_sg[i].len    = static_cast<int>(lenSeq);
    }


    Sample* d_s = nullptr;
    CHECK_CUDA(cudaMalloc(&d_s, s.size() * sizeof(Sample)));
    CHECK_CUDA(cudaMemcpy(d_s, h_s.data(), s.size() * sizeof(Sample), cudaMemcpyHostToDevice));

    Sig* d_sg = nullptr;
    CHECK_CUDA(cudaMalloc(&d_sg, sg.size() * sizeof(Sig)));
    CHECK_CUDA(cudaMemcpy(d_sg, h_sg.data(), sg.size() * sizeof(Sig), cudaMemcpyHostToDevice));


    {
        dim3 block(256);
        dim3 grid((unsigned int)((s.size() + block.x - 1) / block.x));
        preproc<<<grid, block>>>(d_s, (int)s.size());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    ResDev* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, s.size() * sg.size() * sizeof(ResDev)));

    int* d_cnt = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cnt, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_cnt, 0, sizeof(int)));

    int blk   = 256;
    int pairs = (int)(s.size() * sg.size());
    size_t sh = blk * (sizeof(double) + sizeof(int));

    process<<<pairs, blk, sh>>>(d_s, d_sg, (int)s.size(), (int)sg.size(), d_res, d_cnt);
    CHECK_CUDA(cudaDeviceSynchronize());

    int cnt = 0;
    CHECK_CUDA(cudaMemcpy(&cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<ResDev> h_res(cnt);
    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, cnt * sizeof(ResDev), cudaMemcpyDeviceToHost));

    m.reserve(m.size() + cnt);
    for (auto& r : h_res) {
        m.push_back({
            s[r.s_idx].name,
            sg[r.sg_idx].name,
            r.conf,
            r.hash
        });
    }

    cudaFree(d_allSeq_s);
    cudaFree(d_allQual_s);
    cudaFree(d_allP_s);
    cudaFree(d_s);
    cudaFree(d_allSeq_sg);
    cudaFree(d_sg);
    cudaFree(d_res);
    cudaFree(d_cnt);
}
