#include "kseq/kseq.h"
#include "common.h"
#include <iostream>

#define SMP_ALIGN 1024
#define SMP_ALIGN_SHIFT 10

#define SIG_ALIGN 256
#define SIG_ALIGN_SHIFT 8

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

/*
    Naive string matching implementation O(nm) per pair
    Optimization plan:
    1. Multiple threads per sample and signature pair, break sample string into substrings each substring overlaps with the next by len(signature) - 1 characters
    2. Convert to KMP, prebuilt the prefix table for each signature
    Tasks:
    1. Convert Phred from ascii to numeric score + integration hash per sample - 1 thread per sample (can be scaled to use multiple threads each handle substr, no overlap)
    2. Perform matching for sample and signature pair - 1 thread per pair (see above to scale)
    3. Memory optimization (what can be moved to shared memory)
    4. ^Optimization plan
*/
template <typename T>
void check_cuda_error(T err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

__global__ void processSamples(char *d_smps_qual, float *d_smps_phred, int *d_smps_phash, int smps_seq_sz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = idx << SMP_ALIGN_SHIFT;
    if (offset >= smps_seq_sz)
        return;
    int phash = 0;
    for (size_t i = 0; i < SMP_ALIGN; i++)
    {
        int phred = d_smps_qual[offset + i] - 33;
        d_smps_phred[offset + i] = phred;
        phash = (phred + phash) % 97;
    }
    d_smps_phash[idx] = phash;
}

__global__ void genHash(int *d_smps_phash, int *d_smps_hash, int *d_smps_sz, int *d_smps_offset, int smpSz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= smpSz)
        return;
    int hash = 0;
    size_t cc = d_smps_sz[idx] >> SMP_ALIGN_SHIFT;
    cc += (d_smps_sz[idx] & (SMP_ALIGN - 1)) > 0;
    size_t offset = d_smps_offset[idx] >> SMP_ALIGN_SHIFT;
    for (size_t i = 0; i < cc; i++)
        hash = (d_smps_phash[i + offset] + hash) % 97;
    d_smps_hash[idx] = hash;
}

// Similar to process samples make the loop size identical, by breaking into uniform block
__device__ double match(char *smp_seq, char *sig_seq, float *smp_phred, int smp_len, int sig_len)
{
    float hcf = -1.0;
    bool hasMatch = false;
    for (int i = 0; i <= smp_len - sig_len; i++)
    {
        float crf = 0;
        bool matched = true;
        // TEST: can compare if else vs math manip
        for (int j = 0; j < sig_len; j++)
        {
            if (smp_seq[i + j] != sig_seq[j] && sig_seq[j] != 'N' && smp_seq[i + j] != 'N')
            {
                // printf("matching failed at %d = %c, %d = %c", i + j, smp_seq[i + j], j, sig_seq[j]);
                crf = -1.0;
                matched = false;
                break;
            }
            crf += smp_phred[i + j];
        }

        hasMatch = hasMatch || matched;
        if (matched)
            hcf = fmaxf(hcf, crf);
    }

    return hasMatch ? (double)hcf / sig_len : -1.0f;
}

__global__ void findMatches(float *d_smps_phred, char *d_smps_seq, char *d_sigs_seq, double *d_cfs, int *d_smps_offset, int *d_sigs_offset, int *d_smps_sz, int *d_sigs_sz, int sigSz, int smpSz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= smpSz * sigSz)
        return;
    int smp_idx = idx / sigSz, sig_idx = idx % sigSz;

    int smp_offset = d_smps_offset[smp_idx], sig_offset = d_sigs_offset[sig_idx];
    int smp_len = d_smps_sz[smp_idx], sig_len = d_sigs_sz[sig_idx];

    char *smp_seq = d_smps_seq + smp_offset, *sig_seq = d_sigs_seq + sig_offset;
    float *smp_phred = d_smps_phred + smp_offset;
    double cf = match(smp_seq, sig_seq, smp_phred, smp_len, sig_len);
    d_cfs[idx] = cf;
}

void runMatcher(const std::vector<klibpp::KSeq> &samples, const std::vector<klibpp::KSeq> &signatures, std::vector<MatchResult> &matches)
{
    // TEST: Attempt to reduce warp divergence by sorting the samples and signatures by size
    // std::sort(samples.begin(), samples.end(), [](const klibpp::KSeq &a, const klibpp::KSeq &b)
    //           { return a.seq.size() > b.seq.size(); });
    // std::sort(signatures.begin(), signatures.end(), [](const klibpp::KSeq &a, const klibpp::KSeq &b)
    //           { return a.seq.size() > b.seq.size(); });
    const int smpSz = samples.size(), sigSz = signatures.size();

    int *d_smps_offset, *d_sigs_offset, *d_smps_sz, *d_sigs_sz;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_smps_offset, sizeof(int) * smpSz));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sigs_offset, sizeof(int) * sigSz));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_smps_sz, sizeof(int) * smpSz));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sigs_sz, sizeof(int) * sigSz));

    size_t currOffset = 0, smps_seq_sz = 0, sigs_seq_sz = 0;
    int h_smp_offsets[smpSz], h_sig_offsets[sigSz], h_smp_szs[smpSz], h_sig_szs[sigSz];
    for (int i = 0; i < smpSz; i++)
    {
        auto &smp = samples[i];
        h_smp_offsets[i] = currOffset;
        h_smp_szs[i] = smp.seq.size();
        currOffset += smp.seq.size();
        currOffset = (currOffset + SMP_ALIGN - 1) & ~(SMP_ALIGN - 1);
    }
    smps_seq_sz = currOffset;
    currOffset = 0;
    for (int i = 0; i < sigSz; i++)
    {
        auto &sig = signatures[i];
        h_sig_offsets[i] = currOffset;
        h_sig_szs[i] = sig.seq.size();
        currOffset += sig.seq.size();
        currOffset = (currOffset + SIG_ALIGN - 1) & ~(SIG_ALIGN - 1);
    }
    sigs_seq_sz = currOffset;
    // std::cout << "smps seq size " << smps_seq_sz << " sigs seq size " << sigs_seq_sz << '\n';
    // for (int e : h_smp_offsets)
    //     std::cout << e << ' ';
    // std::cout << '\n';
    // for (int e : h_smp_szs)
    //     std::cout << e << ' ';
    // std::cout << '\n';
    // for (int e : h_sig_offsets)
    //     std::cout << e << ' ';
    // std::cout << '\n';
    // for (int e : h_sig_szs)
    //     std::cout << e << ' ';
    // std::cout << '\n';

    CHECK_CUDA_ERROR(cudaMemcpy(d_smps_offset, h_smp_offsets, sizeof(int) * smpSz, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sigs_offset, h_sig_offsets, sizeof(int) * sigSz, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_smps_sz, h_smp_szs, sizeof(int) * smpSz, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sigs_sz, h_sig_szs, sizeof(int) * sigSz, cudaMemcpyHostToDevice));

    char *d_smps_seq, *d_sigs_seq, *d_smps_qual;
    float *d_smps_phred;
    int *d_smps_phash, *d_smps_hash;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_smps_seq, smps_seq_sz));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sigs_seq, sigs_seq_sz));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_smps_qual, smps_seq_sz));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_smps_phred, sizeof(float) * smps_seq_sz));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_smps_hash, sizeof(int) * smpSz));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_smps_phash, sizeof(int) * (smps_seq_sz >> SMP_ALIGN_SHIFT)));

    CHECK_CUDA_ERROR(cudaMemset(d_smps_qual, 33, smps_seq_sz));

    for (int i = 0; i < smpSz; i++)
    {
        auto &smp = samples[i];
        CHECK_CUDA_ERROR(cudaMemcpy(d_smps_seq + h_smp_offsets[i], smp.seq.c_str(), smp.seq.size(), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_smps_qual + h_smp_offsets[i], smp.qual.c_str(), smp.qual.size(), cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < sigSz; i++)
        CHECK_CUDA_ERROR(cudaMemcpy(d_sigs_seq + h_sig_offsets[i], signatures[i].seq.c_str(), signatures[i].seq.size(), cudaMemcpyHostToDevice));

    const int tpb = 256;
    int ttts = smps_seq_sz >> SMP_ALIGN_SHIFT;
    int bpg = (ttts + tpb - 1) / tpb;

    processSamples<<<bpg, tpb>>>(d_smps_qual, d_smps_phred, d_smps_phash, smps_seq_sz);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    bpg = (smpSz + tpb - 1) / tpb;
    genHash<<<bpg, tpb>>>(d_smps_phash, d_smps_hash, d_smps_sz, d_smps_offset, smpSz);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // for (int i = 0; i < smpSz; i++)
    //     std::cout << h_smps_hash[i] << ' ';
    // std::cout << '\n';

    double *d_cfs;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cfs, sizeof(double) * smpSz * sigSz));
    CHECK_CUDA_ERROR(cudaMemset(d_cfs, 0, sizeof(double) * smpSz * sigSz));

    const int ttps = smpSz * sigSz;
    bpg = (ttps + tpb - 1) / tpb;
    findMatches<<<bpg, tpb>>>(
        d_smps_phred, d_smps_seq, d_sigs_seq,
        d_cfs, d_smps_offset, d_sigs_offset, d_smps_sz, d_sigs_sz,
        sigSz, smpSz);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    double *h_cfs = new double[smpSz * sigSz];
    CHECK_CUDA_ERROR(cudaMemcpy(h_cfs, d_cfs, sizeof(double) * smpSz * sigSz, cudaMemcpyDeviceToHost));
    int h_smps_hash[smpSz];
    CHECK_CUDA_ERROR(cudaMemcpy(h_smps_hash, d_smps_hash, sizeof(int) * smpSz, cudaMemcpyDeviceToHost));

    for (int i = 0; i < smpSz * sigSz; i++)
    {
        int smp_idx = i / sigSz, sig_idx = i % sigSz;
        if (h_cfs[i] >= 0)
            matches.push_back({samples[smp_idx].name, signatures[sig_idx].name, h_cfs[i], h_smps_hash[smp_idx]});
    }

    cudaFree(d_smps_offset);
    cudaFree(d_sigs_offset);
    cudaFree(d_smps_sz);
    cudaFree(d_sigs_sz);

    cudaFree(d_smps_seq);
    cudaFree(d_sigs_seq);
    cudaFree(d_smps_qual);
    cudaFree(d_smps_phred);

    cudaFree(d_cfs);
    cudaFree(d_smps_phash);
    cudaFree(d_smps_hash);
}
