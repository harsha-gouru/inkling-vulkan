// Stage 1: Clean, readable CPU reference for Inkling-2-Mini (Inkling2Loop).
// Zero external dependencies; reads ~/models/Inkling-2-Mini/model-00001-of-00001.safetensors via mmap.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static constexpr int D_MODEL     = 2304;
static constexpr int N_PHYS      = 24;
static constexpr int CORE_START  = 6;
static constexpr int CORE_END    = 18;
static constexpr int N_PASSES    = 3;
static constexpr int N_SLOTS     = N_PHYS + (N_PASSES - 1) * (CORE_END - CORE_START); // 48
static constexpr int N_HEADS     = 24;
static constexpr int N_KV_HEADS  = 8;
static constexpr int HEAD_DIM    = 128;
static constexpr int Q_DIM       = N_HEADS * HEAD_DIM;     // 3072
static constexpr int KV_DIM      = N_KV_HEADS * HEAD_DIM;  // 1024
static constexpr int D_REL       = 16;
static constexpr int REL_EXTENT  = 1024;
static constexpr int SWA_WINDOW  = 1024;
static constexpr int SCONV_K     = 4;
static constexpr int SCONV_HIST  = SCONV_K - 1;            // 3
static constexpr int BLOCK_SIZE  = 6;
static constexpr int GATE_RANK   = 64;
static constexpr int N_EXPERTS   = 128;
static constexpr int TOP_K       = 8;
static constexpr int INTER_SIZE  = 1152;
static constexpr int VOCAB_ACT   = 200058;

static constexpr float RMS_EPS         = 1e-6f;
static constexpr float QK_CLIP         = 22.627416997969522f;
static constexpr float REL_CLIP        = 8.0f;
static constexpr float LOG_SCALE_FLOOR = 8192.0f;
static constexpr float LOG_SCALE_ALPHA = 0.1f;
static constexpr float SITU_BETA       = 4.0f;
static constexpr float SITU_LIN_BETA   = 25.0f;
static constexpr float ROUTE_SCALE     = 8.0f;
static constexpr float LOGIT_SOFTCAP   = 30.0f;

static inline bool is_global_layer(int ip) {
    return ip == 5 || ip == 11 || ip == 17 || ip == 23;
}

static float E4M3_LUT[256];
static constexpr float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};
static float UE8M0_LUT[256];

static void init_luts() {
    for (int code = 0; code < 256; ++code) {
        int sign = (code & 0x80) ? -1 : 1;
        int exp  = (code & 0x7F) >> 3;
        int man  = (code & 0x07);
        float v;
        if ((code & 0x7F) == 0x7F) v = NAN;
        else if (exp == 0) v = std::ldexp((float)man, -9);
        else v = std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
        E4M3_LUT[code] = sign * v;
        UE8M0_LUT[code] = (code == 255) ? NAN : std::ldexp(1.0f, code - 127);
    }
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t off0 = 0, off1 = 0;
};

struct Safetensors {
    int fd = -1;
    size_t file_size = 0;
    const uint8_t* base = nullptr;
    const uint8_t* data = nullptr;
    std::map<std::string, TensorInfo> tensors;

    void open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { perror("open safetensors"); std::exit(1); }
        struct stat st{};
        fstat(fd, &st);
        file_size = (size_t)st.st_size;
        void* p = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) { perror("mmap"); std::exit(1); }
        base = (const uint8_t*)p;
        uint64_t hlen = 0;
        std::memcpy(&hlen, base, 8);
        data = base + 8 + hlen;
        std::string hdr((const char*)(base + 8), hlen);
        parse_header(hdr);
    }

    ~Safetensors() {
        if (base) munmap((void*)base, file_size);
        if (fd >= 0) ::close(fd);
    }

    const void* ptr(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            std::cerr << "Missing tensor: " << name << "\n";
            std::exit(1);
        }
        return data + it->second.off0;
    }

    const uint16_t* bf16(const std::string& name) const { return (const uint16_t*)ptr(name); }
    const float*    f32 (const std::string& name) const { return (const float*)ptr(name); }
    const uint8_t*  u8  (const std::string& name) const { return (const uint8_t*)ptr(name); }

    void parse_header(const std::string& s) {
        size_t pos = 0;
        auto skip_ws = [&]() { while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) ++pos; };
        auto parse_str = [&]() -> std::string {
            skip_ws();
            if (s[pos++] != '"') std::exit(1);
            std::string out;
            while (pos < s.size() && s[pos] != '"') {
                if (s[pos] == '\\') ++pos;
                out.push_back(s[pos++]);
            }
            ++pos;
            return out;
        };
        auto parse_int = [&]() -> int64_t {
            skip_ws();
            size_t start = pos;
            if (s[pos] == '-') ++pos;
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
            return std::stoll(s.substr(start, pos - start));
        };
        auto skip_obj = [&]() {
            int depth = 0;
            bool in_str = false;
            do {
                char c = s[pos++];
                if (c == '"' && s[pos - 2] != '\\') in_str = !in_str;
                if (!in_str) {
                    if (c == '{') ++depth;
                    else if (c == '}') --depth;
                }
            } while (depth > 0 && pos < s.size());
        };
        skip_ws();
        if (s[pos++] != '{') std::exit(1);
        while (pos < s.size()) {
            skip_ws();
            if (s[pos] == '}') break;
            std::string key = parse_str();
            skip_ws(); ++pos;
            skip_ws();
            if (key == "__metadata__") {
                skip_obj();
            } else {
                TensorInfo ti;
                ++pos;
                while (pos < s.size()) {
                    skip_ws();
                    if (s[pos] == '}') { ++pos; break; }
                    std::string fk = parse_str();
                    skip_ws(); ++pos;
                    skip_ws();
                    if (fk == "dtype") {
                        ti.dtype = parse_str();
                    } else if (fk == "shape") {
                        ++pos;
                        while (true) {
                            skip_ws();
                            if (s[pos] == ']') { ++pos; break; }
                            ti.shape.push_back(parse_int());
                            skip_ws();
                            if (s[pos] == ',') ++pos;
                        }
                    } else if (fk == "data_offsets") {
                        ++pos;
                        ti.off0 = (uint64_t)parse_int();
                        skip_ws(); if (s[pos] == ',') ++pos;
                        ti.off1 = (uint64_t)parse_int();
                        skip_ws(); if (s[pos] == ']') ++pos;
                    }
                    skip_ws();
                    if (s[pos] == ',') ++pos;
                }
                tensors[key] = std::move(ti);
            }
            skip_ws();
            if (s[pos] == ',') ++pos;
        }
    }
};

static void rms_norm(const float* x, const uint16_t* w_bf16, float* out, int n, float eps = RMS_EPS) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    float inv = 1.0f / std::sqrt((float)(ss / n) + eps);
    if (w_bf16) {
        for (int i = 0; i < n; ++i) out[i] = x[i] * inv * bf16_to_f32(w_bf16[i]);
    } else {
        for (int i = 0; i < n; ++i) out[i] = x[i] * inv;
    }
}

static void matvec_bf16(const uint16_t* W, const float* x, float* y, int rows, int cols) {
    #pragma omp parallel for schedule(static) if(rows >= 256)
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = W + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) acc += bf16_to_f32(wr[c]) * x[c];
        y[r] = acc;
    }
}

static void l2_clip_heads(float* x, const uint16_t* scale_bf16, int n_heads, int dim, float threshold) {
    for (int h = 0; h < n_heads; ++h) {
        float* v = x + h * dim;
        if (scale_bf16) {
            const uint16_t* s = scale_bf16 + h * dim;
            for (int i = 0; i < dim; ++i) v[i] *= bf16_to_f32(s[i]);
        }
        double ss = 0.0;
        for (int i = 0; i < dim; ++i) ss += (double)v[i] * (double)v[i];
        float norm = std::sqrt((float)ss) + 1e-6f;
        float factor = std::min(1.0f, threshold / norm);
        for (int i = 0; i < dim; ++i) v[i] *= factor;
    }
}

static std::vector<float> attn_res_mix(const std::vector<std::vector<float>>& blocks, const std::vector<float>& cur,
                                       const float* queries, int n_queries) {
    int m = (int)blocks.size() + 1;
    std::vector<std::vector<float>> normed(m, std::vector<float>(D_MODEL));
    for (int k = 0; k < (int)blocks.size(); ++k) rms_norm(blocks[k].data(), nullptr, normed[k].data(), D_MODEL);
    rms_norm(cur.data(), nullptr, normed[m - 1].data(), D_MODEL);

    const float inv_sqrt_d = 1.0f / std::sqrt((float)D_MODEL);
    std::vector<float> out(n_queries * D_MODEL, 0.0f);
    for (int qi = 0; qi < n_queries; ++qi) {
        const float* q = queries + qi * D_MODEL;
        std::vector<float> scores(m);
        float mx = -1e30f;
        for (int k = 0; k < m; ++k) {
            float dot = 0.0f;
            for (int i = 0; i < D_MODEL; ++i) dot += normed[k][i] * q[i];
            scores[k] = dot * inv_sqrt_d;
            mx = std::max(mx, scores[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < m; ++k) { scores[k] = std::exp(scores[k] - mx); sum += scores[k]; }
        float* dst = out.data() + qi * D_MODEL;
        for (int k = 0; k < m; ++k) {
            float w = scores[k] / sum;
            for (int i = 0; i < D_MODEL; ++i) dst[i] += w * normed[k][i];
        }
    }
    return out;
}

static void loop_control(const Safetensors& st, int ip, const char* cname,
                         const float* cur, const float* msg, float* out) {
    std::string p = "model.llm.layers." + std::to_string(ip) + ".controllers." + cname + ".";
    std::vector<float> u(2 * D_MODEL);
    rms_norm(cur, nullptr, u.data(), D_MODEL);
    rms_norm(msg, nullptr, u.data() + D_MODEL, D_MODEL);
    float gate[GATE_RANK];
    matvec_bf16(st.bf16(p + "down_weight"), u.data(), gate, GATE_RANK, 2 * D_MODEL);
    float gain[D_MODEL], forget[D_MODEL];
    matvec_bf16(st.bf16(p + "gain_weight"), gate, gain, D_MODEL, GATE_RANK);
    matvec_bf16(st.bf16(p + "forget_weight"), gate, forget, D_MODEL, GATE_RANK);
    for (int i = 0; i < D_MODEL; ++i) {
        float g = std::tanh(gain[i]);
        float f = std::tanh(forget[i]);
        out[i] = (cur[i] - f * cur[i]) + g * msg[i];
    }
}

struct SlotState {
    std::vector<float> k_hist, v_hist, attn_hist, mlp_hist;
    std::vector<std::vector<float>> k_cache, v_cache;

    SlotState()
        : k_hist(SCONV_HIST * KV_DIM, 0.0f),
          v_hist(SCONV_HIST * KV_DIM, 0.0f),
          attn_hist(SCONV_HIST * D_MODEL, 0.0f),
          mlp_hist(SCONV_HIST * D_MODEL, 0.0f) {}
};

static void short_conv_step(const float* x_t, float* hist, int C,
                            const uint16_t* w_bf16, bool w_is_reversed_lag,
                            const float* dyn_t, float* out) {
    const float* win[4] = { hist, hist + C, hist + 2 * C, x_t };
    const float scale = 1.0f / std::sqrt((float)((dyn_t ? 2 : 1) * SCONV_K + 1));

    for (int c = 0; c < C; ++c) {
        float y = 0.0f;
        if (!w_is_reversed_lag) {
            const uint16_t* wc = w_bf16 + c * 4;
            y = bf16_to_f32(wc[0]) * win[0][c] +
                bf16_to_f32(wc[1]) * win[1][c] +
                bf16_to_f32(wc[2]) * win[2][c] +
                bf16_to_f32(wc[3]) * win[3][c];
        } else {
            y = bf16_to_f32(w_bf16[0 * C + c]) * win[3][c] +
                bf16_to_f32(w_bf16[1 * C + c]) * win[2][c] +
                bf16_to_f32(w_bf16[2 * C + c]) * win[1][c] +
                bf16_to_f32(w_bf16[3 * C + c]) * win[0][c];
        }
        if (dyn_t) {
            int h = c / HEAD_DIM;
            const float* dh = dyn_t + h * 4;
            y += dh[0] * win[3][c] + dh[1] * win[2][c] + dh[2] * win[1][c] + dh[3] * win[0][c];
        }
        out[c] = (x_t[c] + y) * scale;
    }

    std::memmove(hist, hist + C, sizeof(float) * 2 * C);
    std::memcpy(hist + 2 * C, x_t, sizeof(float) * C);
}

static void expert_matvec_nvfp4(const uint8_t* w_u8, const uint8_t* s_e4m3, float outer_scale,
                                const float* x, float* y, int expert, int rows, int cols,
                                bool interleaved_w13, float* y_up) {
    const int packed_cols = cols / 2;
    const int scale_cols  = cols / 16;
    const uint8_t* we = w_u8   + (size_t)expert * rows * packed_cols;
    const uint8_t* se = s_e4m3 + (size_t)expert * rows * scale_cols;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint8_t* wr = we + (size_t)r * packed_cols;
        const uint8_t* sr = se + (size_t)r * scale_cols;
        float acc = 0.0f;
        for (int b = 0; b < scale_cols; ++b) {
            float bs = E4M3_LUT[sr[b]] * outer_scale;
            float bacc = 0.0f;
            int c0 = b * 16;
            const uint8_t* wb = wr + b * 8;
            for (int j = 0; j < 8; ++j) {
                uint8_t byte = wb[j];
                bacc += E2M1_LUT[byte & 0x0F] * x[c0 + 2 * j]
                      + E2M1_LUT[byte >> 4]   * x[c0 + 2 * j + 1];
            }
            acc += bacc * bs;
        }
        if (interleaved_w13) {
            if ((r & 1) == 0) y[r >> 1] = acc;
            else              y_up[r >> 1] = acc;
        } else {
            y[r] = acc;
        }
    }
}

static void expert_matvec_mxfp8(const uint8_t* w_e4m3, const uint8_t* s_ue8m0,
                                const float* x, float* y, int expert, int rows, int cols,
                                bool interleaved_w13, float* y_up) {
    const int scale_cols = cols / 32;
    const uint8_t* we = w_e4m3  + (size_t)expert * rows * cols;
    const uint8_t* se = s_ue8m0 + (size_t)expert * rows * scale_cols;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint8_t* wr = we + (size_t)r * cols;
        const uint8_t* sr = se + (size_t)r * scale_cols;
        float acc = 0.0f;
        for (int b = 0; b < scale_cols; ++b) {
            float bs = UE8M0_LUT[sr[b]];
            float bacc = 0.0f;
            int c0 = b * 32;
            const uint8_t* wb = wr + c0;
            for (int j = 0; j < 32; ++j) bacc += E4M3_LUT[wb[j]] * x[c0 + j];
            acc += bacc * bs;
        }
        if (interleaved_w13) {
            if ((r & 1) == 0) y[r >> 1] = acc;
            else              y_up[r >> 1] = acc;
        } else {
            y[r] = acc;
        }
    }
}

static std::vector<float> run_layer(const Safetensors& st, int il, int ip, int pos, float tau,
                                    const std::vector<float>& cur_in,
                                    std::vector<std::vector<float>>& blocks,
                                    const std::vector<float>* messages,
                                    std::vector<SlotState>& slots,
                                    bool debug_print) {
    const bool is_glob = is_global_layer(ip);
    const std::string lp = "model.llm.layers." + std::to_string(ip) + ".";
    SlotState& slot = slots[il];

    const std::vector<float> inpSA = cur_in;
    std::vector<float> in_q = cur_in, in_k = cur_in, in_v = cur_in;

    if (ip > 0) {
        std::string qname = "model.llm.attn_res.attn_queries." + std::to_string(ip - 1);
        int nq = is_glob ? 3 : 1;
        auto mixed = attn_res_mix(blocks, cur_in, st.f32(qname), nq);
        in_q.assign(mixed.begin(), mixed.begin() + D_MODEL);
        if (is_glob) {
            in_k.assign(mixed.begin() + D_MODEL, mixed.begin() + 2 * D_MODEL);
            in_v.assign(mixed.begin() + 2 * D_MODEL, mixed.begin() + 3 * D_MODEL);
        } else {
            in_k = in_q;
            in_v = in_q;
        }
    }

    if (messages) {
        std::vector<float> tq(D_MODEL), tk(D_MODEL), tv(D_MODEL);
        loop_control(st, ip, "attn", in_q.data(), messages->data() + 0 * D_MODEL, tq.data());
        loop_control(st, ip, "k",    in_k.data(), messages->data() + 1 * D_MODEL, tk.data());
        loop_control(st, ip, "v",    in_v.data(), messages->data() + 2 * D_MODEL, tv.data());
        in_q = std::move(tq);
        in_k = std::move(tk);
        in_v = std::move(tv);
    }

    std::vector<float> nq(D_MODEL), nk(D_MODEL), nv(D_MODEL);
    if (!is_glob && !messages) {
        rms_norm(in_q.data(), st.bf16(lp + "attn_norm.weight"), nq.data(), D_MODEL);
        nk = nq;
        nv = nq;
    } else if (!is_glob) {
        const uint16_t* w = st.bf16(lp + "attn_norm.weight");
        rms_norm(in_q.data(), w, nq.data(), D_MODEL);
        rms_norm(in_k.data(), w, nk.data(), D_MODEL);
        rms_norm(in_v.data(), w, nv.data(), D_MODEL);
    } else {
        rms_norm(in_q.data(), st.bf16(lp + "q_input_norm.weight"), nq.data(), D_MODEL);
        rms_norm(in_k.data(), st.bf16(lp + "k_input_norm.weight"), nk.data(), D_MODEL);
        rms_norm(in_v.data(), st.bf16(lp + "v_input_norm.weight"), nv.data(), D_MODEL);
    }

    std::vector<float> Q(Q_DIM), K_raw(KV_DIM), V_raw(KV_DIM), K(KV_DIM), V(KV_DIM), R(N_HEADS * D_REL);
    matvec_bf16(st.bf16(lp + "attn.wq_du.weight"), nq.data(), Q.data(), Q_DIM, D_MODEL);
    matvec_bf16(st.bf16(lp + "attn.wk_dv.weight"), nk.data(), K_raw.data(), KV_DIM, D_MODEL);
    matvec_bf16(st.bf16(lp + "attn.wv_dv.weight"), nv.data(), V_raw.data(), KV_DIM, D_MODEL);
    matvec_bf16(st.bf16(lp + "attn.wr_du.weight"), nq.data(), R.data(), N_HEADS * D_REL, D_MODEL);

    if (is_glob) {
        float dyn_k[32], dyn_v[32];
        matvec_bf16(st.bf16(lp + "attn.k_sconv.weight_proj"), nk.data(), dyn_k, 32, D_MODEL);
        matvec_bf16(st.bf16(lp + "attn.v_sconv.weight_proj"), nv.data(), dyn_v, 32, D_MODEL);
        short_conv_step(K_raw.data(), slot.k_hist.data(), KV_DIM, st.bf16(lp + "attn.k_sconv.static_w"), true, dyn_k, K.data());
        short_conv_step(V_raw.data(), slot.v_hist.data(), KV_DIM, st.bf16(lp + "attn.v_sconv.static_w"), true, dyn_v, V.data());
    } else {
        short_conv_step(K_raw.data(), slot.k_hist.data(), KV_DIM, st.bf16(lp + "attn.k_sconv.weight"), false, nullptr, K.data());
        short_conv_step(V_raw.data(), slot.v_hist.data(), KV_DIM, st.bf16(lp + "attn.v_sconv.weight"), false, nullptr, V.data());
    }

    l2_clip_heads(Q.data(), st.bf16(lp + "attn.q_l2_clip.scale"), N_HEADS, HEAD_DIM, QK_CLIP);
    l2_clip_heads(K.data(), nullptr, N_KV_HEADS, HEAD_DIM, QK_CLIP);
    for (int h = 0; h < N_KV_HEADS; ++h) {
        rms_norm(V.data() + h * HEAD_DIM, nullptr, V.data() + h * HEAD_DIM, HEAD_DIM);
    }
    l2_clip_heads(R.data(), st.bf16(lp + "attn.rel_l2_clip.scale"), N_HEADS, D_REL, REL_CLIP);

    if (is_glob) {
        for (float& v : Q) v *= tau;
        for (float& v : R) v *= tau;
    }

    slot.k_cache.push_back(K);
    slot.v_cache.push_back(V);

    float sinks[N_HEADS];
    matvec_bf16(st.bf16(lp + "attn.attn_sink_proj.weight"), nq.data(), sinks, N_HEADS, D_MODEL);
    const uint16_t* rel_proj = st.bf16(lp + "attn.rel_logits_proj.proj");

    int start_j = is_glob ? 0 : std::max(0, pos - SWA_WINDOW + 1);
    int n_kv = pos - start_j + 1;
    const float attn_scale = 1.0f / std::sqrt((float)HEAD_DIM);

    std::vector<float> attn_out_raw(Q_DIM, 0.0f);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < N_HEADS; ++h) {
        const float* qh = Q.data() + h * HEAD_DIM;
        const float* rh = R.data() + h * D_REL;
        int kvh = h / (N_HEADS / N_KV_HEADS);

        std::vector<float> scores(n_kv);
        float mx = sinks[h];
        for (int idx = 0; idx < n_kv; ++idx) {
            int j = start_j + idx;
            const float* kj = slot.k_cache[j].data() + kvh * HEAD_DIM;
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) dot += qh[d] * kj[d];
            int dist = pos - j;
            float rel_b = 0.0f;
            if (dist >= 0 && dist < REL_EXTENT) {
                for (int r = 0; r < D_REL; ++r) {
                    rel_b += rh[r] * bf16_to_f32(rel_proj[r * REL_EXTENT + dist]);
                }
            }
            float s = dot * attn_scale + rel_b;
            scores[idx] = s;
            mx = std::max(mx, s);
        }
        float denom = std::exp(sinks[h] - mx);
        for (int idx = 0; idx < n_kv; ++idx) {
            scores[idx] = std::exp(scores[idx] - mx);
            denom += scores[idx];
        }
        float inv_d = 1.0f / denom;
        float* oh = attn_out_raw.data() + h * HEAD_DIM;
        for (int idx = 0; idx < n_kv; ++idx) {
            float p = scores[idx] * inv_d;
            const float* vj = slot.v_cache[start_j + idx].data() + kvh * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) oh[d] += p * vj[d];
        }
    }

    std::vector<float> gate_a(Q_DIM);
    matvec_bf16(st.bf16(lp + "attn.gate_attn.weight"), nq.data(), gate_a.data(), Q_DIM, D_MODEL);
    for (int i = 0; i < Q_DIM; ++i) attn_out_raw[i] *= 1.0f / (1.0f + std::exp(-gate_a[i]));

    std::vector<float> wo_out(D_MODEL), attn_out(D_MODEL);
    matvec_bf16(st.bf16(lp + "attn.wo_ud.weight"), attn_out_raw.data(), wo_out.data(), D_MODEL, Q_DIM);
    short_conv_step(wo_out.data(), slot.attn_hist.data(), D_MODEL, st.bf16(lp + "attn_sconv.weight"), false, nullptr, attn_out.data());

    std::vector<float> ffn_inp(D_MODEL);
    if (ip % BLOCK_SIZE == 0) {
        blocks.push_back(inpSA);
        ffn_inp = attn_out;
    } else {
        for (int i = 0; i < D_MODEL; ++i) ffn_inp[i] = inpSA[i] + attn_out[i];
    }

    std::vector<float> xm = attn_res_mix(blocks, ffn_inp, st.f32("model.llm.attn_res.mlp_queries." + std::to_string(ip)), 1);
    if (messages) {
        std::vector<float> tm(D_MODEL);
        loop_control(st, ip, "mlp", xm.data(), messages->data() + 3 * D_MODEL, tm.data());
        xm = std::move(tm);
    }
    std::vector<float> xm_norm(D_MODEL);
    rms_norm(xm.data(), st.bf16(lp + "mlp_norm.weight"), xm_norm.data(), D_MODEL);

    float r_logits[N_EXPERTS], r_probs[N_EXPERTS], r_sel[N_EXPERTS];
    matvec_bf16(st.bf16(lp + "mlp.gate.weight"), xm_norm.data(), r_logits, N_EXPERTS, D_MODEL);
    const float* r_bias = st.f32(lp + "mlp.gate.bias");
    for (int e = 0; e < N_EXPERTS; ++e) {
        r_probs[e] = 1.0f / (1.0f + std::exp(-r_logits[e]));
        r_sel[e]   = r_probs[e] + r_bias[e];
    }
    int idx_arr[N_EXPERTS];
    std::iota(idx_arr, idx_arr + N_EXPERTS, 0);
    std::partial_sort(idx_arr, idx_arr + TOP_K, idx_arr + N_EXPERTS, [&](int a, int b) {
        return r_sel[a] > r_sel[b];
    });
    float w_sum = 0.0f;
    for (int k = 0; k < TOP_K; ++k) w_sum += r_probs[idx_arr[k]];
    float w_norm = ROUTE_SCALE / (w_sum + 1e-20f);

    std::vector<float> moe_raw(D_MODEL, 0.0f);
    std::vector<float> gate_buf(INTER_SIZE), up_buf(INTER_SIZE), act_buf(INTER_SIZE), y_e(D_MODEL);
    const bool is_nvfp4 = (ip < 20);
    const uint8_t* w13 = st.u8(lp + "mlp.experts.w13_weight");
    const uint8_t* s13 = st.u8(lp + "mlp.experts.w13_weight_scale");
    const uint8_t* w2  = st.u8(lp + "mlp.experts.w2_weight");
    const uint8_t* s2  = st.u8(lp + "mlp.experts.w2_weight_scale");
    const float* a13 = is_nvfp4 ? st.f32(lp + "mlp.experts.w13_weight_amax") : nullptr;
    const float* d13 = is_nvfp4 ? st.f32(lp + "mlp.experts.w13_weight_outer_scale_denominator") : nullptr;
    const float* a2  = is_nvfp4 ? st.f32(lp + "mlp.experts.w2_weight_amax") : nullptr;
    const float* d2  = is_nvfp4 ? st.f32(lp + "mlp.experts.w2_weight_outer_scale_denominator") : nullptr;

    for (int k = 0; k < TOP_K; ++k) {
        int e = idx_arr[k];
        float ew = r_probs[e] * w_norm;
        if (is_nvfp4) {
            expert_matvec_nvfp4(w13, s13, a13[e] / d13[0], xm_norm.data(), gate_buf.data(), e, 2 * INTER_SIZE, D_MODEL, true, up_buf.data());
        } else {
            expert_matvec_mxfp8(w13, s13, xm_norm.data(), gate_buf.data(), e, 2 * INTER_SIZE, D_MODEL, true, up_buf.data());
        }
        for (int i = 0; i < INTER_SIZE; ++i) {
            float g = gate_buf[i];
            float u = up_buf[i];
            float act = (SITU_BETA * std::tanh(g / SITU_BETA)) * (1.0f / (1.0f + std::exp(-g)));
            float up  = SITU_LIN_BETA * std::tanh(u / SITU_LIN_BETA);
            act_buf[i] = act * up;
        }
        if (is_nvfp4) {
            expert_matvec_nvfp4(w2, s2, a2[e] / d2[0], act_buf.data(), y_e.data(), e, D_MODEL, INTER_SIZE, false, nullptr);
        } else {
            expert_matvec_mxfp8(w2, s2, act_buf.data(), y_e.data(), e, D_MODEL, INTER_SIZE, false, nullptr);
        }
        for (int i = 0; i < D_MODEL; ++i) moe_raw[i] += ew * y_e[i];
    }

    std::vector<float> mlp_out(D_MODEL), cur_out(D_MODEL);
    short_conv_step(moe_raw.data(), slot.mlp_hist.data(), D_MODEL, st.bf16(lp + "mlp_sconv.weight"), false, nullptr, mlp_out.data());
    for (int i = 0; i < D_MODEL; ++i) cur_out[i] = ffn_inp[i] + mlp_out[i];

    if (debug_print) {
        std::printf("  layer il=%2d ip=%2d l_out[0..3]= %.6f %.6f %.6f %.6f\n",
                    il, ip, cur_out[0], cur_out[1], cur_out[2], cur_out[3]);
    }
    return cur_out;
}

static int forward_one_token(const Safetensors& st, int token_id, int pos,
                             std::vector<SlotState>& slots, bool compute_logits, bool debug_print) {
    const uint16_t* emb_row = st.bf16("model.llm.embed.weight") + (size_t)token_id * D_MODEL;
    std::vector<float> emb_f32(D_MODEL), cur(D_MODEL);
    for (int i = 0; i < D_MODEL; ++i) emb_f32[i] = bf16_to_f32(emb_row[i]);
    rms_norm(emb_f32.data(), st.bf16("model.llm.embed_norm.weight"), cur.data(), D_MODEL);

    float tau = 1.0f + LOG_SCALE_ALPHA * std::log(std::max(1.0f, (float)(pos + 1) / LOG_SCALE_FLOOR));
    std::vector<std::vector<float>> blocks;

    for (int il = 0; il < CORE_START; ++il) {
        cur = run_layer(st, il, il, pos, tau, cur, blocks, nullptr, slots, debug_print);
    }

    const std::vector<float> prelude = cur;
    const auto prelude_blocks = blocks;
    const float* cp_q  = st.f32("model.llm.middle_cycle_cross_pass_queries");
    const float* cp_kv = st.f32("model.llm.middle_cycle_cross_pass_kv_queries");

    for (int pass = 0; pass < N_PASSES; ++pass) {
        std::vector<std::vector<float>> bank;
        std::vector<float> bank_out;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = CORE_START; ip < CORE_END; ++ip) {
            int il = (pass == 0) ? ip : N_PHYS + (pass - 1) * (CORE_END - CORE_START) + (ip - CORE_START);
            if (pass == 0) {
                cur = run_layer(st, il, ip, pos, tau, cur, blocks, nullptr, slots, debug_print);
            } else {
                int i = ip - CORE_START;
                std::vector<float> msg_q(4 * D_MODEL);
                std::memcpy(msg_q.data() + 0 * D_MODEL, cp_q  + (2 * i)     * D_MODEL, sizeof(float) * D_MODEL);
                std::memcpy(msg_q.data() + 1 * D_MODEL, cp_kv + (2 * i)     * D_MODEL, sizeof(float) * D_MODEL);
                std::memcpy(msg_q.data() + 2 * D_MODEL, cp_kv + (2 * i + 1) * D_MODEL, sizeof(float) * D_MODEL);
                std::memcpy(msg_q.data() + 3 * D_MODEL, cp_q  + (2 * i + 1) * D_MODEL, sizeof(float) * D_MODEL);
                auto messages = attn_res_mix(bank, bank_out, msg_q.data(), 4);
                cur = run_layer(st, il, ip, pos, tau, cur, blocks, &messages, slots, debug_print);
            }
        }
    }

    for (int il = CORE_END; il < N_PHYS; ++il) {
        cur = run_layer(st, il, il, pos, tau, cur, blocks, nullptr, slots, debug_print);
    }

    if (!compute_logits) return -1;

    auto out_mix = attn_res_mix(blocks, cur, st.f32("model.llm.attn_res.output_query"), 1);
    std::vector<float> out_norm(D_MODEL);
    rms_norm(out_mix.data(), st.bf16("model.llm.norm.weight"), out_norm.data(), D_MODEL);
    if (debug_print) {
        std::printf("  result_norm[0..3]= %.6f %.6f %.6f %.6f\n",
                    out_norm[0], out_norm[1], out_norm[2], out_norm[3]);
    }

    std::vector<float> logits(VOCAB_ACT);
    matvec_bf16(st.bf16("model.llm.unembed.weight"), out_norm.data(), logits.data(), VOCAB_ACT, D_MODEL);
    int best_id = 0;
    float best_val = -1e30f;
    double lsum = 0.0;
    for (int i = 0; i < VOCAB_ACT; ++i) {
        float v = LOGIT_SOFTCAP * std::tanh(logits[i] / LOGIT_SOFTCAP);
        logits[i] = v;
        lsum += v;
        if (v > best_val) { best_val = v; best_id = i; }
    }
    if (debug_print) {
        std::printf("  top1 id=%d val=%.6f logits[0..2]= %.4f %.4f %.4f logits[-3..]= %.4f %.4f %.4f sum=%.4f\n",
                    best_id, best_val,
                    logits[0], logits[1], logits[2],
                    logits[VOCAB_ACT-3], logits[VOCAB_ACT-2], logits[VOCAB_ACT-1], lsum);
    }
    return best_id;
}

int main(int argc, char** argv) {
    init_luts();
    std::string model_path = std::string(getenv("HOME")) + "/models/Inkling-2-Mini/model-00001-of-00001.safetensors";
    std::vector<int> prompt_tokens;
    int steps = 20;
    bool debug_print = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokens" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) prompt_tokens.push_back(std::stoi(tok));
            }
        } else if (a == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (a == "--debug") {
            debug_print = true;
        }
    }
    if (prompt_tokens.empty()) {
        std::cerr << "Usage: ref_cpu --tokens id1,id2,... [--steps 20] [--debug]\n";
        return 1;
    }

    Safetensors st;
    st.open(model_path);
    std::vector<SlotState> slots(N_SLOTS);

    int pos = 0;
    int next_tok = -1;
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        bool last = (i + 1 == prompt_tokens.size());
        next_tok = forward_one_token(st, prompt_tokens[i], pos++, slots, last, debug_print && last);
    }

    std::vector<int> generated;
    generated.push_back(next_tok);
    std::printf("step 0: %d\n", next_tok);
    std::fflush(stdout);

    for (int s = 1; s < steps; ++s) {
        next_tok = forward_one_token(st, next_tok, pos++, slots, true, false);
        generated.push_back(next_tok);
        std::printf("step %d: %d\n", s, next_tok);
        std::fflush(stdout);
    }

    std::printf("GENERATED:");
    for (size_t i = 0; i < generated.size(); ++i) {
        std::printf("%s%d", i ? "," : " ", generated[i]);
    }
    std::printf("\n");
    return 0;
}
