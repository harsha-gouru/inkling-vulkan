// Inkling-2-Mini Vulkan 1.3 GPU Engine — Hybrid 128k KV Cache + Batched Prefill/Verify (B <= 64).
#include "vk_abi.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <unordered_map>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "VK ERR %d at %s:%d\n", (int)_r, __FILE__, __LINE__); std::exit(1); } } while(0)

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
static constexpr int SWA_WINDOW  = 1024;
static constexpr int SCONV_HIST  = 3;
static constexpr int BLOCK_SIZE  = 6;
static constexpr int GATE_RANK   = 64;
static constexpr int N_EXPERTS   = 128;
static constexpr int TOP_K       = 8;
static constexpr int INTER_SIZE  = 1152;
static constexpr int VOCAB_ACT   = 200058;
static constexpr int MAX_BATCH   = 256;

static constexpr float QK_CLIP         = 22.627416997969522f;
static constexpr float REL_CLIP        = 8.0f;
static constexpr float LOG_SCALE_FLOOR = 8192.0f;
static constexpr float LOG_SCALE_ALPHA = 0.1f;

static inline bool is_global_layer(int ip) { return ip == 5 || ip == 11 || ip == 17 || ip == 23; }
static inline int slot_phys(int il) { return il < N_PHYS ? il : CORE_START + (il - N_PHYS) % (CORE_END - CORE_START); }
static inline uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }

struct TensorInfo { std::string dtype; std::vector<int64_t> shape; uint64_t off0 = 0, off1 = 0; };
struct Safetensors {
    int fd = -1; size_t file_size = 0; const uint8_t* base = nullptr; const uint8_t* data = nullptr;
    std::map<std::string, TensorInfo> tensors;
    void open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY); if (fd < 0) { perror("open"); std::exit(1); }
        struct stat st{}; fstat(fd, &st); file_size = (size_t)st.st_size;
        void* p = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) { perror("mmap"); std::exit(1); }
        base = (const uint8_t*)p; uint64_t hlen = 0; std::memcpy(&hlen, base, 8);
        data = base + 8 + hlen; parse_header(std::string((const char*)(base + 8), hlen));
    }
    ~Safetensors() { if (base) munmap((void*)base, file_size); if (fd >= 0) ::close(fd); }
    void parse_header(const std::string& s) {
        size_t pos = 0;
        auto skip_ws = [&]() { while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) ++pos; };
        auto parse_str = [&]() -> std::string {
            skip_ws(); if (s[pos++] != '"') std::exit(1);
            std::string out; while (pos < s.size() && s[pos] != '"') { if (s[pos] == '\\') ++pos; out.push_back(s[pos++]); }
            ++pos; return out;
        };
        auto parse_int = [&]() -> int64_t {
            skip_ws(); size_t start = pos; if (s[pos] == '-') ++pos;
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
            return std::stoll(s.substr(start, pos - start));
        };
        auto skip_obj = [&]() {
            int depth = 0; bool in_str = false;
            do { char c = s[pos++]; if (c == '"' && s[pos - 2] != '\\') in_str = !in_str;
                 if (!in_str) { if (c == '{') ++depth; else if (c == '}') --depth; }
            } while (depth > 0 && pos < s.size());
        };
        skip_ws(); if (s[pos++] != '{') std::exit(1);
        while (pos < s.size()) {
            skip_ws(); if (s[pos] == '}') break;
            std::string key = parse_str(); skip_ws(); ++pos; skip_ws();
            if (key == "__metadata__") skip_obj();
            else {
                TensorInfo ti; ++pos;
                while (pos < s.size()) {
                    skip_ws(); if (s[pos] == '}') { ++pos; break; }
                    std::string fk = parse_str(); skip_ws(); ++pos; skip_ws();
                    if (fk == "dtype") ti.dtype = parse_str();
                    else if (fk == "shape") { ++pos; while (true) { skip_ws(); if (s[pos] == ']') { ++pos; break; } ti.shape.push_back(parse_int()); skip_ws(); if (s[pos] == ',') ++pos; } }
                    else if (fk == "data_offsets") { ++pos; ti.off0 = (uint64_t)parse_int(); skip_ws(); if (s[pos] == ',') ++pos; ti.off1 = (uint64_t)parse_int(); skip_ws(); if (s[pos] == ']') ++pos; }
                    skip_ws(); if (s[pos] == ',') ++pos;
                }
                tensors[key] = std::move(ti);
            }
            skip_ws(); if (s[pos] == ',') ++pos;
        }
    }
};

struct GpuBuf { VkBuffer buf = 0; VkDeviceMemory mem = 0; uint64_t addr = 0; VkDeviceSize size = 0; void* mapped = nullptr; };
struct Pipe { VkShaderModule sm = 0; VkPipelineLayout pl = 0; VkPipeline pipe = 0; };

struct VkCtx {
    VkInstance inst = nullptr; VkPhysicalDevice phys = nullptr; VkDevice dev = nullptr;
    VkQueue queue = nullptr; uint32_t qfam = 0; VkCommandPool cmdp = 0; VkCommandBuffer cb = nullptr;
    VkFence fence = 0; VkQueryPool qp = 0; VkPhysicalDeviceMemoryProperties mp{}; float ts_period_ns = 10.0f;

    void init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "inkling_vk", 1, "inkling", 1, VK_MAKE_API_VERSION(0, 1, 3, 0)};
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 0, nullptr};
        VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));
        uint32_t ndev = 0; VK_CHECK(vkEnumeratePhysicalDevices(inst, &ndev, nullptr));
        std::vector<VkPhysicalDevice> devs(ndev); VK_CHECK(vkEnumeratePhysicalDevices(inst, &ndev, devs.data()));
        phys = devs[0];
        for (auto d : devs) { VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(d, &p); if (p.deviceType == 2) { phys = d; ts_period_ns = p.limits.timestampPeriod; break; } }
        vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(nqf); vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfs.data());
        for (uint32_t i = 0; i < nqf; ++i) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
        vkGetPhysicalDeviceFeatures2(phys, &f2);
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qfam, 1, &prio};
        VkPhysicalDeviceShaderBfloat16FeaturesKHR en_bf16{(VkStructureType)1000141000, nullptr, VK_TRUE, VK_FALSE, VK_TRUE};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR en_cm{(VkStructureType)1000506000, &en_bf16, VK_TRUE, VK_FALSE};
        VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &en_cm};
        en12.bufferDeviceAddress = VK_TRUE; en12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
        en12.shaderInt8 = f12.shaderInt8; en12.shaderFloat16 = f12.shaderFloat16; en12.scalarBlockLayout = f12.scalarBlockLayout;
        en12.vulkanMemoryModel = f12.vulkanMemoryModel; en12.vulkanMemoryModelDeviceScope = f12.vulkanMemoryModelDeviceScope;
        VkPhysicalDeviceVulkan11Features en11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &en12};
        en11.storageBuffer16BitAccess = f11.storageBuffer16BitAccess;
        const char* dev_exts[] = {"VK_KHR_cooperative_matrix", "VK_KHR_shader_bfloat16"};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &en11, 0, 1, &qci, 0, nullptr, 2, dev_exts, &f2.features};
        VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
        vkGetDeviceQueue(dev, qfam, 0, &queue);
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfam};
        VK_CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &cmdp));
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmdp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
        VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_TIMESTAMP, 1024, 0};
        VK_CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qp));
    }
    uint32_t find_mem(uint32_t bits, VkFlags req) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & req) == req) return i;
        std::exit(1);
    }
    GpuBuf alloc(VkDeviceSize sz, VkFlags mem_flags, bool map_it = false) {
        GpuBuf g{}; g.size = sz;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &g.buf));
        VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(dev, g.buf, &mr);
        VkMemoryAllocateFlagsInfo afi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &afi, mr.size, find_mem(mr.memoryTypeBits, mem_flags)};
        VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &g.mem));
        VK_CHECK(vkBindBufferMemory(dev, g.buf, g.mem, 0));
        VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, g.buf};
        g.addr = vkGetBufferDeviceAddress(dev, &bdai);
        if (map_it) VK_CHECK(vkMapMemory(dev, g.mem, 0, sz, 0, &g.mapped));
        return g;
    }
    Pipe load_pipe(const std::string& spv_path, uint32_t push_size) {
        std::ifstream f(spv_path, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "Cannot open %s\n", spv_path.c_str()); std::exit(1); }
        size_t sz = (size_t)f.tellg(); f.seekg(0); std::vector<uint32_t> code(sz / 4); f.read((char*)code.data(), sz);
        Pipe p{};
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, sz, code.data()};
        VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &p.sm));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &p.pl));
        VkComputePipelineCreateInfo cpci{}; cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, p.sm, "main", nullptr};
        cpci.layout = p.pl;
        VK_CHECK(vkCreateComputePipelines(dev, 0, 1, &cpci, nullptr, &p.pipe));
        return p;
    }
    void begin_cmd() {
        VK_CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    }
    void submit_and_wait() {
        VK_CHECK(vkEndCommandBuffer(cb));
        VK_CHECK(vkResetFences(dev, 1, &fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ULL));
    }
    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void barrier_xfer() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void upload_bytes(VkBuffer dst_buf, VkDeviceSize dst_off, const void* src, size_t nbytes, GpuBuf& staging) {
        size_t done = 0;
        while (done < nbytes) {
            size_t chunk = std::min<size_t>(nbytes - done, staging.size);
            std::memcpy(staging.mapped, (const uint8_t*)src + done, chunk);
            begin_cmd(); VkBufferCopy bc{0, dst_off + done, chunk};
            vkCmdCopyBuffer(cb, staging.buf, dst_buf, 1, &bc); submit_and_wait();
            done += chunk;
        }
    }
};


static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f;
}
static inline uint8_t f32_to_e4m3(float v) {
    uint32_t sign = (v < 0.0f) ? 0x80u : 0u;
    float av = std::fabs(v);
    if (av >= 448.0f) return (uint8_t)(sign | 0x7Eu);
    if (av < 0.0009765625f) return (uint8_t)sign;
    if (av < 0.015625f) {
        int m = (int)std::lrintf(av * 512.0f);
        if (m >= 8) return (uint8_t)(sign | 0x08u);
        return (uint8_t)(sign | (uint32_t)m);
    }
    uint32_t u; std::memcpy(&u, &av, 4);
    int exp = (int)((u >> 23) & 0xFFu) - 127 + 7;
    uint32_t frac = u & 0x7FFFFFu;
    uint32_t man = (frac + 0x00080000u) >> 20;
    if (man >= 8u) { man = 0; exp++; }
    if (exp >= 15 && man >= 7u) { exp = 15; man = 6u; }
    if (exp > 15) { exp = 15; man = 6u; }
    return (uint8_t)(sign | ((uint32_t)exp << 3) | man);
}

struct Engine {
    VkCtx vk;
    std::vector<GpuBuf> weight_slabs;
    bool use_fp8 = false;
    GpuBuf fp8_slab;
    std::unordered_map<std::string, uint32_t> fp8_woff, fp8_soff;
    std::map<std::string, uint64_t> waddr;
    GpuBuf msg_q_buf, rel_proj_t_buf, scratch, hist_buf, hist_steps_buf, hist_ckpt_buf, target_h_buf, tok_ids_host, out_ids_host;
    bool mtp_loaded = false;
    GpuBuf mtp_slab, mtp_scratch, mtp_swa_kv, mtp_glob_k, mtp_glob_v, mtp_toks_host;
    std::map<std::string, uint64_t> mtp_waddr;
    float mtp_fw[4][4]{};
    uint64_t mtp_k_addr[3]{}, mtp_v_addr[3]{};
    uint32_t mtp_ring_mask[3]{1023u, 0u, 1023u};

    // Hybrid KV cache: 40 SWA ring-buffers (cap=1024 each) in swa_kv_buf,
    // 8 Global linear buffers (cap=max_ctx each) in glob_kv_bufs[8] (K and V separate -> <= 0.5 GiB each at 128k!)
    GpuBuf swa_kv_buf;
    std::vector<GpuBuf> glob_k_bufs, glob_v_bufs;
    uint64_t slot_k_addr[N_SLOTS]{};
    uint64_t slot_v_addr[N_SLOTS]{};
    uint32_t slot_ring_mask[N_SLOTS]{};
    int max_ctx = 128000;

    // Single-token pipelines
    Pipe p_embed, p_rmsnorm, p_attn_res, p_matvec, p_matvec_attn, p_matvec_fp8, p_matvec_fp8_attn, pb_mat_fp8, p_sconv, p_l2clip, p_attn, p_attn_red, p_elem, p_router, p_nvfp4, p_mxfp8, p_argmax, p_ctrl_down, p_ctrl_apply;
    // Batched pipelines (B <= 64)
    Pipe pb_embed, pb_attn_res, pb_mat, pb_sconv, pb_kv_store, pb_l2clip, pb_attn, pb_router, pb_group, pb_nvfp4, pb_mxfp8, pb_reduce, pb_ctrl_down, pb_ctrl_apply;
    Pipe pm_fuse, pm_rope, pm_attn, pm_attn_sk, pm_attn_red, pm_top256, pm_gru, pm_cand, pb_attn_fa2, pb_attn_sk, pb_attn_red, pb_gemm, pb_nvfp4_gemm;

    struct Off {
        uint64_t cur[2], ffn_inp, prelude, bank_block, bank_out;
        uint64_t blocks[4];
        uint64_t mixed, messages;
        uint64_t nq, nk, nv;
        uint64_t ctrl_gate[3], ctrl_gain[3];
        uint64_t Q, K_raw, V_raw, K, V, R, dyn_k, dyn_v, sinks;
        uint64_t attn_raw, gate_a, wo_out, attn_out, attn_part_ml, attn_part_o;
        uint64_t xm, xm_norm, r_logits, top_ids, top_w;
        uint64_t exp_counts, exp_offsets, pair_tok, pair_k;
        uint64_t act_buf, y_exps, moe_raw, mlp_out, logits, total;
    } off{};

    std::vector<int> cached_tokens;
    size_t ckpt_len = 0;

    uint64_t sa(uint64_t o) const { return scratch.addr + o; }
    uint64_t wa(const std::string& k) const { return waddr.at(k); }

    void init(const std::string& model_path, const std::string& shader_dir, int max_context, bool enable_fp8 = false) {
        use_fp8 = enable_fp8;
        max_ctx = max_context;
        vk.init();

        // Single-token pipelines
        p_embed      = vk.load_pipe(shader_dir + "/embed_rmsnorm.spv", 32);
        p_rmsnorm    = vk.load_pipe(shader_dir + "/rmsnorm.spv", 28);
        p_attn_res   = vk.load_pipe(shader_dir + "/attn_res_mix.spv", 120);
        p_matvec     = vk.load_pipe(shader_dir + "/matvec_bf16.spv", 32);
        p_matvec_attn = vk.load_pipe(shader_dir + "/matvec_bf16_attn.spv", 116);
        p_matvec_fp8      = vk.load_pipe(shader_dir + "/matvec_fp8.spv", 40);
        p_matvec_fp8_attn = vk.load_pipe(shader_dir + "/matvec_fp8_attn.spv", 124);
        pb_mat_fp8        = vk.load_pipe(shader_dir + "/mat_fp8_batch.spv", 44);
        p_sconv      = vk.load_pipe(shader_dir + "/short_conv.spv", 56);
        p_l2clip     = vk.load_pipe(shader_dir + "/l2_clip.spv", 28);
        p_attn       = vk.load_pipe(shader_dir + "/attn_decode_splitk.spv", 96);
        p_attn_red   = vk.load_pipe(shader_dir + "/attn_reduce.spv", 44);
        p_elem       = vk.load_pipe(shader_dir + "/elemwise.spv", 40);
        p_router     = vk.load_pipe(shader_dir + "/moe_router.spv", 32);
        p_nvfp4      = vk.load_pipe(shader_dir + "/moe_nvfp4_matvec.spv", 76);
        p_mxfp8      = vk.load_pipe(shader_dir + "/moe_mxfp8_matvec.spv", 60);
        p_argmax     = vk.load_pipe(shader_dir + "/softcap_argmax.spv", 40);
        p_ctrl_down  = vk.load_pipe(shader_dir + "/ctrl_down_fused.spv", 104);
        p_ctrl_apply = vk.load_pipe(shader_dir + "/ctrl_apply_norm_fused.spv", 152);

        // Batched pipelines
        pb_embed      = vk.load_pipe(shader_dir + "/embed_batch.spv", 36);
        pb_attn_res   = vk.load_pipe(shader_dir + "/attn_res_mix_batch.spv", 124);
        pb_mat        = vk.load_pipe(shader_dir + "/mat_bf16_batch.spv", 36);
        pb_gemm       = vk.load_pipe(shader_dir + "/mat_bf16_gemm.spv", 36);
        pb_sconv      = vk.load_pipe(shader_dir + "/short_conv_batch.spv", 60);
        pm_fuse       = vk.load_pipe(shader_dir + "/mtp_fuse_norm.spv", 68);
        pm_rope       = vk.load_pipe(shader_dir + "/mtp_rope_qk.spv", 52);
        pm_attn       = vk.load_pipe(shader_dir + "/mtp_attn.spv", 60);
        pm_top256     = vk.load_pipe(shader_dir + "/mtp_top256.spv", 28);
        pm_gru        = vk.load_pipe(shader_dir + "/mtp_markov_gru.spv", 52);
        pm_cand       = vk.load_pipe(shader_dir + "/mtp_cand_score.spv", 52);
        pb_kv_store   = vk.load_pipe(shader_dir + "/kv_store_batch.spv", 40);
        pb_l2clip     = vk.load_pipe(shader_dir + "/l2_clip_batch.spv", 36);
        pb_attn       = vk.load_pipe(shader_dir + "/attn_prefill_batch.spv", 84);
        pb_attn_fa2   = vk.load_pipe(shader_dir + "/attn_prefill_fa2.spv", 80);
        pb_attn_sk    = vk.load_pipe(shader_dir + "/attn_verify_splitk.spv", 72);
        pb_attn_red   = vk.load_pipe(shader_dir + "/attn_verify_reduce.spv", 44);
        pm_attn_sk    = vk.load_pipe(shader_dir + "/mtp_attn_splitk.spv", 68);
        pm_attn_red   = vk.load_pipe(shader_dir + "/mtp_attn_reduce.spv", 28);
        pb_router     = vk.load_pipe(shader_dir + "/moe_router_batch.spv", 32);
        pb_group      = vk.load_pipe(shader_dir + "/moe_group_batch.spv", 44);
        pb_nvfp4      = vk.load_pipe(shader_dir + "/moe_nvfp4_batch.spv", 92);
        pb_nvfp4_gemm = vk.load_pipe(shader_dir + "/moe_nvfp4_gemm.spv", 92);
        pb_mxfp8      = vk.load_pipe(shader_dir + "/moe_mxfp8_batch.spv", 76);
        pb_reduce     = vk.load_pipe(shader_dir + "/moe_reduce_batch.spv", 28);
        pb_ctrl_down  = vk.load_pipe(shader_dir + "/ctrl_down_batch.spv", 104);
        pb_ctrl_apply = vk.load_pipe(shader_dir + "/ctrl_apply_batch.spv", 152);

        // Upload LLM weights
        Safetensors st; st.open(model_path);
        GpuBuf staging = vk.alloc(64ULL * 1024 * 1024, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);

        const uint64_t SLAB_MAX = 2ULL * 1024 * 1024 * 1024;
        std::vector<std::vector<std::pair<std::string, uint64_t>>> slab_plan(1);
        std::vector<uint64_t> slab_sizes; uint64_t cur_sz = 0;
        for (const auto& [name, ti] : st.tensors) {
            if (name.rfind("model.llm.", 0) != 0) continue;
            uint64_t aligned = align_up(ti.off1 - ti.off0, 256);
            if (cur_sz + aligned > SLAB_MAX && cur_sz > 0) { slab_sizes.push_back(cur_sz); slab_plan.emplace_back(); cur_sz = 0; }
            slab_plan.back().push_back({name, cur_sz}); cur_sz += aligned;
        }
        slab_sizes.push_back(cur_sz);

        auto t0 = std::chrono::steady_clock::now();
        uint64_t total_w = 0;
        for (size_t s = 0; s < slab_plan.size(); ++s) {
            GpuBuf slab = vk.alloc(slab_sizes[s], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            for (const auto& [name, offset] : slab_plan[s]) {
                const auto& ti = st.tensors.at(name); uint64_t sz = ti.off1 - ti.off0;
                vk.upload_bytes(slab.buf, offset, st.data + ti.off0, sz, staging);
                waddr[name] = slab.addr + offset; total_w += sz;
            }
            weight_slabs.push_back(slab);
        }
        {
            const float* cp_q  = (const float*)(st.data + st.tensors.at("model.llm.middle_cycle_cross_pass_queries").off0);
            const float* cp_kv = (const float*)(st.data + st.tensors.at("model.llm.middle_cycle_cross_pass_kv_queries").off0);
            std::vector<float> packed(12 * 4 * D_MODEL);
            for (int i = 0; i < 12; ++i) {
                float* dst = packed.data() + i * 4 * D_MODEL;
                std::memcpy(dst + 0 * D_MODEL, cp_q  + (2 * i)     * D_MODEL, sizeof(float) * D_MODEL);
                std::memcpy(dst + 1 * D_MODEL, cp_kv + (2 * i)     * D_MODEL, sizeof(float) * D_MODEL);
                std::memcpy(dst + 2 * D_MODEL, cp_kv + (2 * i + 1) * D_MODEL, sizeof(float) * D_MODEL);
                std::memcpy(dst + 3 * D_MODEL, cp_q  + (2 * i + 1) * D_MODEL, sizeof(float) * D_MODEL);
            }
            msg_q_buf = vk.alloc(packed.size() * 4, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            vk.upload_bytes(msg_q_buf.buf, 0, packed.data(), packed.size() * 4, staging);
        }
        {
            std::vector<uint16_t> rel_t((size_t)N_PHYS * 1024 * 16);
            for (int ip = 0; ip < N_PHYS; ++ip) {
                const auto& ti = st.tensors.at("model.llm.layers." + std::to_string(ip) + ".attn.rel_logits_proj.proj");
                const uint16_t* src = (const uint16_t*)(st.data + ti.off0);
                uint16_t* dst = rel_t.data() + (size_t)ip * 1024 * 16;
                for (int r = 0; r < 16; ++r)
                    for (int c = 0; c < 1024; ++c)
                        dst[c * 16 + r] = src[r * 1024 + c];
            }
            rel_proj_t_buf = vk.alloc(rel_t.size() * 2, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            vk.upload_bytes(rel_proj_t_buf.buf, 0, rel_t.data(), rel_t.size() * 2, staging);
        }
        if (use_fp8) {
            auto tf0 = std::chrono::steady_clock::now();
            std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>> targets;
            for (int ip = 0; ip < N_PHYS; ++ip) {
                std::string lp = "model.llm.layers." + std::to_string(ip) + ".";
                targets.push_back({lp + "attn.wq_du.weight", {Q_DIM, D_MODEL}});
                targets.push_back({lp + "attn.wk_dv.weight", {KV_DIM, D_MODEL}});
                targets.push_back({lp + "attn.wv_dv.weight", {KV_DIM, D_MODEL}});
                targets.push_back({lp + "attn.wr_du.weight", {N_HEADS * D_REL, D_MODEL}});
                targets.push_back({lp + "attn.attn_sink_proj.weight", {N_HEADS, D_MODEL}});
                targets.push_back({lp + "attn.gate_attn.weight", {Q_DIM, D_MODEL}});
                targets.push_back({lp + "attn.wo_ud.weight", {D_MODEL, Q_DIM}});
                if (st.tensors.count(lp + "attn.k_sconv.weight_proj")) {
                    targets.push_back({lp + "attn.k_sconv.weight_proj", {32, D_MODEL}});
                    targets.push_back({lp + "attn.v_sconv.weight_proj", {32, D_MODEL}});
                }
            }
            targets.push_back({"model.llm.unembed.weight", {VOCAB_ACT, D_MODEL}});
            uint64_t fcur = 0;
            for (auto& [nm, rc] : targets) {
                fp8_woff[nm] = (uint32_t)fcur; fcur += align_up((uint64_t)rc.first * rc.second, 256);
                fp8_soff[nm] = (uint32_t)fcur; fcur += align_up((uint64_t)rc.first * 4ULL, 256);
            }
            fp8_slab = vk.alloc(fcur, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            std::vector<uint8_t> tmp_w;
            std::vector<float> tmp_s;
            for (auto& [nm, rc] : targets) {
                uint32_t R = rc.first, C = rc.second;
                const uint16_t* src = (const uint16_t*)(st.data + st.tensors.at(nm).off0);
                tmp_w.resize((size_t)R * C); tmp_s.resize(R);
                #pragma omp parallel for schedule(static)
                for (int r = 0; r < (int)R; ++r) {
                    const uint16_t* row = src + (size_t)r * C;
                    float amax = 0.0f;
                    for (uint32_t c = 0; c < C; ++c) {
                        float v = std::fabs(bf16_to_f32(row[c]));
                        if (v > amax) amax = v;
                    }
                    float sc = (amax > 0.0f) ? (amax / 448.0f) : 1.0f;
                    float inv = 1.0f / sc;
                    tmp_s[r] = sc;
                    uint8_t* dst = tmp_w.data() + (size_t)r * C;
                    for (uint32_t c = 0; c < C; ++c) dst[c] = f32_to_e4m3(bf16_to_f32(row[c]) * inv);
                }
                vk.upload_bytes(fp8_slab.buf, fp8_woff[nm], tmp_w.data(), tmp_w.size(), staging);
                vk.upload_bytes(fp8_slab.buf, fp8_soff[nm], tmp_s.data(), tmp_s.size() * 4ULL, staging);
            }
            double tf_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
            std::fprintf(stderr, "Quantized FP8 E4M3 dense weights (%.2f GiB) in %.2f s\n", fcur / (1024.0*1024.0*1024.0), tf_s);
        }

        vkUnmapMemory(vk.dev, staging.mem); vkDestroyBuffer(vk.dev, staging.buf, nullptr); vkFreeMemory(vk.dev, staging.mem, nullptr);

        // Allocate scratch sized for MAX_BATCH = 64 (~30 MiB total!)
        uint64_t cur_off = 0;
        auto alloc_s = [&](size_t floats) -> uint64_t { uint64_t o = cur_off; cur_off = align_up(cur_off + floats * 4, 256); return o; };
        off.cur[0] = alloc_s(MAX_BATCH * D_MODEL); off.cur[1] = alloc_s(MAX_BATCH * D_MODEL);
        off.ffn_inp = alloc_s(MAX_BATCH * D_MODEL); off.prelude = alloc_s(MAX_BATCH * D_MODEL);
        off.bank_block = alloc_s(MAX_BATCH * D_MODEL); off.bank_out = alloc_s(MAX_BATCH * D_MODEL);
        for (int i = 0; i < 4; ++i) off.blocks[i] = alloc_s(MAX_BATCH * D_MODEL);
        off.mixed = alloc_s(4 * MAX_BATCH * D_MODEL);
        off.messages = alloc_s(4 * MAX_BATCH * D_MODEL);
        off.nq = alloc_s(MAX_BATCH * D_MODEL); off.nk = alloc_s(MAX_BATCH * D_MODEL); off.nv = alloc_s(MAX_BATCH * D_MODEL);
        for (int c = 0; c < 3; ++c) {
            off.ctrl_gate[c] = alloc_s(MAX_BATCH * GATE_RANK);
            off.ctrl_gain[c] = alloc_s(MAX_BATCH * D_MODEL);
        }
        off.Q = alloc_s(MAX_BATCH * Q_DIM); off.K_raw = alloc_s(MAX_BATCH * KV_DIM); off.V_raw = alloc_s(MAX_BATCH * KV_DIM);
        off.K = alloc_s(MAX_BATCH * KV_DIM); off.V = alloc_s(MAX_BATCH * KV_DIM); off.R = alloc_s(MAX_BATCH * N_HEADS * D_REL);
        off.dyn_k = alloc_s(MAX_BATCH * 32); off.dyn_v = alloc_s(MAX_BATCH * 32); off.sinks = alloc_s(MAX_BATCH * N_HEADS);
        off.attn_raw = alloc_s(MAX_BATCH * Q_DIM); off.gate_a = alloc_s(MAX_BATCH * Q_DIM);
        off.wo_out = alloc_s(MAX_BATCH * D_MODEL); off.attn_out = alloc_s(MAX_BATCH * D_MODEL);
        off.attn_part_ml = alloc_s(N_HEADS * 8 * 32 * 2); off.attn_part_o = alloc_s(N_HEADS * 8 * 32 * HEAD_DIM);
        off.xm = alloc_s(MAX_BATCH * D_MODEL); off.xm_norm = alloc_s(MAX_BATCH * D_MODEL);
        off.r_logits = alloc_s(MAX_BATCH * N_EXPERTS); off.top_ids = alloc_s(MAX_BATCH * TOP_K); off.top_w = alloc_s(MAX_BATCH * TOP_K);
        off.exp_counts = alloc_s(N_EXPERTS); off.exp_offsets = alloc_s(2 * N_EXPERTS);
        off.pair_tok = alloc_s(MAX_BATCH * TOP_K); off.pair_k = alloc_s(MAX_BATCH * TOP_K);
        off.act_buf = alloc_s(MAX_BATCH * TOP_K * INTER_SIZE);
        off.y_exps = alloc_s(MAX_BATCH * TOP_K * D_MODEL);
        off.moe_raw = alloc_s(MAX_BATCH * D_MODEL); off.mlp_out = alloc_s(MAX_BATCH * D_MODEL);
        off.logits = alloc_s(8 * VOCAB_ACT); // up to 8 verify tokens logits at once
        off.total = cur_off;
        scratch = vk.alloc(off.total, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Hybrid KV cache allocation: 40 SWA slots (ring=1024) + 8 Global slots (linear=max_ctx)
        VkDeviceSize swa_slot_bytes = (VkDeviceSize)SWA_WINDOW * KV_DIM * 4ULL; // 4 MiB per K or V
        swa_kv_buf = vk.alloc(40ULL * 2ULL * swa_slot_bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceSize glob_slot_bytes = (VkDeviceSize)max_ctx * KV_DIM * 4ULL;   // 500 MiB per K or V at 128k
        int swa_idx = 0;
        for (int il = 0; il < N_SLOTS; ++il) {
            if (is_global_layer(slot_phys(il))) {
                GpuBuf gk = vk.alloc(glob_slot_bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                GpuBuf gv = vk.alloc(glob_slot_bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                slot_k_addr[il] = gk.addr; slot_v_addr[il] = gv.addr; slot_ring_mask[il] = 0u;
                glob_k_bufs.push_back(gk); glob_v_bufs.push_back(gv);
            } else {
                slot_k_addr[il] = swa_kv_buf.addr + (uint64_t)(swa_idx * 2)     * swa_slot_bytes;
                slot_v_addr[il] = swa_kv_buf.addr + (uint64_t)(swa_idx * 2 + 1) * swa_slot_bytes;
                slot_ring_mask[il] = SWA_WINDOW - 1;
                swa_idx++;
            }
        }

        hist_buf = vk.alloc((VkDeviceSize)N_SLOTS * SCONV_HIST * (2 * KV_DIM + 2 * D_MODEL) * 4ULL, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        hist_ckpt_buf = vk.alloc(hist_buf.size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        target_h_buf = vk.alloc(4ULL * MAX_BATCH * D_MODEL * 4ULL, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        tok_ids_host = vk.alloc(MAX_BATCH * 4, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
        out_ids_host = vk.alloc(MAX_BATCH * 4, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);

        reset_state();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double kv_gib = (swa_kv_buf.size + (double)glob_k_bufs.size() * 2.0 * glob_slot_bytes) / (1024.0*1024.0*1024.0);
        std::fprintf(stderr, "Loaded %.3f GB weights in %.2f s | Hybrid KV=%.2f GiB (max_ctx=%d, 40 SWA + 8 Global) | scratch=%.1f MiB\n",
                     total_w / 1e9, dt, kv_gib, max_ctx, off.total / (1024.0*1024.0));
    }


    uint64_t mwa(const std::string& name) const { return mtp_waddr.at(name); }
    static constexpr uint64_t MTP_FF = 9216;
    struct MtpOff {
        uint64_t ctx_h, ctx_k, ctx_v, X, Xn, Q, Kd, Vd, attn_o, oproj, gate, up, down, H, ctx_m;
        uint64_t cand_ids, cand_vals, state, joint, proj, mk, dh, scores;
        uint64_t total;
    } moff{};
    uint64_t msa(uint64_t o) const { return mtp_scratch.addr + o; }

    void init_mtp(const std::string& mtp_path) {
        auto t0 = std::chrono::steady_clock::now();
        Safetensors msf; msf.open(mtp_path);
        size_t total_b = 0;
        for (auto& [n, ti] : msf.tensors) total_b = align_up(total_b + (ti.off1 - ti.off0), 256);
        mtp_slab = vk.alloc(total_b, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        GpuBuf staging = vk.alloc(64ULL * 1024 * 1024, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
        size_t off_b = 0;
        for (auto& [n, ti] : msf.tensors) {
            size_t sz = ti.off1 - ti.off0;
            vk.upload_bytes(mtp_slab.buf, off_b, msf.data + ti.off0, sz, staging);
            mtp_waddr[n] = mtp_slab.addr + off_b;
            off_b = align_up(off_b + sz, 256);
        }
        const uint16_t* fw = (const uint16_t*)(msf.data + msf.tensors.at("fuse_weight").off0);
        for (int c = 0; c < 4; ++c) {
            float mx = -1e30f;
            for (int j = 0; j < 4; ++j) {
                uint32_t u = (uint32_t)fw[c * 4 + j] << 16; float f; std::memcpy(&f, &u, 4);
                mtp_fw[c][j] = f; mx = std::max(mx, f);
            }
            float sm = 0; for (int j = 0; j < 4; ++j) { mtp_fw[c][j] = std::exp(mtp_fw[c][j] - mx); sm += mtp_fw[c][j]; }
            for (int j = 0; j < 4; ++j) mtp_fw[c][j] /= sm;
        }
        vkUnmapMemory(vk.dev, staging.mem); vkDestroyBuffer(vk.dev, staging.buf, nullptr); vkFreeMemory(vk.dev, staging.mem, nullptr);

        hist_steps_buf = vk.alloc(8ULL * hist_buf.size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceSize swa_b = (VkDeviceSize)SWA_WINDOW * KV_DIM * 4ULL;
        mtp_swa_kv = vk.alloc(4ULL * swa_b, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceSize glob_b = (VkDeviceSize)max_ctx * KV_DIM * 4ULL;
        mtp_glob_k = vk.alloc(glob_b, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        mtp_glob_v = vk.alloc(glob_b, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        mtp_k_addr[0] = mtp_swa_kv.addr + 0 * swa_b; mtp_v_addr[0] = mtp_swa_kv.addr + 1 * swa_b;
        mtp_k_addr[1] = mtp_glob_k.addr;             mtp_v_addr[1] = mtp_glob_v.addr;
        mtp_k_addr[2] = mtp_swa_kv.addr + 2 * swa_b; mtp_v_addr[2] = mtp_swa_kv.addr + 3 * swa_b;

        uint64_t cur_o = 0;
        auto ma = [&](size_t fl) -> uint64_t { uint64_t o = cur_o; cur_o = align_up(cur_o + fl * 4, 256); return o; };
        moff.ctx_h = ma(MAX_BATCH * D_MODEL); moff.ctx_k = ma(MAX_BATCH * KV_DIM); moff.ctx_v = ma(MAX_BATCH * KV_DIM);
        moff.X = ma(8 * D_MODEL); moff.Xn = ma(8 * D_MODEL);
        moff.Q = ma(8 * Q_DIM); moff.Kd = ma(8 * KV_DIM); moff.Vd = ma(8 * KV_DIM);
        moff.attn_o = ma(8 * Q_DIM); moff.oproj = ma(8 * D_MODEL);
        moff.gate = ma(8 * MTP_FF); moff.up = ma(8 * MTP_FF); moff.down = ma(8 * D_MODEL);
        moff.H = ma(8 * D_MODEL); moff.ctx_m = ma(D_MODEL);
        moff.cand_ids = ma(7 * 1024); moff.cand_vals = ma(7 * 1024);
        moff.state = ma(256); moff.joint = ma(5120); moff.proj = ma(768); moff.mk = ma(256); moff.dh = ma(D_MODEL); moff.scores = ma(1024);
        moff.total = cur_o;
        mtp_scratch = vk.alloc(moff.total, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        mtp_toks_host = vk.alloc(16 * 4, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
        mtp_loaded = true;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "Loaded MTP DFlare (%.1f MB) in %.2f s | MTP KV=%.2f GiB | hist_steps=%.1f MiB\n",
                     total_b / 1e6, dt, (mtp_swa_kv.size + mtp_glob_k.size + mtp_glob_v.size) / (1024.0*1024.0*1024.0),
                     hist_steps_buf.size / (1024.0*1024.0));
    }

    uint64_t th_addr(int slot) const { return target_h_buf.addr + (uint64_t)slot * MAX_BATCH * D_MODEL * 4ULL; }

    void mtp_update_ctx_kv(uint32_t n_toks, uint32_t pos_start) {
        vk.begin_cmd();
        for (int l = 0; l < 3; ++l) {
            std::string lp = "layers." + std::to_string(l) + ".";
            struct { uint64_t h0, h1, h2, h3, nw, o; float w0, w1, w2, w3; uint32_t dim; } pf{
                th_addr(0), th_addr(1), th_addr(2), th_addr(3), mwa(lp + "dflare_norm.weight"), msa(moff.ctx_h),
                mtp_fw[l][0], mtp_fw[l][1], mtp_fw[l][2], mtp_fw[l][3], D_MODEL
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_fuse.pipe);
            vkCmdPushConstants(vk.cb, pm_fuse.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pf), &pf);
            vkCmdDispatch(vk.cb, n_toks, 1, 1); vk.barrier();

            nbb_mat(mwa(lp + "self_attn.k_proj_ctx.weight"), msa(moff.ctx_h), msa(moff.ctx_k), KV_DIM, D_MODEL, n_toks);
            nbb_mat(mwa(lp + "self_attn.v_proj_ctx.weight"), msa(moff.ctx_h), msa(moff.ctx_v), KV_DIM, D_MODEL, n_toks);
            vk.barrier();

            struct { uint64_t x, nw, vi, kc, vc; uint32_t nh, ps, rm; } pr{
                msa(moff.ctx_k), mwa(lp + "self_attn.k_norm_ctx.weight"), msa(moff.ctx_v),
                mtp_k_addr[l], mtp_v_addr[l], N_KV_HEADS, pos_start, mtp_ring_mask[l]
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_rope.pipe);
            vkCmdPushConstants(vk.cb, pm_rope.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pr), &pr);
            vkCmdDispatch(vk.cb, N_KV_HEADS, n_toks, 1); vk.barrier();
        }
        uint64_t last_off = (uint64_t)(n_toks - 1) * D_MODEL * 4ULL;
        struct { uint64_t h0, h1, h2, h3, nw, o; float w0, w1, w2, w3; uint32_t dim; } pfm{
            th_addr(0) + last_off, th_addr(1) + last_off, th_addr(2) + last_off, th_addr(3) + last_off,
            mwa("markov_head.dflare_norm.weight"), msa(moff.ctx_m),
            mtp_fw[3][0], mtp_fw[3][1], mtp_fw[3][2], mtp_fw[3][3], D_MODEL
        };
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_fuse.pipe);
        vkCmdPushConstants(vk.cb, pm_fuse.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pfm), &pfm);
        vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
        vk.submit_and_wait();
    }

    void mtp_draft_block(int anchor_tok, uint32_t ctx_len, uint32_t n_draft, int* out_draft) {
        uint32_t* mt = (uint32_t*)mtp_toks_host.mapped;
        mt[0] = (uint32_t)anchor_tok;
        const uint32_t BS = 8u;

        vk.begin_cmd();
        {
            struct { uint64_t e, n, o; uint32_t tid, dim; } pc{
                wa("model.llm.embed.weight"), wa("model.llm.embed_norm.weight"), msa(moff.X), (uint32_t)anchor_tok, D_MODEL
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_embed.pipe);
            vkCmdPushConstants(vk.cb, p_embed.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, 1, 1, 1);
            nb_elem(mwa("mask_embed"), 0, msa(moff.X) + D_MODEL * 4ULL, 0, (BS - 1) * D_MODEL, 6);
            vk.barrier();
        }
        for (int l = 0; l < 3; ++l) {
            std::string lp = "layers." + std::to_string(l) + ".";
            nb_rmsnorm(msa(moff.X), mwa(lp + "input_layernorm.weight"), msa(moff.Xn), D_MODEL, BS); vk.barrier();
            nbb_mat(mwa(lp + "self_attn.q_proj.weight"), msa(moff.Xn), msa(moff.Q), Q_DIM, D_MODEL, BS);
            nbb_mat(mwa(lp + "self_attn.k_proj.weight"), msa(moff.Xn), msa(moff.Kd), KV_DIM, D_MODEL, BS);
            nbb_mat(mwa(lp + "self_attn.v_proj.weight"), msa(moff.Xn), msa(moff.Vd), KV_DIM, D_MODEL, BS);
            vk.barrier();

            struct { uint64_t x, nw, vi, kc, vc; uint32_t nh, ps, rm; } prq{
                msa(moff.Q), mwa(lp + "self_attn.q_norm.weight"), 0, 0, 0, N_HEADS, ctx_len, 0
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_rope.pipe);
            vkCmdPushConstants(vk.cb, pm_rope.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(prq), &prq);
            vkCmdDispatch(vk.cb, N_HEADS, BS, 1);
            struct { uint64_t x, nw, vi, kc, vc; uint32_t nh, ps, rm; } prk{
                msa(moff.Kd), mwa(lp + "self_attn.k_norm.weight"), 0, 0, 0, N_KV_HEADS, ctx_len, 0
            };
            vkCmdPushConstants(vk.cb, pm_rope.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(prk), &prk);
            vkCmdDispatch(vk.cb, N_KV_HEADS, BS, 1);
            vk.barrier();

            if (mtp_ring_mask[l] == 0u && ctx_len > 512u) {
                uint32_t n_splits = 32u;
                struct { uint64_t q, kc, vc, kd, vd, pml, po; uint32_t cl, bs, ns; } pas{
                    msa(moff.Q), mtp_k_addr[l], mtp_v_addr[l], msa(moff.Kd), msa(moff.Vd),
                    sa(off.attn_part_ml), sa(off.attn_part_o), ctx_len, BS, n_splits
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_attn_sk.pipe);
                vkCmdPushConstants(vk.cb, pm_attn_sk.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pas), &pas);
                vkCmdDispatch(vk.cb, N_HEADS, n_splits, 1); vk.barrier();
                struct { uint64_t pml, po, o; uint32_t ns; } par{
                    sa(off.attn_part_ml), sa(off.attn_part_o), msa(moff.attn_o), n_splits
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_attn_red.pipe);
                vkCmdPushConstants(vk.cb, pm_attn_red.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(par), &par);
                vkCmdDispatch(vk.cb, N_HEADS, BS, 1); vk.barrier();
            } else {
                struct { uint64_t q, kc, vc, kd, vd, o; uint32_t cl, bs, rm; } pa{
                    msa(moff.Q), mtp_k_addr[l], mtp_v_addr[l], msa(moff.Kd), msa(moff.Vd), msa(moff.attn_o),
                    ctx_len, BS, mtp_ring_mask[l]
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_attn.pipe);
                vkCmdPushConstants(vk.cb, pm_attn.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pa), &pa);
                vkCmdDispatch(vk.cb, N_HEADS, BS, 1); vk.barrier();
            }

            nbb_mat(mwa(lp + "self_attn.o_proj.weight"), msa(moff.attn_o), msa(moff.oproj), D_MODEL, Q_DIM, BS); vk.barrier();
            nb_elem(msa(moff.X), msa(moff.oproj), msa(moff.X), 0, BS * D_MODEL, 0); vk.barrier();

            nb_rmsnorm(msa(moff.X), mwa(lp + "post_attention_layernorm.weight"), msa(moff.Xn), D_MODEL, BS); vk.barrier();
            nbb_mat(mwa(lp + "mlp.gate_proj.weight"), msa(moff.Xn), msa(moff.gate), MTP_FF, D_MODEL, BS);
            nbb_mat(mwa(lp + "mlp.up_proj.weight"),   msa(moff.Xn), msa(moff.up),   MTP_FF, D_MODEL, BS);
            vk.barrier();
            nb_elem(msa(moff.gate), msa(moff.up), msa(moff.gate), 0, BS * MTP_FF, 5); vk.barrier();
            nbb_mat(mwa(lp + "mlp.down_proj.weight"), msa(moff.gate), msa(moff.down), D_MODEL, MTP_FF, BS); vk.barrier();
            nb_elem(msa(moff.X), msa(moff.down), msa(moff.X), 0, BS * D_MODEL, 0); vk.barrier();
        }
        nb_rmsnorm(msa(moff.X), mwa("norm.weight"), msa(moff.H), D_MODEL, BS); vk.barrier();

        {
            struct { uint64_t a, b, c, d, o, tp; uint32_t m; } gb{
                msa(moff.H) + D_MODEL * 4ULL, mwa("markov_head.hidden_out.bias"), 0, 0, msa(moff.Xn), 0, 2u
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_gru.pipe);
            vkCmdPushConstants(vk.cb, pm_gru.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gb), &gb);
            vkCmdDispatch(vk.cb, n_draft, 1, 1); vk.barrier();
        }
        nbb_mat_w("model.llm.unembed.weight", msa(moff.Xn), sa(off.logits), VOCAB_ACT, D_MODEL, n_draft); vk.barrier();
        {
            struct { uint64_t l, oi, ov; uint32_t v; } pt{sa(off.logits), msa(moff.cand_ids), msa(moff.cand_vals), VOCAB_ACT};
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_top256.pipe);
            vkCmdPushConstants(vk.cb, pm_top256.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pt), &pt);
            vkCmdDispatch(vk.cb, n_draft, 1, 1);
            nb_elem(0, 0, msa(moff.state), 0, 256, 7);
            vk.barrier();
        }
        for (uint32_t k = 0; k < n_draft; ++k) {
            struct { uint64_t a, b, c, d, o, tp; uint32_t m; } g0{
                msa(moff.state), mwa("markov_head.markov_w1.weight"), msa(moff.H) + (uint64_t)(k + 1) * D_MODEL * 4ULL,
                msa(moff.ctx_m), msa(moff.joint), mtp_toks_host.addr + (uint64_t)k * 4ULL, 0u
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_gru.pipe);
            vkCmdPushConstants(vk.cb, pm_gru.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(g0), &g0);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();

            nb_matvec(mwa("markov_head.rnn_proj.weight"), msa(moff.joint), msa(moff.proj), 768, 5120); vk.barrier();

            struct { uint64_t a, b, c, d, o, tp; uint32_t m; } g1{
                msa(moff.proj), mwa("markov_head.rnn_proj.bias"), msa(moff.state), 0, msa(moff.mk), 0, 1u
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_gru.pipe);
            vkCmdPushConstants(vk.cb, pm_gru.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(g1), &g1);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();

            nb_matvec(mwa("markov_head.hidden_out.weight"), msa(moff.mk), msa(moff.dh), D_MODEL, 256); vk.barrier();

            // hidden_out.bias already folded into base_logits!

            uint64_t cids = msa(moff.cand_ids) + (uint64_t)k * 1024ULL * 4ULL;
            uint64_t cvals = msa(moff.cand_vals) + (uint64_t)k * 1024ULL * 4ULL;
            uint64_t otok = mtp_toks_host.addr + (uint64_t)(k + 1) * 4ULL;
            struct { uint64_t u, dh, ci, cv, sc, ot; uint32_t st; } cs0{
                wa("model.llm.unembed.weight"), msa(moff.dh), cids, cvals, msa(moff.scores), otok, 0u
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_cand.pipe);
            vkCmdPushConstants(vk.cb, pm_cand.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cs0), &cs0);
            vkCmdDispatch(vk.cb, 1024, 1, 1); vk.barrier();
            cs0.st = 1u;
            vkCmdPushConstants(vk.cb, pm_cand.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cs0), &cs0);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
        }
        vk.submit_and_wait();
        for (uint32_t k = 0; k < n_draft; ++k) out_draft[k] = (int)mt[k + 1];
    }

    void mtp_post_verify_and_draft(uint32_t acc, uint32_t spec_k, uint32_t pos_start, int anchor_tok, int* out_draft, bool do_draft = true) {
        uint32_t* mt = (uint32_t*)mtp_toks_host.mapped;
        mt[0] = (uint32_t)anchor_tok;
        const uint32_t BS = 8u;
        const uint32_t ctx_len = pos_start + acc;

        vk.begin_cmd();
        // 1. Rollback short_conv hist if acc < spec_k + 1
        if (acc < spec_k + 1u) {
            uint32_t step_idx = acc - 1u;
            std::vector<VkBufferCopy> copies; copies.reserve(N_SLOTS * 4);
            const uint32_t offs[4] = {0, KV_DIM, 2 * KV_DIM, 2 * KV_DIM + D_MODEL};
            const uint32_t cs[4]   = {KV_DIM, KV_DIM, D_MODEL, D_MODEL};
            const uint64_t slot_h  = SCONV_HIST * (2 * KV_DIM + 2 * D_MODEL);
            for (int il = 0; il < N_SLOTS; ++il) {
                for (int k = 0; k < 4; ++k) {
                    uint64_t src_off = ((uint64_t)il * 8ULL * slot_h + 8ULL * SCONV_HIST * offs[k] + (uint64_t)step_idx * SCONV_HIST * cs[k]) * 4ULL;
                    uint64_t dst_off = ((uint64_t)il * slot_h + (uint64_t)SCONV_HIST * offs[k]) * 4ULL;
                    copies.push_back({src_off, dst_off, (uint64_t)SCONV_HIST * cs[k] * 4ULL});
                }
            }
            vkCmdCopyBuffer(vk.cb, hist_steps_buf.buf, hist_buf.buf, (uint32_t)copies.size(), copies.data());
        }
        // 2. Update MTP ctx KV for the `acc` accepted tokens + ctx_m
        for (int l = 0; l < 3; ++l) {
            std::string lp = "layers." + std::to_string(l) + ".";
            struct { uint64_t h0, h1, h2, h3, nw, o; float w0, w1, w2, w3; uint32_t dim; } pf{
                th_addr(0), th_addr(1), th_addr(2), th_addr(3), mwa(lp + "dflare_norm.weight"), msa(moff.ctx_h),
                mtp_fw[l][0], mtp_fw[l][1], mtp_fw[l][2], mtp_fw[l][3], D_MODEL
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_fuse.pipe);
            vkCmdPushConstants(vk.cb, pm_fuse.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pf), &pf);
            vkCmdDispatch(vk.cb, acc, 1, 1); vk.barrier();

            nbb_mat(mwa(lp + "self_attn.k_proj_ctx.weight"), msa(moff.ctx_h), msa(moff.ctx_k), KV_DIM, D_MODEL, acc);
            nbb_mat(mwa(lp + "self_attn.v_proj_ctx.weight"), msa(moff.ctx_h), msa(moff.ctx_v), KV_DIM, D_MODEL, acc);
            vk.barrier();

            struct { uint64_t x, nw, vi, kc, vc; uint32_t nh, ps, rm; } pr{
                msa(moff.ctx_k), mwa(lp + "self_attn.k_norm_ctx.weight"), msa(moff.ctx_v),
                mtp_k_addr[l], mtp_v_addr[l], N_KV_HEADS, pos_start, mtp_ring_mask[l]
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_rope.pipe);
            vkCmdPushConstants(vk.cb, pm_rope.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pr), &pr);
            vkCmdDispatch(vk.cb, N_KV_HEADS, acc, 1); vk.barrier();
        }
        {
            uint64_t last_off = (uint64_t)(acc - 1u) * D_MODEL * 4ULL;
            struct { uint64_t h0, h1, h2, h3, nw, o; float w0, w1, w2, w3; uint32_t dim; } pfm{
                th_addr(0) + last_off, th_addr(1) + last_off, th_addr(2) + last_off, th_addr(3) + last_off,
                mwa("markov_head.dflare_norm.weight"), msa(moff.ctx_m),
                mtp_fw[3][0], mtp_fw[3][1], mtp_fw[3][2], mtp_fw[3][3], D_MODEL
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_fuse.pipe);
            vkCmdPushConstants(vk.cb, pm_fuse.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pfm), &pfm);
            vkCmdDispatch(vk.cb, 1, 1, 1);
        }
        // 3. Draft next block in the exact same command buffer!
        {
            struct { uint64_t e, n, o; uint32_t tid, dim; } pc{
                wa("model.llm.embed.weight"), wa("model.llm.embed_norm.weight"), msa(moff.X), (uint32_t)anchor_tok, D_MODEL
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_embed.pipe);
            vkCmdPushConstants(vk.cb, p_embed.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, 1, 1, 1);
            nb_elem(mwa("mask_embed"), 0, msa(moff.X) + D_MODEL * 4ULL, 0, (BS - 1) * D_MODEL, 6);
            vk.barrier();
        }
        for (int l = 0; l < 3; ++l) {
            std::string lp = "layers." + std::to_string(l) + ".";
            nb_rmsnorm(msa(moff.X), mwa(lp + "input_layernorm.weight"), msa(moff.Xn), D_MODEL, BS); vk.barrier();
            nbb_mat(mwa(lp + "self_attn.q_proj.weight"), msa(moff.Xn), msa(moff.Q), Q_DIM, D_MODEL, BS);
            nbb_mat(mwa(lp + "self_attn.k_proj.weight"), msa(moff.Xn), msa(moff.Kd), KV_DIM, D_MODEL, BS);
            nbb_mat(mwa(lp + "self_attn.v_proj.weight"), msa(moff.Xn), msa(moff.Vd), KV_DIM, D_MODEL, BS);
            vk.barrier();

            struct { uint64_t x, nw, vi, kc, vc; uint32_t nh, ps, rm; } prq{
                msa(moff.Q), mwa(lp + "self_attn.q_norm.weight"), 0, 0, 0, N_HEADS, ctx_len, 0
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_rope.pipe);
            vkCmdPushConstants(vk.cb, pm_rope.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(prq), &prq);
            vkCmdDispatch(vk.cb, N_HEADS, BS, 1);
            struct { uint64_t x, nw, vi, kc, vc; uint32_t nh, ps, rm; } prk{
                msa(moff.Kd), mwa(lp + "self_attn.k_norm.weight"), 0, 0, 0, N_KV_HEADS, ctx_len, 0
            };
            vkCmdPushConstants(vk.cb, pm_rope.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(prk), &prk);
            vkCmdDispatch(vk.cb, N_KV_HEADS, BS, 1);
            vk.barrier();

            if (mtp_ring_mask[l] == 0u && ctx_len > 512u) {
                uint32_t n_splits = 32u;
                struct { uint64_t q, kc, vc, kd, vd, pml, po; uint32_t cl, bs, ns; } pas{
                    msa(moff.Q), mtp_k_addr[l], mtp_v_addr[l], msa(moff.Kd), msa(moff.Vd),
                    sa(off.attn_part_ml), sa(off.attn_part_o), ctx_len, BS, n_splits
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_attn_sk.pipe);
                vkCmdPushConstants(vk.cb, pm_attn_sk.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pas), &pas);
                vkCmdDispatch(vk.cb, N_HEADS, n_splits, 1); vk.barrier();
                struct { uint64_t pml, po, o; uint32_t ns; } par{
                    sa(off.attn_part_ml), sa(off.attn_part_o), msa(moff.attn_o), n_splits
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_attn_red.pipe);
                vkCmdPushConstants(vk.cb, pm_attn_red.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(par), &par);
                vkCmdDispatch(vk.cb, N_HEADS, BS, 1); vk.barrier();
            } else {
                struct { uint64_t q, kc, vc, kd, vd, o; uint32_t cl, bs, rm; } pa{
                    msa(moff.Q), mtp_k_addr[l], mtp_v_addr[l], msa(moff.Kd), msa(moff.Vd), msa(moff.attn_o),
                    ctx_len, BS, mtp_ring_mask[l]
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_attn.pipe);
                vkCmdPushConstants(vk.cb, pm_attn.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pa), &pa);
                vkCmdDispatch(vk.cb, N_HEADS, BS, 1); vk.barrier();
            }

            nbb_mat(mwa(lp + "self_attn.o_proj.weight"), msa(moff.attn_o), msa(moff.oproj), D_MODEL, Q_DIM, BS); vk.barrier();
            nb_elem(msa(moff.X), msa(moff.oproj), msa(moff.X), 0, BS * D_MODEL, 0); vk.barrier();

            nb_rmsnorm(msa(moff.X), mwa(lp + "post_attention_layernorm.weight"), msa(moff.Xn), D_MODEL, BS); vk.barrier();
            nbb_mat(mwa(lp + "mlp.gate_proj.weight"), msa(moff.Xn), msa(moff.gate), MTP_FF, D_MODEL, BS);
            nbb_mat(mwa(lp + "mlp.up_proj.weight"),   msa(moff.Xn), msa(moff.up),   MTP_FF, D_MODEL, BS);
            vk.barrier();
            nb_elem(msa(moff.gate), msa(moff.up), msa(moff.gate), 0, BS * MTP_FF, 5); vk.barrier();
            nbb_mat(mwa(lp + "mlp.down_proj.weight"), msa(moff.gate), msa(moff.down), D_MODEL, MTP_FF, BS); vk.barrier();
            nb_elem(msa(moff.X), msa(moff.down), msa(moff.X), 0, BS * D_MODEL, 0); vk.barrier();
        }
        nb_rmsnorm(msa(moff.X), mwa("norm.weight"), msa(moff.H), D_MODEL, BS); vk.barrier();
        {
            struct { uint64_t a, b, c, d, o, tp; uint32_t m; } gb{
                msa(moff.H) + D_MODEL * 4ULL, mwa("markov_head.hidden_out.bias"), 0, 0, msa(moff.Xn), 0, 2u
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_gru.pipe);
            vkCmdPushConstants(vk.cb, pm_gru.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gb), &gb);
            vkCmdDispatch(vk.cb, spec_k, 1, 1); vk.barrier();
        }
        nbb_mat_w("model.llm.unembed.weight", msa(moff.Xn), sa(off.logits), VOCAB_ACT, D_MODEL, spec_k); vk.barrier();
        {
            struct { uint64_t l, oi, ov; uint32_t v; } pt{sa(off.logits), msa(moff.cand_ids), msa(moff.cand_vals), VOCAB_ACT};
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_top256.pipe);
            vkCmdPushConstants(vk.cb, pm_top256.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pt), &pt);
            vkCmdDispatch(vk.cb, spec_k, 1, 1);
            nb_elem(0, 0, msa(moff.state), 0, 256, 7);
            vk.barrier();
        }
        for (uint32_t k = 0; k < spec_k; ++k) {
            struct { uint64_t a, b, c, d, o, tp; uint32_t m; } g0{
                msa(moff.state), mwa("markov_head.markov_w1.weight"), msa(moff.H) + (uint64_t)(k + 1) * D_MODEL * 4ULL,
                msa(moff.ctx_m), msa(moff.joint), mtp_toks_host.addr + (uint64_t)k * 4ULL, 0u
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_gru.pipe);
            vkCmdPushConstants(vk.cb, pm_gru.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(g0), &g0);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
            nb_matvec(mwa("markov_head.rnn_proj.weight"), msa(moff.joint), msa(moff.proj), 768, 5120); vk.barrier();
            struct { uint64_t a, b, c, d, o, tp; uint32_t m; } g1{
                msa(moff.proj), mwa("markov_head.rnn_proj.bias"), msa(moff.state), 0, msa(moff.mk), 0, 1u
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_gru.pipe);
            vkCmdPushConstants(vk.cb, pm_gru.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(g1), &g1);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
            nb_matvec(mwa("markov_head.hidden_out.weight"), msa(moff.mk), msa(moff.dh), D_MODEL, 256); vk.barrier();
            uint64_t cids = msa(moff.cand_ids) + (uint64_t)k * 1024ULL * 4ULL;
            uint64_t cvals = msa(moff.cand_vals) + (uint64_t)k * 1024ULL * 4ULL;
            uint64_t otok = mtp_toks_host.addr + (uint64_t)(k + 1) * 4ULL;
            struct { uint64_t u, dh, ci, cv, sc, ot; uint32_t st; } cs0{
                wa("model.llm.unembed.weight"), msa(moff.dh), cids, cvals, msa(moff.scores), otok, 0u
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pm_cand.pipe);
            vkCmdPushConstants(vk.cb, pm_cand.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cs0), &cs0);
            vkCmdDispatch(vk.cb, 1024, 1, 1); vk.barrier();
            cs0.st = 1u;
            vkCmdPushConstants(vk.cb, pm_cand.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cs0), &cs0);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
        }
        if (!do_draft) { vk.submit_and_wait(); return; }
        vk.submit_and_wait();
        for (uint32_t k = 0; k < spec_k; ++k) out_draft[k] = (int)mt[k + 1];
    }

    void rollback_hist(uint32_t step_idx) {
        vk.begin_cmd();
        std::vector<VkBufferCopy> copies; copies.reserve(N_SLOTS * 4);
        const uint32_t offs[4] = {0, KV_DIM, 2 * KV_DIM, 2 * KV_DIM + D_MODEL};
        const uint32_t cs[4]   = {KV_DIM, KV_DIM, D_MODEL, D_MODEL};
        const uint64_t slot_h  = SCONV_HIST * (2 * KV_DIM + 2 * D_MODEL);
        for (int il = 0; il < N_SLOTS; ++il) {
            for (int k = 0; k < 4; ++k) {
                uint64_t src_off = ((uint64_t)il * 8ULL * slot_h + 8ULL * SCONV_HIST * offs[k] + (uint64_t)step_idx * SCONV_HIST * cs[k]) * 4ULL;
                uint64_t dst_off = ((uint64_t)il * slot_h + (uint64_t)SCONV_HIST * offs[k]) * 4ULL;
                copies.push_back({src_off, dst_off, (uint64_t)SCONV_HIST * cs[k] * 4ULL});
            }
        }
        vkCmdCopyBuffer(vk.cb, hist_steps_buf.buf, hist_buf.buf, (uint32_t)copies.size(), copies.data());
        vk.submit_and_wait();
    }

    void reset_state() {
        cached_tokens.clear(); ckpt_len = 0;
        vk.begin_cmd();
        vkCmdFillBuffer(vk.cb, hist_buf.buf, 0, hist_buf.size, 0);
        vk.submit_and_wait();
    }

    uint64_t slot_hist_addr(int il, int ch_off) const {
        return hist_buf.addr + (uint64_t)il * SCONV_HIST * (2 * KV_DIM + 2 * D_MODEL) * 4ULL + (uint64_t)SCONV_HIST * ch_off * 4ULL;
    }

    // ---- Single-token helpers (B == 1 fast path) ----
    void nb_rmsnorm(uint64_t in_a, uint64_t w_a, uint64_t out_a, uint32_t dim, uint32_t n_rows = 1) {
        struct { uint64_t i, w, o; uint32_t d; } pc{in_a, w_a, out_a, dim};
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_rmsnorm.pipe);
        vkCmdPushConstants(vk.cb, p_rmsnorm.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, n_rows, 1, 1);
    }
    void nb_matvec(uint64_t w_a, uint64_t x_a, uint64_t y_a, uint32_t rows, uint32_t cols) {
        struct { uint64_t w, x, y; uint32_t r, c; } pc{w_a, x_a, y_a, rows, cols};
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_matvec.pipe);
        vkCmdPushConstants(vk.cb, p_matvec.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, (rows + 3) / 4, 1, 1);
    }
    void nb_matvec_w(const std::string& nm, uint64_t x_a, uint64_t y_a, uint32_t rows, uint32_t cols) {
        if (use_fp8) {
            auto it = fp8_woff.find(nm);
            if (it != fp8_woff.end()) {
                struct { uint64_t w, s, x, y; uint32_t r, c; } pc{
                    fp8_slab.addr + it->second, fp8_slab.addr + fp8_soff.at(nm), x_a, y_a, rows, cols
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_matvec_fp8.pipe);
                vkCmdPushConstants(vk.cb, p_matvec_fp8.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(vk.cb, (rows + 3) / 4, 1, 1);
                return;
            }
        }
        nb_matvec(wa(nm), x_a, y_a, rows, cols);
    }
    void nb_attn_res(const std::vector<uint64_t>& blocks, uint64_t cur_a, uint64_t q_a, uint64_t out_a, uint32_t nq,
                     uint64_t nw0 = 0, uint64_t nw1 = 0, uint64_t nw2 = 0) {
        struct { uint64_t st[8]; uint64_t q, o; uint64_t nw[3]; uint32_t m, nq, dim, do_norm; } pc{};
        for (size_t i = 0; i < blocks.size(); ++i) pc.st[i] = blocks[i];
        pc.st[blocks.size()] = cur_a; pc.q = q_a; pc.o = out_a;
        pc.nw[0] = nw0; pc.nw[1] = nw1; pc.nw[2] = nw2;
        pc.m = (uint32_t)blocks.size() + 1; pc.nq = nq; pc.dim = D_MODEL; pc.do_norm = (nw0 != 0) ? 1u : 0u;
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_attn_res.pipe);
        vkCmdPushConstants(vk.cb, p_attn_res.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, nq, 1, 1);
    }
    void nb_sconv(uint64_t x_a, uint64_t h_a, uint64_t w_a, uint64_t dyn_a, uint64_t out_a, uint32_t C, uint32_t rev, uint64_t res_a = 0) {
        struct { uint64_t x, h, w, d, r, o; uint32_t C, rev; } pc{x_a, h_a, w_a, dyn_a, res_a, out_a, C, rev};
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_sconv.pipe);
        vkCmdPushConstants(vk.cb, p_sconv.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, (C + 63) / 64, 1, 1);
    }
    void nb_l2clip(uint64_t x_a, uint64_t s_a, uint32_t n_heads, uint32_t dim, float thresh, float pmul) {
        struct { uint64_t x, s; uint32_t d; float t, p; } pc{x_a, s_a, dim, thresh, pmul};
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_l2clip.pipe);
        vkCmdPushConstants(vk.cb, p_l2clip.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, n_heads, 1, 1);
    }
    void nb_elem(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint32_t n, uint32_t mode) {
        struct { uint64_t a, b, c, d; uint32_t n, m; } pc{a, b, c, d, n, mode};
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_elem.pipe);
        vkCmdPushConstants(vk.cb, p_elem.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, (n + 63) / 64, 1, 1);
    }

    // ---- Batched helpers (B > 1) ----
    void nbb_mat_w(const std::string& nm, uint64_t x_a, uint64_t y_a, uint32_t rows, uint32_t cols, uint32_t B) {
        if (use_fp8 && B < 32u && (cols % 16u == 0u)) {
            auto it = fp8_woff.find(nm);
            if (it != fp8_woff.end()) {
                struct { uint64_t w, s, x, y; uint32_t r, c, B; } pc{
                    fp8_slab.addr + it->second, fp8_slab.addr + fp8_soff.at(nm), x_a, y_a, rows, cols, B
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_mat_fp8.pipe);
                vkCmdPushConstants(vk.cb, pb_mat_fp8.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(vk.cb, (rows + 3) / 4, (B + 7) / 8, 1);
                return;
            }
        }
        nbb_mat(wa(nm), x_a, y_a, rows, cols, B);
    }
    void nbb_mat(uint64_t w_a, uint64_t x_a, uint64_t y_a, uint32_t rows, uint32_t cols, uint32_t B) {
        struct { uint64_t w, x, y; uint32_t r, c, B; } pc{w_a, x_a, y_a, rows, cols, B};
        if (B >= 32 && (cols % 32u == 0u)) {
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_gemm.pipe);
            vkCmdPushConstants(vk.cb, pb_gemm.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, (rows + 63) / 64, (B + 63) / 64, 1);
        } else {
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_mat.pipe);
            vkCmdPushConstants(vk.cb, pb_mat.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, (rows + 3) / 4, (B + 7) / 8, 1);
        }
    }
    void nbb_attn_res(const std::vector<uint64_t>& blocks, uint64_t cur_a, uint64_t q_a, uint64_t out_a, uint32_t nq, uint32_t B,
                      uint64_t nw0 = 0, uint64_t nw1 = 0, uint64_t nw2 = 0) {
        struct { uint64_t st[8]; uint64_t q, o; uint64_t nw[3]; uint32_t m, nq, dim, B, do_norm; } pc{};
        for (size_t i = 0; i < blocks.size(); ++i) pc.st[i] = blocks[i];
        pc.st[blocks.size()] = cur_a; pc.q = q_a; pc.o = out_a;
        pc.nw[0] = nw0; pc.nw[1] = nw1; pc.nw[2] = nw2;
        pc.m = (uint32_t)blocks.size() + 1; pc.nq = nq; pc.dim = D_MODEL; pc.B = B; pc.do_norm = (nw0 != 0) ? 1u : 0u;
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_attn_res.pipe);
        vkCmdPushConstants(vk.cb, pb_attn_res.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, nq, B, 1);
    }
    void nbb_sconv(uint64_t x_a, uint64_t h_a, uint64_t w_a, uint64_t dyn_a, uint64_t out_a, uint32_t C, uint32_t B, uint32_t rev, uint64_t hs_a = 0) {
        struct { uint64_t x, h, w, d, o, hs; uint32_t C, B, rev; } pc{x_a, h_a, w_a, dyn_a, out_a, hs_a, C, B, rev};
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_sconv.pipe);
        vkCmdPushConstants(vk.cb, pb_sconv.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, (C + 63) / 64, 1, 1);
    }
    void nbb_l2clip(uint64_t x_a, uint64_t s_a, uint32_t n_heads, uint32_t dim, uint32_t pos_start, uint32_t is_glob, float thresh, uint32_t B) {
        struct { uint64_t x, s; uint32_t nh, d, ps, ig; float t; } pc{x_a, s_a, n_heads, dim, pos_start, is_glob, thresh};
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_l2clip.pipe);
        vkCmdPushConstants(vk.cb, pb_l2clip.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, n_heads, B, 1);
    }
    void nbb_kv_store(int il, uint32_t pos_start, uint32_t B) {
        struct { uint64_t ki, vi, kc, vc; uint32_t ps, mask; } pc{
            sa(off.K), sa(off.V), slot_k_addr[il], slot_v_addr[il], pos_start, slot_ring_mask[il]
        };
        vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_kv_store.pipe);
        vkCmdPushConstants(vk.cb, pb_kv_store.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(vk.cb, KV_DIM / 64, B, 1);
    }

    // ---------------- Single-token Layer (B == 1) ----------------
    void record_layer_1(int il, int ip, int pos, float tau, int& cur_slot, std::vector<uint64_t>& blocks, bool has_msg,
                        bool prof = false, uint32_t* qidx = nullptr, std::vector<std::pair<int,std::pair<uint32_t,uint32_t>>>* spans = nullptr) {
        auto tb = [&]() -> uint32_t { if (!prof) return 0; uint32_t id = (*qidx)++; vkCmdWriteTimestamp(vk.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, vk.qp, id); return id; };
        auto te = [&](int cat, uint32_t s) { if (!prof) return; uint32_t e = (*qidx)++; vkCmdWriteTimestamp(vk.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk.qp, e); spans->push_back({cat, {s, e}}); };
        const bool is_glob = is_global_layer(ip);
        const std::string lp = "model.llm.layers." + std::to_string(ip) + ".";
        const uint64_t cur_off = off.cur[cur_slot], cur_a = sa(cur_off);

        uint32_t ts1 = tb();
        uint64_t nq_a = sa(off.nq), nk_a = sa(off.nk), nv_a = sa(off.nv);
        if (!has_msg) {
            if (ip == 0) {
                nb_rmsnorm(cur_a, wa(lp + "attn_norm.weight"), nq_a, D_MODEL);
                nk_a = nq_a; nv_a = nq_a; vk.barrier();
            } else if (!is_glob) {
                nb_attn_res(blocks, cur_a, wa("model.llm.attn_res.attn_queries." + std::to_string(ip - 1)), nq_a, 1, wa(lp + "attn_norm.weight"));
                nk_a = nq_a; nv_a = nq_a; vk.barrier();
            } else {
                nb_attn_res(blocks, cur_a, wa("model.llm.attn_res.attn_queries." + std::to_string(ip - 1)), sa(off.mixed), 3,
                            wa(lp + "q_input_norm.weight"), wa(lp + "k_input_norm.weight"), wa(lp + "v_input_norm.weight"));
                nq_a = sa(off.mixed); nk_a = sa(off.mixed) + D_MODEL * 4; nv_a = sa(off.mixed) + 2 * D_MODEL * 4; vk.barrier();
            }
        } else {
            uint32_t nq = is_glob ? 3 : 1;
            nb_attn_res(blocks, cur_a, wa("model.llm.attn_res.attn_queries." + std::to_string(ip - 1)), sa(off.mixed), nq);
            vk.barrier();
            uint64_t in_q = sa(off.mixed), in_k = is_glob ? sa(off.mixed) + D_MODEL * 4 : in_q, in_v = is_glob ? sa(off.mixed) + 2 * D_MODEL * 4 : in_q;

            struct { uint64_t cur[3], msg[3], dw[3], go[3]; uint32_t dim, rank; } pcd{};
            pcd.cur[0] = in_q; pcd.cur[1] = in_k; pcd.cur[2] = in_v;
            pcd.msg[0] = sa(off.messages); pcd.msg[1] = sa(off.messages) + D_MODEL * 4; pcd.msg[2] = sa(off.messages) + 2 * D_MODEL * 4;
            pcd.dw[0] = wa(lp + "controllers.attn.down_weight"); pcd.dw[1] = wa(lp + "controllers.k.down_weight"); pcd.dw[2] = wa(lp + "controllers.v.down_weight");
            pcd.go[0] = sa(off.ctrl_gate[0]); pcd.go[1] = sa(off.ctrl_gate[1]); pcd.go[2] = sa(off.ctrl_gate[2]);
            pcd.dim = D_MODEL; pcd.rank = GATE_RANK;
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_ctrl_down.pipe);
            vkCmdPushConstants(vk.cb, p_ctrl_down.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcd), &pcd);
            vkCmdDispatch(vk.cb, GATE_RANK, 3, 1); vk.barrier();

            struct { uint64_t cur[3], msg[3], g[3], gw[3], fw[3], o[3]; uint32_t dim, rank; } pca{};
            pca.cur[0] = in_q; pca.cur[1] = in_k; pca.cur[2] = in_v;
            pca.msg[0] = pcd.msg[0]; pca.msg[1] = pcd.msg[1]; pca.msg[2] = pcd.msg[2];
            pca.g[0] = pcd.go[0]; pca.g[1] = pcd.go[1]; pca.g[2] = pcd.go[2];
            pca.gw[0] = wa(lp + "controllers.attn.gain_weight"); pca.fw[0] = wa(lp + "controllers.attn.forget_weight");
            pca.gw[1] = wa(lp + "controllers.k.gain_weight");    pca.fw[1] = wa(lp + "controllers.k.forget_weight");
            pca.gw[2] = wa(lp + "controllers.v.gain_weight");    pca.fw[2] = wa(lp + "controllers.v.forget_weight");
            pca.o[0] = sa(off.ctrl_gain[0]); pca.o[1] = sa(off.ctrl_gain[1]); pca.o[2] = sa(off.ctrl_gain[2]);
            pca.dim = D_MODEL; pca.rank = GATE_RANK;
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_ctrl_apply.pipe);
            vkCmdPushConstants(vk.cb, p_ctrl_apply.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pca), &pca);
            vkCmdDispatch(vk.cb, D_MODEL / 64, 3, 1); vk.barrier();

            if (!is_glob) {
                uint64_t nw = wa(lp + "attn_norm.weight");
                nb_rmsnorm(pca.o[0], nw, nq_a, D_MODEL); nb_rmsnorm(pca.o[1], nw, nk_a, D_MODEL); nb_rmsnorm(pca.o[2], nw, nv_a, D_MODEL);
            } else {
                nb_rmsnorm(pca.o[0], wa(lp + "q_input_norm.weight"), nq_a, D_MODEL);
                nb_rmsnorm(pca.o[1], wa(lp + "k_input_norm.weight"), nk_a, D_MODEL);
                nb_rmsnorm(pca.o[2], wa(lp + "v_input_norm.weight"), nv_a, D_MODEL);
            }
            vk.barrier();
        }

        te(1, ts1); uint32_t ts2 = tb();
        {
            struct {
                uint64_t s_base;
                uint64_t w[8];
                uint32_t nq_o, nk_o, nv_o;
                uint32_t y_o[8];
            } pca{
                scratch.addr,
                { wa(lp + "attn.wq_du.weight"), wa(lp + "attn.wk_dv.weight"), wa(lp + "attn.wv_dv.weight"),
                  wa(lp + "attn.wr_du.weight"), wa(lp + "attn.attn_sink_proj.weight"), wa(lp + "attn.gate_attn.weight"),
                  is_glob ? wa(lp + "attn.k_sconv.weight_proj") : 0ULL, is_glob ? wa(lp + "attn.v_sconv.weight_proj") : 0ULL },
                (uint32_t)(nq_a - scratch.addr), (uint32_t)(nk_a - scratch.addr), (uint32_t)(nv_a - scratch.addr),
                { (uint32_t)off.Q, (uint32_t)off.K_raw, (uint32_t)off.V_raw, (uint32_t)off.R,
                  (uint32_t)off.sinks, (uint32_t)off.gate_a, (uint32_t)off.dyn_k, (uint32_t)off.dyn_v }
            };
            if (use_fp8) {
                struct {
                    uint64_t s_base, fp8_base;
                    uint32_t w_o[8], sc_o[8];
                    uint32_t nq_o, nk_o, nv_o;
                    uint32_t y_o[8];
                } pcf{
                    scratch.addr, fp8_slab.addr,
                    { fp8_woff.at(lp + "attn.wq_du.weight"), fp8_woff.at(lp + "attn.wk_dv.weight"), fp8_woff.at(lp + "attn.wv_dv.weight"),
                      fp8_woff.at(lp + "attn.wr_du.weight"), fp8_woff.at(lp + "attn.attn_sink_proj.weight"), fp8_woff.at(lp + "attn.gate_attn.weight"),
                      is_glob ? fp8_woff.at(lp + "attn.k_sconv.weight_proj") : 0u, is_glob ? fp8_woff.at(lp + "attn.v_sconv.weight_proj") : 0u },
                    { fp8_soff.at(lp + "attn.wq_du.weight"), fp8_soff.at(lp + "attn.wk_dv.weight"), fp8_soff.at(lp + "attn.wv_dv.weight"),
                      fp8_soff.at(lp + "attn.wr_du.weight"), fp8_soff.at(lp + "attn.attn_sink_proj.weight"), fp8_soff.at(lp + "attn.gate_attn.weight"),
                      is_glob ? fp8_soff.at(lp + "attn.k_sconv.weight_proj") : 0u, is_glob ? fp8_soff.at(lp + "attn.v_sconv.weight_proj") : 0u },
                    (uint32_t)(nq_a - scratch.addr), (uint32_t)(nk_a - scratch.addr), (uint32_t)(nv_a - scratch.addr),
                    { (uint32_t)off.Q, (uint32_t)off.K_raw, (uint32_t)off.V_raw, (uint32_t)off.R,
                      (uint32_t)off.sinks, (uint32_t)off.gate_a, (uint32_t)off.dyn_k, (uint32_t)off.dyn_v }
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_matvec_fp8_attn.pipe);
                vkCmdPushConstants(vk.cb, p_matvec_fp8_attn.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 124, &pcf);
                vkCmdDispatch(vk.cb, (is_glob ? 8664u : 8600u) / 4u, 1, 1);
            } else {
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_matvec_attn.pipe);
                vkCmdPushConstants(vk.cb, p_matvec_attn.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 116, &pca);
                vkCmdDispatch(vk.cb, (is_glob ? 8664u : 8600u) / 4u, 1, 1);
            }
        }
        vk.barrier();

        if (is_glob) {
            nb_sconv(sa(off.K_raw), slot_hist_addr(il, 0),      wa(lp + "attn.k_sconv.static_w"), sa(off.dyn_k), sa(off.K), KV_DIM, 1);
            nb_sconv(sa(off.V_raw), slot_hist_addr(il, KV_DIM), wa(lp + "attn.v_sconv.static_w"), sa(off.dyn_v), sa(off.V), KV_DIM, 1);
        } else {
            nb_sconv(sa(off.K_raw), slot_hist_addr(il, 0),      wa(lp + "attn.k_sconv.weight"), 0, sa(off.K), KV_DIM, 0);
            nb_sconv(sa(off.V_raw), slot_hist_addr(il, KV_DIM), wa(lp + "attn.v_sconv.weight"), 0, sa(off.V), KV_DIM, 0);
        }
        nb_l2clip(sa(off.Q), wa(lp + "attn.q_l2_clip.scale"),   N_HEADS, HEAD_DIM, QK_CLIP,  is_glob ? tau : 1.0f);
        nb_l2clip(sa(off.R), wa(lp + "attn.rel_l2_clip.scale"), N_HEADS, D_REL,    REL_CLIP, is_glob ? tau : 1.0f);
        vk.barrier();

        nb_l2clip(sa(off.K), 0, N_KV_HEADS, HEAD_DIM, QK_CLIP, 1.0f);
        nb_rmsnorm(sa(off.V), 0, sa(off.V), HEAD_DIM, N_KV_HEADS);
        vk.barrier();

        nbb_kv_store(il, (uint32_t)pos, 1);
        vk.barrier();

        te(2, ts2); uint32_t ts3 = tb();
        {
            uint32_t start_j = is_glob ? 0u : (uint32_t)std::max(0, pos - SWA_WINDOW + 1);
            uint32_t n_kv = (uint32_t)pos - start_j + 1u;
            uint32_t n_splits = (n_kv <= 8u) ? 1u : ((n_kv <= 32u) ? 8u : ((n_kv <= 2048u) ? 16u : 64u));
            uint64_t rpt_addr = rel_proj_t_buf.addr + (uint64_t)ip * 1024ULL * 16ULL * 2ULL;
            struct { uint64_t q, r, rp, kc, vc, pml, po, s, g, o; uint32_t pos, sj, ns, mask; } pc{
                sa(off.Q), sa(off.R), rpt_addr,
                slot_k_addr[il], slot_v_addr[il], sa(off.attn_part_ml), sa(off.attn_part_o),
                sa(off.sinks), sa(off.gate_a), sa(off.attn_raw),
                (uint32_t)pos, start_j, n_splits, slot_ring_mask[il]
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_attn.pipe);
            vkCmdPushConstants(vk.cb, p_attn.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, N_HEADS, n_splits, 1); vk.barrier();

            if (n_splits > 1u) {
                struct { uint64_t pml, po, s, g, o; uint32_t ns; } pcr{
                    sa(off.attn_part_ml), sa(off.attn_part_o), sa(off.sinks), sa(off.gate_a), sa(off.attn_raw), n_splits
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_attn_red.pipe);
                vkCmdPushConstants(vk.cb, p_attn_red.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcr), &pcr);
                vkCmdDispatch(vk.cb, N_HEADS, 1, 1); vk.barrier();
            }
        }

        te(3, ts3); uint32_t ts2b = tb();
        nb_matvec_w(lp + "attn.wo_ud.weight", sa(off.attn_raw), sa(off.wo_out), D_MODEL, Q_DIM); vk.barrier();
        uint64_t ffn_inp_a;
        if (ip % BLOCK_SIZE == 0) {
            nb_sconv(sa(off.wo_out), slot_hist_addr(il, 2 * KV_DIM), wa(lp + "attn_sconv.weight"), 0, sa(off.attn_out), D_MODEL, 0, 0);
            int bidx = (int)blocks.size();
            nb_elem(cur_a, 0, sa(off.blocks[bidx]), 0, D_MODEL, 4);
            blocks.push_back(sa(off.blocks[bidx]));
            ffn_inp_a = sa(off.attn_out);
            vk.barrier();
        } else {
            nb_sconv(sa(off.wo_out), slot_hist_addr(il, 2 * KV_DIM), wa(lp + "attn_sconv.weight"), 0, sa(off.ffn_inp), D_MODEL, 0, cur_a);
            ffn_inp_a = sa(off.ffn_inp);
            vk.barrier();
        }

        te(2, ts2b); uint32_t ts1b = tb();
        if (!has_msg) {
            nb_attn_res(blocks, ffn_inp_a, wa("model.llm.attn_res.mlp_queries." + std::to_string(ip)), sa(off.xm_norm), 1, wa(lp + "mlp_norm.weight"));
            vk.barrier();
        } else {
            nb_attn_res(blocks, ffn_inp_a, wa("model.llm.attn_res.mlp_queries." + std::to_string(ip)), sa(off.xm), 1); vk.barrier();
            struct { uint64_t cur[3], msg[3], dw[3], go[3]; uint32_t dim, rank; } pcd{};
            pcd.cur[0] = sa(off.xm); pcd.msg[0] = sa(off.messages) + 3 * D_MODEL * 4;
            pcd.dw[0] = wa(lp + "controllers.mlp.down_weight"); pcd.go[0] = sa(off.ctrl_gate[0]);
            pcd.dim = D_MODEL; pcd.rank = GATE_RANK;
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_ctrl_down.pipe);
            vkCmdPushConstants(vk.cb, p_ctrl_down.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcd), &pcd);
            vkCmdDispatch(vk.cb, GATE_RANK, 1, 1); vk.barrier();

            struct { uint64_t cur[3], msg[3], g[3], gw[3], fw[3], o[3]; uint32_t dim, rank; } pca{};
            pca.cur[0] = sa(off.xm); pca.msg[0] = pcd.msg[0]; pca.g[0] = pcd.go[0];
            pca.gw[0] = wa(lp + "controllers.mlp.gain_weight"); pca.fw[0] = wa(lp + "controllers.mlp.forget_weight");
            pca.o[0] = sa(off.ctrl_gain[0]); pca.dim = D_MODEL; pca.rank = GATE_RANK;
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_ctrl_apply.pipe);
            vkCmdPushConstants(vk.cb, p_ctrl_apply.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pca), &pca);
            vkCmdDispatch(vk.cb, D_MODEL / 64, 1, 1); vk.barrier();
            nb_rmsnorm(pca.o[0], wa(lp + "mlp_norm.weight"), sa(off.xm_norm), D_MODEL); vk.barrier();
        }

        te(1, ts1b); uint32_t ts4 = tb();
        nb_matvec(wa(lp + "mlp.gate.weight"), sa(off.xm_norm), sa(off.r_logits), N_EXPERTS, D_MODEL); vk.barrier();
        {
            struct { uint64_t l, b, ids, w; } pc{sa(off.r_logits), wa(lp + "mlp.gate.bias"), sa(off.top_ids), sa(off.top_w)};
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_router.pipe);
            vkCmdPushConstants(vk.cb, p_router.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
        }

        te(4, ts4); uint32_t ts5 = tb();
        if (ip < 20) {
            struct { uint64_t w, s, amax, den, ids, topw, x, out; uint32_t r, c, mode; } pc13{
                wa(lp + "mlp.experts.w13_weight"), wa(lp + "mlp.experts.w13_weight_scale"),
                wa(lp + "mlp.experts.w13_weight_amax"), wa(lp + "mlp.experts.w13_weight_outer_scale_denominator"),
                sa(off.top_ids), sa(off.top_w), sa(off.xm_norm), sa(off.act_buf), 2 * INTER_SIZE, D_MODEL, 0
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_nvfp4.pipe);
            vkCmdPushConstants(vk.cb, p_nvfp4.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc13), &pc13);
            vkCmdDispatch(vk.cb, INTER_SIZE / 4, TOP_K, 1); vk.barrier();
            struct { uint64_t w, s, amax, den, ids, topw, x, out; uint32_t r, c, mode; } pc2{
                wa(lp + "mlp.experts.w2_weight"), wa(lp + "mlp.experts.w2_weight_scale"),
                wa(lp + "mlp.experts.w2_weight_amax"), wa(lp + "mlp.experts.w2_weight_outer_scale_denominator"),
                sa(off.top_ids), sa(off.top_w), sa(off.act_buf), sa(off.moe_raw), D_MODEL, INTER_SIZE, 1
            };
            vkCmdPushConstants(vk.cb, p_nvfp4.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
            vkCmdDispatch(vk.cb, D_MODEL / 4, 1, 1); vk.barrier();
        } else {
            struct { uint64_t w, s, ids, topw, x, out; uint32_t r, c, mode; } pc13{
                wa(lp + "mlp.experts.w13_weight"), wa(lp + "mlp.experts.w13_weight_scale"),
                sa(off.top_ids), sa(off.top_w), sa(off.xm_norm), sa(off.act_buf), 2 * INTER_SIZE, D_MODEL, 0
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_mxfp8.pipe);
            vkCmdPushConstants(vk.cb, p_mxfp8.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc13), &pc13);
            vkCmdDispatch(vk.cb, INTER_SIZE / 4, TOP_K, 1); vk.barrier();
            struct { uint64_t w, s, ids, topw, x, out; uint32_t r, c, mode; } pc2{
                wa(lp + "mlp.experts.w2_weight"), wa(lp + "mlp.experts.w2_weight_scale"),
                sa(off.top_ids), sa(off.top_w), sa(off.act_buf), sa(off.moe_raw), D_MODEL, INTER_SIZE, 1
            };
            vkCmdPushConstants(vk.cb, p_mxfp8.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
            vkCmdDispatch(vk.cb, D_MODEL / 4, 1, 1); vk.barrier();
        }
        int next_slot = cur_slot ^ 1;
        nb_sconv(sa(off.moe_raw), slot_hist_addr(il, 2 * KV_DIM + D_MODEL), wa(lp + "mlp_sconv.weight"), 0, sa(off.cur[next_slot]), D_MODEL, 0, ffn_inp_a);
        vk.barrier();
        cur_slot = next_slot;
        te(5, ts5);
        if (mtp_loaded) {
            int th_s = (il == 4) ? 0 : ((il == 28) ? 1 : ((il == 46) ? 2 : -1));
            if (th_s >= 0) { nb_elem(sa(off.cur[cur_slot]), 0, th_addr(th_s), 0, D_MODEL, 4); vk.barrier(); }
        }
    }

    int forward_one_token(int token_id, int pos, bool compute_logits, double* prof8 = nullptr) {
        bool prof = (prof8 != nullptr); uint32_t qidx = 0;
        std::vector<std::pair<int,std::pair<uint32_t,uint32_t>>> spans;
        vk.begin_cmd();
        if (prof) vkCmdResetQueryPool(vk.cb, vk.qp, 0, 1024);
        uint32_t tall0 = 0; if (prof) { tall0 = qidx++; vkCmdWriteTimestamp(vk.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, vk.qp, tall0); }
        int cur_slot = 0;
        {
            struct { uint64_t e, n, o; uint32_t tid, dim; } pc{
                wa("model.llm.embed.weight"), wa("model.llm.embed_norm.weight"), sa(off.cur[0]), (uint32_t)token_id, D_MODEL
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_embed.pipe);
            vkCmdPushConstants(vk.cb, p_embed.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
        }
        float tau = 1.0f + LOG_SCALE_ALPHA * std::log(std::max(1.0f, (float)(pos + 1) / LOG_SCALE_FLOOR));
        std::vector<uint64_t> blocks;

        for (int il = 0; il < CORE_START; ++il) record_layer_1(il, il, pos, tau, cur_slot, blocks, false, prof, &qidx, &spans);
        { nb_elem(sa(off.cur[cur_slot]), 0, sa(off.prelude), 0, D_MODEL, 4); vk.barrier(); }
        const auto prelude_blocks = blocks;

        for (int pass = 0; pass < N_PASSES; ++pass) {
            std::vector<uint64_t> bank;
            if (pass > 0) {
                int dst_slot = cur_slot ^ 1;
                nb_elem(sa(off.blocks[prelude_blocks.size() + 1]), 0, sa(off.bank_block), 0, D_MODEL, 4);
                nb_elem(sa(off.cur[cur_slot]), 0, sa(off.bank_out), 0, D_MODEL, 4);
                nb_elem(sa(off.prelude), 0, sa(off.cur[dst_slot]), 0, D_MODEL, 4);
                vk.barrier();
                cur_slot = dst_slot; bank.push_back(sa(off.bank_block)); blocks = prelude_blocks;
            }
            for (int ip = CORE_START; ip < CORE_END; ++ip) {
                int il = (pass == 0) ? ip : N_PHYS + (pass - 1) * (CORE_END - CORE_START) + (ip - CORE_START);
                if (pass == 0) record_layer_1(il, ip, pos, tau, cur_slot, blocks, false, prof, &qidx, &spans);
                else {
                    uint64_t mq_addr = msg_q_buf.addr + (uint64_t)(ip - CORE_START) * 4ULL * D_MODEL * 4ULL;
                    nb_attn_res(bank, sa(off.bank_out), mq_addr, sa(off.messages), 4); vk.barrier();
                    record_layer_1(il, ip, pos, tau, cur_slot, blocks, true, prof, &qidx, &spans);
                }
            }
        }
        for (int il = CORE_END; il < N_PHYS; ++il) record_layer_1(il, il, pos, tau, cur_slot, blocks, false, prof, &qidx, &spans);

        if (compute_logits || mtp_loaded) {
            uint32_t tu0 = 0; if (prof) { tu0 = qidx++; vkCmdWriteTimestamp(vk.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, vk.qp, tu0); }
            nb_attn_res(blocks, sa(off.cur[cur_slot]), wa("model.llm.attn_res.output_query"), sa(off.xm_norm), 1, wa("model.llm.norm.weight")); vk.barrier();
            if (mtp_loaded) { nb_elem(sa(off.xm_norm), 0, th_addr(3), 0, D_MODEL, 4); vk.barrier(); }
            if (compute_logits) {
                nb_matvec_w("model.llm.unembed.weight", sa(off.xm_norm), sa(off.logits), VOCAB_ACT, D_MODEL); vk.barrier();
            uint64_t pv = sa(off.y_exps), pi = sa(off.y_exps) + 256 * 4;
            struct { uint64_t l, pv, pi, o; uint32_t n, st; } pc0{sa(off.logits), pv, pi, out_ids_host.addr, VOCAB_ACT, 0};
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_argmax.pipe);
            vkCmdPushConstants(vk.cb, p_argmax.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc0), &pc0);
            vkCmdDispatch(vk.cb, 256, 1, 1); vk.barrier();
            struct { uint64_t l, pv, pi, o; uint32_t n, st; } pc1{sa(off.logits), pv, pi, out_ids_host.addr, 256, 1};
            vkCmdPushConstants(vk.cb, p_argmax.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
            if (prof) { uint32_t tu1 = qidx++; vkCmdWriteTimestamp(vk.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk.qp, tu1); spans.push_back({6, {tu0, tu1}}); }
            }
        }
        uint32_t tall1 = 0; if (prof) { tall1 = qidx++; vkCmdWriteTimestamp(vk.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk.qp, tall1); }
        vk.submit_and_wait();
        if (prof) {
            std::vector<uint64_t> ts(qidx); VK_CHECK(vkGetQueryPoolResults(vk.dev, vk.qp, 0, qidx, qidx*8, ts.data(), 8, VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
            double sc = (double)vk.ts_period_ns * 1e-6;
            prof8[7] += (ts[tall1] - ts[tall0]) * sc;
            for (auto& [cat, pr] : spans) prof8[cat] += (ts[pr.second] - ts[pr.first]) * sc;
        }
        if (!compute_logits) return -1;
        return (int)(*(const uint32_t*)out_ids_host.mapped);
    }

    // ---------------- Batched Layer (B = 1..64) ----------------
    uint64_t slot_hist_step_addr(int il, uint32_t chan_off, bool save) const {
        if (!save) return 0;
        const uint64_t slot_floats = 8ULL * SCONV_HIST * (2 * KV_DIM + 2 * D_MODEL);
        return hist_steps_buf.addr + ((uint64_t)il * slot_floats + 8ULL * SCONV_HIST * (uint64_t)chan_off) * 4ULL;
    }
    void record_layer_batch(int il, int ip, uint32_t pos_start, uint32_t B, int& cur_slot, std::vector<uint64_t>& blocks, bool has_msg, bool save_hs = false) {
        const bool is_glob = is_global_layer(ip);
        const std::string lp = "model.llm.layers." + std::to_string(ip) + ".";
        const uint64_t cur_off = off.cur[cur_slot], cur_a = sa(cur_off);
        const uint64_t b_stride = (uint64_t)B * D_MODEL * 4ULL;

        uint64_t nq_a = sa(off.nq), nk_a = sa(off.nk), nv_a = sa(off.nv);
        if (!has_msg) {
            if (ip == 0) {
                nb_rmsnorm(cur_a, wa(lp + "attn_norm.weight"), nq_a, D_MODEL, B);
                nk_a = nq_a; nv_a = nq_a; vk.barrier();
            } else if (!is_glob) {
                nbb_attn_res(blocks, cur_a, wa("model.llm.attn_res.attn_queries." + std::to_string(ip - 1)), nq_a, 1, B, wa(lp + "attn_norm.weight"));
                nk_a = nq_a; nv_a = nq_a; vk.barrier();
            } else {
                nbb_attn_res(blocks, cur_a, wa("model.llm.attn_res.attn_queries." + std::to_string(ip - 1)), sa(off.mixed), 3, B,
                             wa(lp + "q_input_norm.weight"), wa(lp + "k_input_norm.weight"), wa(lp + "v_input_norm.weight"));
                nq_a = sa(off.mixed); nk_a = sa(off.mixed) + b_stride; nv_a = sa(off.mixed) + 2 * b_stride; vk.barrier();
            }
        } else {
            uint32_t nq = is_glob ? 3 : 1;
            nbb_attn_res(blocks, cur_a, wa("model.llm.attn_res.attn_queries." + std::to_string(ip - 1)), sa(off.mixed), nq, B);
            vk.barrier();
            uint64_t in_q = sa(off.mixed), in_k = is_glob ? sa(off.mixed) + b_stride : in_q, in_v = is_glob ? sa(off.mixed) + 2 * b_stride : in_q;

            struct { uint64_t cur[3], msg[3], dw[3], go[3]; uint32_t dim, rank; } pcd{};
            pcd.cur[0] = in_q; pcd.cur[1] = in_k; pcd.cur[2] = in_v;
            pcd.msg[0] = sa(off.messages); pcd.msg[1] = sa(off.messages) + b_stride; pcd.msg[2] = sa(off.messages) + 2 * b_stride;
            pcd.dw[0] = wa(lp + "controllers.attn.down_weight"); pcd.dw[1] = wa(lp + "controllers.k.down_weight"); pcd.dw[2] = wa(lp + "controllers.v.down_weight");
            pcd.go[0] = sa(off.ctrl_gate[0]); pcd.go[1] = sa(off.ctrl_gate[1]); pcd.go[2] = sa(off.ctrl_gate[2]);
            pcd.dim = D_MODEL; pcd.rank = GATE_RANK;
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_ctrl_down.pipe);
            vkCmdPushConstants(vk.cb, pb_ctrl_down.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcd), &pcd);
            vkCmdDispatch(vk.cb, GATE_RANK, 3, B); vk.barrier();

            struct { uint64_t cur[3], msg[3], g[3], gw[3], fw[3], o[3]; uint32_t dim, rank; } pca{};
            pca.cur[0] = in_q; pca.cur[1] = in_k; pca.cur[2] = in_v;
            pca.msg[0] = pcd.msg[0]; pca.msg[1] = pcd.msg[1]; pca.msg[2] = pcd.msg[2];
            pca.g[0] = pcd.go[0]; pca.g[1] = pcd.go[1]; pca.g[2] = pcd.go[2];
            pca.gw[0] = wa(lp + "controllers.attn.gain_weight"); pca.fw[0] = wa(lp + "controllers.attn.forget_weight");
            pca.gw[1] = wa(lp + "controllers.k.gain_weight");    pca.fw[1] = wa(lp + "controllers.k.forget_weight");
            pca.gw[2] = wa(lp + "controllers.v.gain_weight");    pca.fw[2] = wa(lp + "controllers.v.forget_weight");
            pca.o[0] = sa(off.ctrl_gain[0]); pca.o[1] = sa(off.ctrl_gain[1]); pca.o[2] = sa(off.ctrl_gain[2]);
            pca.dim = D_MODEL; pca.rank = GATE_RANK;
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_ctrl_apply.pipe);
            vkCmdPushConstants(vk.cb, pb_ctrl_apply.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pca), &pca);
            vkCmdDispatch(vk.cb, D_MODEL / 64, 3, B); vk.barrier();

            if (!is_glob) {
                uint64_t nw = wa(lp + "attn_norm.weight");
                nb_rmsnorm(pca.o[0], nw, nq_a, D_MODEL, B); nb_rmsnorm(pca.o[1], nw, nk_a, D_MODEL, B); nb_rmsnorm(pca.o[2], nw, nv_a, D_MODEL, B);
            } else {
                nb_rmsnorm(pca.o[0], wa(lp + "q_input_norm.weight"), nq_a, D_MODEL, B);
                nb_rmsnorm(pca.o[1], wa(lp + "k_input_norm.weight"), nk_a, D_MODEL, B);
                nb_rmsnorm(pca.o[2], wa(lp + "v_input_norm.weight"), nv_a, D_MODEL, B);
            }
            vk.barrier();
        }

        nbb_mat_w(lp + "attn.wq_du.weight", nq_a, sa(off.Q), Q_DIM, D_MODEL, B);
        nbb_mat_w(lp + "attn.wk_dv.weight", nk_a, sa(off.K_raw), KV_DIM, D_MODEL, B);
        nbb_mat_w(lp + "attn.wv_dv.weight", nv_a, sa(off.V_raw), KV_DIM, D_MODEL, B);
        nbb_mat_w(lp + "attn.wr_du.weight", nq_a, sa(off.R), N_HEADS * D_REL, D_MODEL, B);
        nbb_mat_w(lp + "attn.attn_sink_proj.weight", nq_a, sa(off.sinks), N_HEADS, D_MODEL, B);
        nbb_mat_w(lp + "attn.gate_attn.weight", nq_a, sa(off.gate_a), Q_DIM, D_MODEL, B);
        if (is_glob) {
            nbb_mat_w(lp + "attn.k_sconv.weight_proj", nk_a, sa(off.dyn_k), 32, D_MODEL, B);
            nbb_mat_w(lp + "attn.v_sconv.weight_proj", nv_a, sa(off.dyn_v), 32, D_MODEL, B);
        }
        vk.barrier();

        if (is_glob) {
            nbb_sconv(sa(off.K_raw), slot_hist_addr(il, 0),      wa(lp + "attn.k_sconv.static_w"), sa(off.dyn_k), sa(off.K), KV_DIM, B, 1, slot_hist_step_addr(il, 0, save_hs));
            nbb_sconv(sa(off.V_raw), slot_hist_addr(il, KV_DIM), wa(lp + "attn.v_sconv.static_w"), sa(off.dyn_v), sa(off.V), KV_DIM, B, 1, slot_hist_step_addr(il, KV_DIM, save_hs));
        } else {
            nbb_sconv(sa(off.K_raw), slot_hist_addr(il, 0),      wa(lp + "attn.k_sconv.weight"), 0, sa(off.K), KV_DIM, B, 0, slot_hist_step_addr(il, 0, save_hs));
            nbb_sconv(sa(off.V_raw), slot_hist_addr(il, KV_DIM), wa(lp + "attn.v_sconv.weight"), 0, sa(off.V), KV_DIM, B, 0, slot_hist_step_addr(il, KV_DIM, save_hs));
        }
        nbb_l2clip(sa(off.Q), wa(lp + "attn.q_l2_clip.scale"),   N_HEADS, HEAD_DIM, pos_start, is_glob ? 1u : 0u, QK_CLIP,  B);
        nbb_l2clip(sa(off.R), wa(lp + "attn.rel_l2_clip.scale"), N_HEADS, D_REL,    pos_start, is_glob ? 1u : 0u, REL_CLIP, B);
        vk.barrier();

        nbb_l2clip(sa(off.K), 0, N_KV_HEADS, HEAD_DIM, pos_start, 0u, QK_CLIP, B);
        nb_rmsnorm(sa(off.V), 0, sa(off.V), HEAD_DIM, B * N_KV_HEADS);
        vk.barrier();

        nbb_kv_store(il, pos_start, B); vk.barrier();

        {
            if (B <= 8u && is_glob && (pos_start + B) > 512u) {
                uint32_t n_splits = 32u;
                struct { uint64_t q, r, rp, kc, vc, pml, po; uint32_t ps, B, ns, tkv; } pcs{
                    sa(off.Q), sa(off.R), rel_proj_t_buf.addr + (uint64_t)ip * 1024ULL * 16ULL * 2ULL,
                    slot_k_addr[il], slot_v_addr[il], sa(off.attn_part_ml), sa(off.attn_part_o),
                    pos_start, B, n_splits, pos_start + B
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_attn_sk.pipe);
                vkCmdPushConstants(vk.cb, pb_attn_sk.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcs), &pcs);
                vkCmdDispatch(vk.cb, N_HEADS, n_splits, 1); vk.barrier();
                struct { uint64_t pml, po, s, g, o; uint32_t ns; } pcr{
                    sa(off.attn_part_ml), sa(off.attn_part_o), sa(off.sinks), sa(off.gate_a), sa(off.attn_raw), n_splits
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_attn_red.pipe);
                vkCmdPushConstants(vk.cb, pb_attn_red.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcr), &pcr);
                vkCmdDispatch(vk.cb, N_HEADS, B, 1); vk.barrier();
            } else if (B <= 8u) {
                struct { uint64_t q, r, s, g, rp, kc, vc, o; uint32_t ps, ig, mask, B, qt; } pc{
                    sa(off.Q), sa(off.R), sa(off.sinks), sa(off.gate_a), rel_proj_t_buf.addr + (uint64_t)ip * 1024ULL * 16ULL * 2ULL,
                    slot_k_addr[il], slot_v_addr[il], sa(off.attn_raw), pos_start, is_glob ? 1u : 0u, slot_ring_mask[il], B, 1u
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_attn.pipe);
                vkCmdPushConstants(vk.cb, pb_attn.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(vk.cb, N_HEADS, B, 1); vk.barrier();
            } else {
                struct { uint64_t q, r, s, g, rp, kc, vc, o; uint32_t ps, ig, mask, B; } pc{
                    sa(off.Q), sa(off.R), sa(off.sinks), sa(off.gate_a), rel_proj_t_buf.addr + (uint64_t)ip * 1024ULL * 16ULL * 2ULL,
                    slot_k_addr[il], slot_v_addr[il], sa(off.attn_raw), pos_start, is_glob ? 1u : 0u, slot_ring_mask[il], B
                };
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_attn_fa2.pipe);
                vkCmdPushConstants(vk.cb, pb_attn_fa2.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(vk.cb, N_HEADS, (B + 63) / 64, 1); vk.barrier();
            }
        }

        nbb_mat_w(lp + "attn.wo_ud.weight", sa(off.attn_raw), sa(off.wo_out), D_MODEL, Q_DIM, B); vk.barrier();
        nbb_sconv(sa(off.wo_out), slot_hist_addr(il, 2 * KV_DIM), wa(lp + "attn_sconv.weight"), 0, sa(off.attn_out), D_MODEL, B, 0, slot_hist_step_addr(il, 2 * KV_DIM, save_hs));

        uint64_t ffn_inp_a;
        if (ip % BLOCK_SIZE == 0) {
            int bidx = (int)blocks.size();
            vk.barrier_xfer();
            VkBufferCopy bc{cur_off, off.blocks[bidx], b_stride};
            vkCmdCopyBuffer(vk.cb, scratch.buf, scratch.buf, 1, &bc);
            blocks.push_back(sa(off.blocks[bidx]));
            ffn_inp_a = sa(off.attn_out);
            vk.barrier_xfer();
        } else {
            vk.barrier();
            nb_elem(cur_a, sa(off.attn_out), sa(off.ffn_inp), 0, B * D_MODEL, 0);
            ffn_inp_a = sa(off.ffn_inp); vk.barrier();
        }

        if (!has_msg) {
            nbb_attn_res(blocks, ffn_inp_a, wa("model.llm.attn_res.mlp_queries." + std::to_string(ip)), sa(off.xm_norm), 1, B, wa(lp + "mlp_norm.weight"));
            vk.barrier();
        } else {
            nbb_attn_res(blocks, ffn_inp_a, wa("model.llm.attn_res.mlp_queries." + std::to_string(ip)), sa(off.xm), 1, B); vk.barrier();
            struct { uint64_t cur[3], msg[3], dw[3], go[3]; uint32_t dim, rank; } pcd{};
            pcd.cur[0] = sa(off.xm); pcd.msg[0] = sa(off.messages) + 3 * b_stride;
            pcd.dw[0] = wa(lp + "controllers.mlp.down_weight"); pcd.go[0] = sa(off.ctrl_gate[0]);
            pcd.dim = D_MODEL; pcd.rank = GATE_RANK;
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_ctrl_down.pipe);
            vkCmdPushConstants(vk.cb, pb_ctrl_down.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcd), &pcd);
            vkCmdDispatch(vk.cb, GATE_RANK, 1, B); vk.barrier();

            struct { uint64_t cur[3], msg[3], g[3], gw[3], fw[3], o[3]; uint32_t dim, rank; } pca{};
            pca.cur[0] = sa(off.xm); pca.msg[0] = pcd.msg[0]; pca.g[0] = pcd.go[0];
            pca.gw[0] = wa(lp + "controllers.mlp.gain_weight"); pca.fw[0] = wa(lp + "controllers.mlp.forget_weight");
            pca.o[0] = sa(off.ctrl_gain[0]); pca.dim = D_MODEL; pca.rank = GATE_RANK;
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_ctrl_apply.pipe);
            vkCmdPushConstants(vk.cb, pb_ctrl_apply.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pca), &pca);
            vkCmdDispatch(vk.cb, D_MODEL / 64, 1, B); vk.barrier();
            nb_rmsnorm(pca.o[0], wa(lp + "mlp_norm.weight"), sa(off.xm_norm), D_MODEL, B); vk.barrier();
        }

        nbb_mat(wa(lp + "mlp.gate.weight"), sa(off.xm_norm), sa(off.r_logits), N_EXPERTS, D_MODEL, B); vk.barrier();
        {
            struct { uint64_t l, b, ids, w; } pc{sa(off.r_logits), wa(lp + "mlp.gate.bias"), sa(off.top_ids), sa(off.top_w)};
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_router.pipe);
            vkCmdPushConstants(vk.cb, pb_router.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, B, 1, 1); vk.barrier();
        }
        {
            struct { uint64_t ids, cnt, off, pt, pk; uint32_t B; } pc{
                sa(off.top_ids), sa(off.exp_counts), sa(off.exp_offsets), sa(off.pair_tok), sa(off.pair_k), B
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_group.pipe);
            vkCmdPushConstants(vk.cb, pb_group.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
        }
        if (ip < 20) {
            struct { uint64_t w, s, amax, den, cnt, off, pt, pk, x, out; uint32_t r, c, mode; } pc13{
                wa(lp + "mlp.experts.w13_weight"), wa(lp + "mlp.experts.w13_weight_scale"),
                wa(lp + "mlp.experts.w13_weight_amax"), wa(lp + "mlp.experts.w13_weight_outer_scale_denominator"),
                sa(off.exp_counts), sa(off.exp_offsets), sa(off.pair_tok), sa(off.pair_k),
                sa(off.xm_norm), sa(off.act_buf), 2 * INTER_SIZE, D_MODEL, 0
            };
            uint32_t n_e = std::min<uint32_t>(N_EXPERTS, B * TOP_K);
            if (B >= 32u) {
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_nvfp4_gemm.pipe);
                vkCmdPushConstants(vk.cb, pb_nvfp4_gemm.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc13), &pc13);
                vkCmdDispatch(vk.cb, (2 * INTER_SIZE + 63) / 64, n_e, 1); vk.barrier();
            } else {
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_nvfp4.pipe);
                vkCmdPushConstants(vk.cb, pb_nvfp4.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc13), &pc13);
                vkCmdDispatch(vk.cb, INTER_SIZE / 4, n_e, 1); vk.barrier();
            }

            struct { uint64_t w, s, amax, den, cnt, off, pt, pk, x, out; uint32_t r, c, mode; } pc2{
                wa(lp + "mlp.experts.w2_weight"), wa(lp + "mlp.experts.w2_weight_scale"),
                wa(lp + "mlp.experts.w2_weight_amax"), wa(lp + "mlp.experts.w2_weight_outer_scale_denominator"),
                sa(off.exp_counts), sa(off.exp_offsets), sa(off.pair_tok), sa(off.pair_k),
                sa(off.act_buf), sa(off.y_exps), D_MODEL, INTER_SIZE, 1
            };
            if (B >= 32u) {
                vkCmdPushConstants(vk.cb, pb_nvfp4_gemm.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
                vkCmdDispatch(vk.cb, (D_MODEL + 63) / 64, n_e, 1); vk.barrier();
            } else {
                vkCmdPushConstants(vk.cb, pb_nvfp4.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
                vkCmdDispatch(vk.cb, D_MODEL / 4, n_e, 1); vk.barrier();
            }
        } else {
            struct { uint64_t w, s, cnt, off, pt, pk, x, out; uint32_t r, c, mode; } pc13{
                wa(lp + "mlp.experts.w13_weight"), wa(lp + "mlp.experts.w13_weight_scale"),
                sa(off.exp_counts), sa(off.exp_offsets), sa(off.pair_tok), sa(off.pair_k),
                sa(off.xm_norm), sa(off.act_buf), 2 * INTER_SIZE, D_MODEL, 0
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_mxfp8.pipe);
            vkCmdPushConstants(vk.cb, pb_mxfp8.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc13), &pc13);
            vkCmdDispatch(vk.cb, INTER_SIZE / 4, std::min<uint32_t>(N_EXPERTS, B * TOP_K), 1); vk.barrier();

            struct { uint64_t w, s, cnt, off, pt, pk, x, out; uint32_t r, c, mode; } pc2{
                wa(lp + "mlp.experts.w2_weight"), wa(lp + "mlp.experts.w2_weight_scale"),
                sa(off.exp_counts), sa(off.exp_offsets), sa(off.pair_tok), sa(off.pair_k),
                sa(off.act_buf), sa(off.y_exps), D_MODEL, INTER_SIZE, 1
            };
            vkCmdPushConstants(vk.cb, pb_mxfp8.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
            vkCmdDispatch(vk.cb, D_MODEL / 4, std::min<uint32_t>(N_EXPERTS, B * TOP_K), 1); vk.barrier();
        }
        {
            struct { uint64_t ye, w, o; uint32_t d; } pc{sa(off.y_exps), sa(off.top_w), sa(off.moe_raw), D_MODEL};
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_reduce.pipe);
            vkCmdPushConstants(vk.cb, pb_reduce.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, (D_MODEL + 63) / 64, B, 1); vk.barrier();
        }
        nbb_sconv(sa(off.moe_raw), slot_hist_addr(il, 2 * KV_DIM + D_MODEL), wa(lp + "mlp_sconv.weight"), 0, sa(off.mlp_out), D_MODEL, B, 0, slot_hist_step_addr(il, 2 * KV_DIM + D_MODEL, save_hs)); vk.barrier();
        int next_slot = cur_slot ^ 1;
        nb_elem(ffn_inp_a, sa(off.mlp_out), sa(off.cur[next_slot]), 0, B * D_MODEL, 0); vk.barrier();
        cur_slot = next_slot;
        if (mtp_loaded) {
            int th_s = (il == 4) ? 0 : ((il == 28) ? 1 : ((il == 46) ? 2 : -1));
            if (th_s >= 0) { nb_elem(sa(off.cur[cur_slot]), 0, th_addr(th_s), 0, B * D_MODEL, 4); vk.barrier(); }
        }
    }

    int forward_batch_prefill(const int* toks, uint32_t B, uint32_t pos_start, bool compute_last_logit, bool verify_all = false, int* verify_out = nullptr) {
        if (B == 1 && !verify_all) {
            int r = forward_one_token(toks[0], (int)pos_start, compute_last_logit);
            if (mtp_loaded) mtp_update_ctx_kv(1, pos_start);
            return r;
        }
        uint32_t* dst_ids = (uint32_t*)tok_ids_host.mapped;
        for (uint32_t t = 0; t < B; ++t) dst_ids[t] = (uint32_t)toks[t];

        vk.begin_cmd();
        int cur_slot = 0;
        const uint64_t b_stride = (uint64_t)B * D_MODEL * 4ULL;
        {
            struct { uint64_t e, n, t, o; uint32_t dim; } pc{
                wa("model.llm.embed.weight"), wa("model.llm.embed_norm.weight"), tok_ids_host.addr, sa(off.cur[0]), D_MODEL
            };
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pb_embed.pipe);
            vkCmdPushConstants(vk.cb, pb_embed.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(vk.cb, B, 1, 1); vk.barrier();
        }

        std::vector<uint64_t> blocks;
        for (int il = 0; il < CORE_START; ++il) record_layer_batch(il, il, pos_start, B, cur_slot, blocks, false, verify_all);
        { vk.barrier_xfer(); VkBufferCopy bc{off.cur[cur_slot], off.prelude, b_stride}; vkCmdCopyBuffer(vk.cb, scratch.buf, scratch.buf, 1, &bc); vk.barrier_xfer(); }
        const auto prelude_blocks = blocks;

        for (int pass = 0; pass < N_PASSES; ++pass) {
            std::vector<uint64_t> bank;
            if (pass > 0) {
                int dst_slot = cur_slot ^ 1;
                vk.barrier_xfer();
                VkBufferCopy bc1{off.blocks[prelude_blocks.size() + 1], off.bank_block, b_stride};
                VkBufferCopy bc2{off.cur[cur_slot], off.bank_out, b_stride};
                VkBufferCopy bc3{off.prelude, off.cur[dst_slot], b_stride};
                vkCmdCopyBuffer(vk.cb, scratch.buf, scratch.buf, 1, &bc1);
                vkCmdCopyBuffer(vk.cb, scratch.buf, scratch.buf, 1, &bc2);
                vkCmdCopyBuffer(vk.cb, scratch.buf, scratch.buf, 1, &bc3);
                vk.barrier_xfer();
                cur_slot = dst_slot; bank.push_back(sa(off.bank_block)); blocks = prelude_blocks;
            }
            for (int ip = CORE_START; ip < CORE_END; ++ip) {
                int il = (pass == 0) ? ip : N_PHYS + (pass - 1) * (CORE_END - CORE_START) + (ip - CORE_START);
                if (pass == 0) record_layer_batch(il, ip, pos_start, B, cur_slot, blocks, false, verify_all);
                else {
                    uint64_t mq_addr = msg_q_buf.addr + (uint64_t)(ip - CORE_START) * 4ULL * D_MODEL * 4ULL;
                    nbb_attn_res(bank, sa(off.bank_out), mq_addr, sa(off.messages), 4, B); vk.barrier();
                    record_layer_batch(il, ip, pos_start, B, cur_slot, blocks, true, verify_all);
                }
            }
        }
        for (int il = CORE_END; il < N_PHYS; ++il) record_layer_batch(il, il, pos_start, B, cur_slot, blocks, false, verify_all);

        if (mtp_loaded || verify_all) {
            nbb_attn_res(blocks, sa(off.cur[cur_slot]), wa("model.llm.attn_res.output_query"), sa(off.xm_norm), 1, B, wa("model.llm.norm.weight")); vk.barrier();
            if (mtp_loaded) { nb_elem(sa(off.xm_norm), 0, th_addr(3), 0, B * D_MODEL, 4); vk.barrier(); }
        }
        if (verify_all) {
            nbb_mat_w("model.llm.unembed.weight", sa(off.xm_norm), sa(off.logits), VOCAB_ACT, D_MODEL, B); vk.barrier();
            uint64_t pv = sa(off.y_exps), pi = sa(off.y_exps) + 256 * 4;
            for (uint32_t t = 0; t < B; ++t) {
                uint64_t la = sa(off.logits) + (uint64_t)t * VOCAB_ACT * 4ULL;
                uint64_t oa = out_ids_host.addr + (uint64_t)t * 4ULL;
                struct { uint64_t l, pv, pi, o; uint32_t n, st; } pc0{la, pv, pi, oa, VOCAB_ACT, 0};
                vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_argmax.pipe);
                vkCmdPushConstants(vk.cb, p_argmax.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc0), &pc0);
                vkCmdDispatch(vk.cb, 256, 1, 1); vk.barrier();
                struct { uint64_t l, pv, pi, o; uint32_t n, st; } pc1{la, pv, pi, oa, 256, 1};
                vkCmdPushConstants(vk.cb, p_argmax.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
                vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
            }
        } else if (compute_last_logit) {
            uint64_t last_off = (uint64_t)(B - 1) * D_MODEL * 4ULL;
            if (!mtp_loaded) {
                std::vector<uint64_t> last_blocks;
                for (uint64_t ba : blocks) last_blocks.push_back(ba + last_off);
                nb_attn_res(last_blocks, sa(off.cur[cur_slot]) + last_off, wa("model.llm.attn_res.output_query"), sa(off.xm_norm), 1, wa("model.llm.norm.weight")); vk.barrier();
                nb_matvec_w("model.llm.unembed.weight", sa(off.xm_norm), sa(off.logits), VOCAB_ACT, D_MODEL); vk.barrier();
            } else {
                nb_matvec_w("model.llm.unembed.weight", sa(off.xm_norm) + last_off, sa(off.logits), VOCAB_ACT, D_MODEL); vk.barrier();
            }
            uint64_t pv = sa(off.y_exps), pi = sa(off.y_exps) + 256 * 4;
            struct { uint64_t l, pv, pi, o; uint32_t n, st; } pc0{sa(off.logits), pv, pi, out_ids_host.addr, VOCAB_ACT, 0};
            vkCmdBindPipeline(vk.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_argmax.pipe);
            vkCmdPushConstants(vk.cb, p_argmax.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc0), &pc0);
            vkCmdDispatch(vk.cb, 256, 1, 1); vk.barrier();
            struct { uint64_t l, pv, pi, o; uint32_t n, st; } pc1{sa(off.logits), pv, pi, out_ids_host.addr, 256, 1};
            vkCmdPushConstants(vk.cb, p_argmax.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
            vkCmdDispatch(vk.cb, 1, 1, 1); vk.barrier();
        }
        vk.submit_and_wait();
        if (verify_all && verify_out) {
            const uint32_t* p = (const uint32_t*)out_ids_host.mapped;
            for (uint32_t t = 0; t < B; ++t) verify_out[t] = (int)p[t];
            return verify_out[0];
        }
        if (mtp_loaded && !verify_all) mtp_update_ctx_kv(B, pos_start);
        if (!compute_last_logit) return -1;
        return (int)(*(const uint32_t*)out_ids_host.mapped);
    }

    // Prefill with automatic prefix cache reuse (Requirement D)
    int prefill(const std::vector<int>& tokens, int batch_size, int& out_reread) {
        size_t lcp = 0;
        while (lcp < cached_tokens.size() && lcp < tokens.size() && cached_tokens[lcp] == tokens[lcp]) ++lcp;
        if (lcp < cached_tokens.size()) {
            if (ckpt_len > 0 && lcp >= ckpt_len && (cached_tokens.size() - ckpt_len) < 1000 && lcp < tokens.size()) {
                vk.begin_cmd();
                VkBufferCopy bc{0, 0, hist_ckpt_buf.size};
                vkCmdCopyBuffer(vk.cb, hist_ckpt_buf.buf, hist_buf.buf, 1, &bc);
                vk.submit_and_wait();
                lcp = ckpt_len;
            } else {
                reset_state();
                lcp = 0;
            }
        }
        if (lcp == tokens.size() && lcp > 0) { reset_state(); lcp = 0; }

        int next_tok = -1;
        size_t pos = lcp;
        out_reread = (int)(tokens.size() - lcp);
        while (pos < tokens.size()) {
            uint32_t chunk = (uint32_t)std::min<size_t>((size_t)batch_size, tokens.size() - pos);
            bool last = (pos + chunk == tokens.size());
            next_tok = forward_batch_prefill(tokens.data() + pos, chunk, (uint32_t)pos, last);
            pos += chunk;
        }
        cached_tokens = tokens;
        vk.begin_cmd();
        VkBufferCopy bc{0, 0, hist_ckpt_buf.size};
        vkCmdCopyBuffer(vk.cb, hist_buf.buf, hist_ckpt_buf.buf, 1, &bc);
        vk.submit_and_wait();
        ckpt_len = tokens.size();
        return next_tok;
    }
};

static std::vector<int> parse_tokens_csv(const std::string& s) {
    std::vector<int> out; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stoi(tok));
    return out;
}

int main(int argc, char** argv) {
    std::string model_path = std::string(getenv("HOME")) + "/models/Inkling-2-Mini/model-00001-of-00001.safetensors";
    std::string shader_dir = std::string(getenv("HOME")) + "/inkling2/shaders";
    std::vector<int> prompt_tokens;
    int steps = 20, max_ctx = 4096, prefill_batch = 256, spec_k = 7;
    bool check_vram = false, profile = false, dump_logits = false, use_spec = false, stdio_server = false, use_fp8 = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokens" && i + 1 < argc) prompt_tokens = parse_tokens_csv(argv[++i]);
        else if (a == "--tokens-file" && i + 1 < argc) {
            std::ifstream f(argv[++i]); std::string s((std::istreambuf_iterator<char>(f)), {});
            prompt_tokens = parse_tokens_csv(s);
        } else if (a == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (a == "--max-ctx" && i + 1 < argc) max_ctx = std::stoi(argv[++i]);
        else if (a == "--prefill-batch" && i + 1 < argc) prefill_batch = std::min(MAX_BATCH, std::stoi(argv[++i]));
        else if (a == "--check-vram") check_vram = true;
        else if (a == "--profile") profile = true;
        else if (a == "--dump-logits") dump_logits = true;
        else if (a == "--spec") use_spec = true;
        else if (a == "--spec-k" && i + 1 < argc) { use_spec = true; spec_k = std::clamp(std::stoi(argv[++i]), 1, 7); }
        else if (a == "--stdio-server") stdio_server = true;
        else if (a == "--fp8") use_fp8 = true;
    }

    Engine eng; eng.init(model_path, shader_dir, max_ctx, use_fp8);
    if (use_spec) eng.init_mtp(std::string(getenv("HOME")) + "/models/Inkling-2-Mini/mtp.safetensors");
    if (check_vram) {
        std::ifstream fu("/sys/class/drm/card1/device/mem_info_vram_used");
        std::ifstream ft("/sys/class/drm/card1/device/mem_info_vram_total");
        uint64_t used = 0, tot = 0; fu >> used; ft >> tot;
        std::fprintf(stderr, "VRAM (max_ctx=%d): used=%.2f GiB / total=%.2f GiB (free=%.2f GiB)\n",
                     max_ctx, used / (1024.0*1024.0*1024.0), tot / (1024.0*1024.0*1024.0), (tot - used) / (1024.0*1024.0*1024.0));
        if (prompt_tokens.empty()) return 0;
    }
    if (stdio_server) {
        std::fprintf(stderr, "READY\n");
        std::fflush(stderr);
        auto parse_int_array = [](const std::string& line, const std::string& key) -> std::vector<int> {
            std::vector<int> res;
            size_t p = line.find("\"" + key + "\"");
            if (p == std::string::npos) return res;
            p = line.find('[', p);
            size_t e = line.find(']', p);
            if (p == std::string::npos || e == std::string::npos) return res;
            std::stringstream ss(line.substr(p + 1, e - p - 1));
            std::string item;
            while (std::getline(ss, item, ',')) {
                size_t a = item.find_first_of("-0123456789");
                if (a != std::string::npos) res.push_back(std::stoi(item.substr(a)));
            }
            return res;
        };
        auto parse_int_val = [](const std::string& line, const std::string& key, int def) -> int {
            size_t p = line.find("\"" + key + "\"");
            if (p == std::string::npos) return def;
            p = line.find(':', p);
            if (p == std::string::npos) return def;
            size_t a = line.find_first_of("-0123456789", p);
            if (a == std::string::npos) return def;
            return std::stoi(line.substr(a));
        };
        auto parse_bool_val = [](const std::string& line, const std::string& key, bool def) -> bool {
            size_t p = line.find("\"" + key + "\"");
            if (p == std::string::npos) return def;
            p = line.find(':', p);
            if (p == std::string::npos) return def;
            size_t t = line.find("true", p), f = line.find("false", p), c = line.find_first_of(",}", p);
            if (t != std::string::npos && (c == std::string::npos || t < c)) return true;
            if (f != std::string::npos && (c == std::string::npos || f < c)) return false;
            return def;
        };

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            if (line.find("\"cmd\":\"reset\"") != std::string::npos || line.find("\"cmd\": \"reset\"") != std::string::npos) {
                eng.reset_state();
                std::printf("{\"ok\":true}\n");
                std::fflush(stdout);
                continue;
            }
            std::vector<int> ptoks = parse_int_array(line, "tokens");
            int max_toks = std::max(1, parse_int_val(line, "max_tokens", 256));
            std::vector<int> stops = parse_int_array(line, "stop_tokens");
            bool req_spec = parse_bool_val(line, "spec", use_spec) && eng.mtp_loaded;
            bool adaptive = parse_bool_val(line, "adaptive", true);
            auto is_stop = [&](int t) { for (int st : stops) if (t == st) return true; return false; };

            if (ptoks.empty() || (int)ptoks.size() + max_toks >= max_ctx) {
                std::printf("{\"error\":\"invalid prompt length or exceeds max_ctx\"}\n");
                std::fflush(stdout);
                continue;
            }

            auto tp0 = std::chrono::steady_clock::now();
            int reread = 0;
            int next_tok = eng.prefill(ptoks, prefill_batch, reread);
            double prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();

            auto td0 = std::chrono::steady_clock::now();
            std::vector<int> gen;
            gen.push_back(next_tok);
            int pos = (int)ptoks.size();
            bool stopped = is_stop(next_tok);

            if (!stopped && max_toks > 1) {
                if (!req_spec) {
                    while ((int)gen.size() < max_toks && pos < max_ctx - 1) {
                        int in_tok = next_tok;
                        next_tok = eng.forward_one_token(in_tok, pos, true);
                        if (eng.mtp_loaded) eng.mtp_update_ctx_kv(1, (uint32_t)pos);
                        eng.cached_tokens.push_back(in_tok);
                        pos++;
                        gen.push_back(next_tok);
                        if (is_stop(next_tok)) break;
                    }
                } else {
                    bool spec_active = true;
                    int draft[7];
                    eng.mtp_draft_block(next_tok, (uint32_t)pos, (uint32_t)spec_k, draft);
                    int win_acc = 0, win_rounds = 0;
                    while ((int)gen.size() < max_toks && pos < max_ctx - 10) {
                        if (!spec_active) {
                            int in_tok = next_tok;
                            next_tok = eng.forward_one_token(in_tok, pos, true);
                            if (eng.mtp_loaded) eng.mtp_update_ctx_kv(1, (uint32_t)pos);
                            eng.cached_tokens.push_back(in_tok);
                            pos++;
                            gen.push_back(next_tok);
                            if (is_stop(next_tok)) break;
                            continue;
                        }
                        int ver_in[8]; ver_in[0] = next_tok;
                        for (int k = 0; k < spec_k; ++k) ver_in[k + 1] = draft[k];
                        int ver_out[8];
                        eng.forward_batch_prefill(ver_in, (uint32_t)(spec_k + 1), (uint32_t)pos, false, true, ver_out);
                        int acc = 1;
                        while (acc <= spec_k && ver_out[acc - 1] == draft[acc - 1]) acc++;

                        int eff_acc = 0;
                        bool hit_stop = false;
                        for (int i = 0; i < acc && (int)gen.size() < max_toks; ++i) {
                            gen.push_back(ver_out[i]);
                            eff_acc++;
                            if (is_stop(ver_out[i])) { hit_stop = true; break; }
                        }
                        next_tok = ver_out[eff_acc - 1];
                        for (int i = 0; i < eff_acc; ++i) eng.cached_tokens.push_back(ver_in[i]);

                        bool done = hit_stop || (int)gen.size() >= max_toks || (pos + eff_acc >= max_ctx - 10);
                        win_acc += acc; win_rounds++;
                        if (adaptive && win_rounds == 4 && win_acc < 12) {
                            spec_active = false;
                        }
                        eng.mtp_post_verify_and_draft((uint32_t)eff_acc, (uint32_t)spec_k, (uint32_t)pos, next_tok, draft, !done && spec_active);
                        pos += eff_acc;
                        if (done) break;
                    }
                }
            }
            double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            std::printf("{\"tokens\":[");
            for (size_t i = 0; i < gen.size(); ++i) std::printf("%s%d", i ? "," : "", gen[i]);
            std::printf("],\"prompt_tokens\":%zu,\"reread\":%d,\"completion_tokens\":%zu,\"prefill_ms\":%.2f,\"decode_ms\":%.2f}\n",
                        ptoks.size(), reread, gen.size(), prefill_ms, decode_ms);
            std::fflush(stdout);
        }
        return 0;
    }
    if (prompt_tokens.empty()) return 1;

    auto tp0 = std::chrono::steady_clock::now();
    int reread = 0;
    int next_tok = eng.prefill(prompt_tokens, prefill_batch, reread);
    if (dump_logits) {
        GpuBuf dl = eng.vk.alloc(VOCAB_ACT * 4, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
        eng.vk.begin_cmd();
        VkBufferCopy bc{eng.off.logits, 0, VOCAB_ACT * 4};
        vkCmdCopyBuffer(eng.vk.cb, eng.scratch.buf, dl.buf, 1, &bc);
        eng.vk.submit_and_wait();
        const float* lg = (const float*)dl.mapped;
        FILE* fp = fopen(prefill_batch == 1 ? "logits_b1.bin" : "logits_b18.bin", "wb");
        fwrite(lg, 4, VOCAB_ACT, fp); fclose(fp);
    }
    double tp_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
    std::fprintf(stderr, "Prefill %zu tokens (B=%d, reread=%d): %.3f s (%.1f tok/s)\n",
                 prompt_tokens.size(), prefill_batch, reread, tp_sec, prompt_tokens.size() / tp_sec);

    std::vector<int> generated = {next_tok};
    int pos = (int)prompt_tokens.size();
    std::vector<double> step_ms;
    double prof8[8] = {0}; int pn = 0;
    if (!use_spec) {
        for (int s = 1; s < steps; ++s) {
            auto t0 = std::chrono::steady_clock::now();
            bool do_p = profile && s >= 50 && s < 60;
            next_tok = eng.forward_one_token(next_tok, pos++, true, do_p ? prof8 : nullptr);
            if (do_p) pn++;
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (s >= 2) step_ms.push_back(ms);
            generated.push_back(next_tok);
        }
    } else {
        int n_rounds = 0, total_acc = 0, warm_toks = 0;
        double total_ver_ms = 0, total_post_ms = 0, total_wall_ms = 0;
        int draft[7];
        eng.mtp_draft_block(next_tok, (uint32_t)pos, (uint32_t)spec_k, draft);
        while ((int)generated.size() < steps) {
            auto t0 = std::chrono::steady_clock::now();
            int ver_in[8]; ver_in[0] = next_tok;
            for (int k = 0; k < spec_k; ++k) ver_in[k + 1] = draft[k];
            int ver_out[8];
            eng.forward_batch_prefill(ver_in, (uint32_t)(spec_k + 1), (uint32_t)pos, false, true, ver_out);
            auto t1 = std::chrono::steady_clock::now();
            int acc = 1;
            while (acc <= spec_k && ver_out[acc - 1] == draft[acc - 1]) acc++;
            for (int i = 0; i < acc && (int)generated.size() < steps; ++i) generated.push_back(ver_out[i]);
            next_tok = ver_out[acc - 1];
            if ((int)generated.size() < steps) {
                eng.mtp_post_verify_and_draft((uint32_t)acc, (uint32_t)spec_k, (uint32_t)pos, next_tok, draft);
            }
            auto t2 = std::chrono::steady_clock::now();
            pos += acc;
            n_rounds++;
            if (n_rounds >= 2) {
                total_acc += acc; warm_toks += acc;
                total_ver_ms  += std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_post_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
                total_wall_ms += std::chrono::duration<double, std::milli>(t2 - t0).count();
            }
        }
        if (warm_toks > 0) {
            int wr = n_rounds - 1;
            std::fprintf(stderr, "Speculative decode (K=%d, %d rounds, %d toks after warmup, ctx~%d): %.1f tok/s | avg_acc=%.2f/%d | ver=%.2f ms, rb+upd+draft=%.2f ms, total=%.2f ms/round\n",
                         spec_k, wr, warm_toks, (int)prompt_tokens.size() + steps / 2,
                         1000.0 * warm_toks / total_wall_ms, (double)total_acc / wr, spec_k + 1,
                         total_ver_ms / wr, total_post_ms / wr, total_wall_ms / wr);
        }
    }

    std::printf("GENERATED:");
    for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : " ", generated[i]);
    std::printf("\n");
    if (!step_ms.empty()) {
        std::sort(step_ms.begin(), step_ms.end());
        double med = step_ms[step_ms.size() / 2];
        std::fprintf(stderr, "Decode wall-clock (%zu steps after warmup, ctx~%d): median=%.2f ms (%.1f tok/s), min=%.2f, max=%.2f\n",
                     step_ms.size(), (int)prompt_tokens.size() + steps / 2, med, 1000.0 / med, step_ms.front(), step_ms.back());
    }
    if (profile && pn > 0) {
        const char* names[7] = {"embed","attnres+ctrl","attn_proj+conv","attn_core","moe_router","moe_experts","unembed"};
        double tot = prof8[7] / pn;
        std::fprintf(stderr, "Quiet-Box Steady-State GPU Breakdown (steps 50..59, avg of %d tokens, ctx~%d):\n", pn, (int)prompt_tokens.size() + 55);
        for (int i = 1; i < 7; ++i) std::fprintf(stderr, "  %-16s: %6.2f ms (%5.1f%%)\n", names[i], prof8[i]/pn, 100.0*(prof8[i]/pn)/tot);
        std::fprintf(stderr, "  %-16s: %6.2f ms (100.0%%)\n", "GPU_TOTAL", tot);
    }
    return 0;
}
