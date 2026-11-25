#include "vk_abi.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { std::fprintf(stderr, "VK ERR %d at %d\n", (int)_r, __LINE__); std::exit(1); } } while(0)

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    size_t sz = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> buf(sz / 4); f.read((char*)buf.data(), sz); return buf;
}

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "b", 1, "b", 1, VK_MAKE_API_VERSION(0,1,3,0)};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    VkInstance inst; VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t nd = 0; VK_CHECK(vkEnumeratePhysicalDevices(inst, &nd, nullptr));
    std::vector<VkPhysicalDevice> devs(nd); VK_CHECK(vkEnumeratePhysicalDevices(inst, &nd, devs.data()));
    VkPhysicalDevice phys = devs[0]; float ts_ns = 10.0f;
    for (auto d : devs) { VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(d, &p); if (p.deviceType == 2) { phys = d; ts_ns = p.limits.timestampPeriod; break; } }
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf); vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfs.data());
    uint32_t qfam = 0; for (uint32_t i = 0; i < nqf; ++i) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    vkGetPhysicalDeviceFeatures2(phys, &f2);
    float prio = 1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qfam, 1, &prio};
    VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    en12.bufferDeviceAddress = VK_TRUE; en12.scalarBlockLayout = f12.scalarBlockLayout;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &en12, 0, 1, &qci, 0, nullptr, 0, nullptr, &f2.features};
    VkDevice dev; VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qfam, 0, &q);

    auto alloc = [&](VkDeviceSize sz) -> uint64_t {
        VkBuffer b; VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &b));
        VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(dev, b, &mr);
        uint32_t mt = 0; for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((mr.memoryTypeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
        VkMemoryAllocateFlagsInfo afi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &afi, mr.size, mt};
        VkDeviceMemory m; VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &m));
        VK_CHECK(vkBindBufferMemory(dev, b, m, 0));
        VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, b};
        return vkGetBufferDeviceAddress(dev, &bdai);
    };
    auto load = [&](const char* path, uint32_t ps, VkPipelineLayout& pl, VkPipeline& pipe) {
        auto spv = read_spv(path); VkShaderModule sm;
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size()*4, spv.data()};
        VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, ps};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkComputePipelineCreateInfo cpci{}; cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
        cpci.layout = pl; VK_CHECK(vkCreateComputePipelines(dev, 0, 1, &cpci, nullptr, &pipe));
    };

    uint64_t a = alloc(64ULL * 1024 * 1024);
    VkCommandPool cp; VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfam};
    VK_CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBuffer cb; VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkQueryPool qp; VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_TIMESTAMP, 4, 0};
    VK_CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qp));

    auto barrier = [&]() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };

    auto time_it = [&](const char* name, int iters, auto fn) {
        std::vector<double> ms;
        for (int r = 0; r < 7; ++r) {
            vkResetCommandBuffer(cb, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cb, &bi);
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            for (int i = 0; i < iters; ++i) fn();
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            vkEndCommandBuffer(cb);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
            vkQueueSubmit(q, 1, &si, 0); vkQueueWaitIdle(q);
            uint64_t ts[2]; vkGetQueryPoolResults(dev, qp, 0, 2, 16, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (r >= 2) ms.push_back((ts[1] - ts[0]) * ts_ns * 1e-6);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("%-30s: %6.3f ms total (%5.1f us/call, N=%d)\n", name, ms[2], ms[2]*1000.0/iters, iters);
    };

    VkPipelineLayout pl_rms, pl_ar, pl_cd, pl_ca, pl_rt;
    VkPipeline p_rms, p_ar, p_cd, p_ca, p_rt;
    load("shaders/rmsnorm.spv", 28, pl_rms, p_rms);
    load("shaders/attn_res_mix.spv", 120, pl_ar, p_ar);
    load("shaders/ctrl_down_fused.spv", 104, pl_cd, p_cd);
    load("shaders/ctrl_apply_norm_fused.spv", 152, pl_ca, p_ca);
    load("shaders/moe_router.spv", 32, pl_rt, p_rt);

    time_it("rmsnorm + barrier", 200, [&]() {
        struct { uint64_t i, w, o; uint32_t d; } pc{a, a+16384, a+32768, 2304};
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_rms);
        vkCmdPushConstants(cb, pl_rms, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, 1, 1, 1); barrier();
    });
    time_it("attn_res_mix(m=3,norm=1)+bar", 120, [&]() {
        struct { uint64_t st[8]; uint64_t q, o; uint64_t nw[3]; uint32_t m, nq, dim, dn; } pc{};
        pc.st[0]=a; pc.st[1]=a+16384; pc.st[2]=a+32768; pc.q=a+49152; pc.o=a+65536; pc.nw[0]=a+81920; pc.m=3; pc.nq=1; pc.dim=2304; pc.dn=1;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_ar);
        vkCmdPushConstants(cb, pl_ar, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, 1, 1, 1); barrier();
    });
    time_it("ctrl_down(3x) + barrier", 24, [&]() {
        struct { uint64_t cur[3], msg[3], dw[3], go[3]; uint32_t dim, rank; } pc{};
        for(int c=0;c<3;++c){pc.cur[c]=a; pc.msg[c]=a+16384; pc.dw[c]=a+1048576*(c+1); pc.go[c]=a+32768+c*256;}
        pc.dim=2304; pc.rank=64;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_cd);
        vkCmdPushConstants(cb, pl_cd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, 64, 3, 1); barrier();
    });
    time_it("ctrl_apply(3x) + barrier", 24, [&]() {
        struct { uint64_t cur[3], msg[3], g[3], gw[3], fw[3], o[3]; uint32_t dim, rank; } pc{};
        for(int c=0;c<3;++c){pc.cur[c]=a; pc.msg[c]=a+16384; pc.g[c]=a+32768; pc.gw[c]=a+1048576*(c+1); pc.fw[c]=a+1048576*(c+4); pc.o[c]=a+65536+c*16384;}
        pc.dim=2304; pc.rank=64;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_ca);
        vkCmdPushConstants(cb, pl_ca, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, 36, 3, 1); barrier();
    });
    time_it("moe_router(1-thr) + barrier", 48, [&]() {
        struct { uint64_t l, b, ids, w; } pc{a, a+1024, a+2048, a+3072};
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_rt);
        vkCmdPushConstants(cb, pl_rt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, 1, 1, 1); barrier();
    });
    return 0;
}
