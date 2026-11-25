#include "vk_abi.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { std::fprintf(stderr, "VK ERR %d at %s:%d\n", (int)_r, __FILE__, __LINE__); std::exit(1); } } while(0)

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path); std::exit(1); }
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint32_t> buf(sz / 4);
    f.read((char*)buf.data(), sz);
    return buf;
}

int main() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance inst = nullptr;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t ndev = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ndev, nullptr));
    std::vector<VkPhysicalDevice> devs(ndev);
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ndev, devs.data()));

    VkPhysicalDevice phys = devs[0];
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        std::printf("Device: %s (type=%u, api=%u.%u.%u, maxStorage=%u MiB, maxPush=%u, tsPeriod=%.3f ns)\n",
            p.deviceName, p.deviceType, (p.apiVersion>>22)&0x7f, (p.apiVersion>>12)&0x3ff, p.apiVersion&0xfff,
            p.limits.maxStorageBufferRange / (1024*1024), p.limits.maxPushConstantsSize, p.limits.timestampPeriod);
        if (p.deviceType == 2) phys = d; // discrete GPU
    }

    VkPhysicalDeviceSubgroupProperties sg{};
    sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(phys, &p2);
    std::printf("Subgroup size: %u, sharedMem: %u KiB\n", sg.subgroupSize, p2.properties.limits.maxComputeSharedMemorySize / 1024);

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan11Features f11{};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(phys, &f2);
    std::printf("Features: 16bitStorage=%u 8bitStorage=%u float16=%u int8=%u BDA=%u scalarLayout=%u\n",
        f11.storageBuffer16BitAccess, f12.storageBuffer8BitAccess, f12.shaderFloat16,
        f12.shaderInt8, f12.bufferDeviceAddress, f12.scalarBlockLayout);

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        std::printf("Heap %u: %.2f GiB (flags=0x%x)\n", i, mp.memoryHeaps[i].size / (1024.0*1024.0*1024.0), mp.memoryHeaps[i].flags);
    }

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfs.data());
    uint32_t qfam = 0;
    for (uint32_t i = 0; i < nqf; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures enabled_f{};
    enabled_f._pad[51] = VK_TRUE; // shaderInt64
    VkPhysicalDeviceVulkan12Features en12{};
    en12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    en12.bufferDeviceAddress = VK_TRUE;
    en12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
    en12.shaderInt8 = f12.shaderInt8;
    en12.shaderFloat16 = f12.shaderFloat16;
    en12.scalarBlockLayout = f12.scalarBlockLayout;
    VkPhysicalDeviceVulkan11Features en11{};
    en11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    en11.pNext = &en12;
    en11.storageBuffer16BitAccess = f11.storageBuffer16BitAccess;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &en11;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &f2.features;
    VkDevice dev = nullptr;
    VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
    VkQueue queue = nullptr;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    auto find_mem = [&](uint32_t bits, VkFlags req) -> uint32_t {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & req) == req) return i;
        std::exit(1);
    };

    auto alloc_buf = [&](VkDeviceSize sz, VkBuffer& buf, VkDeviceMemory& mem, uint64_t& addr) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sz;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(dev, buf, &mr);
        VkMemoryAllocateFlagsInfo afi{};
        afi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        afi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &afi;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(dev, buf, mem, 0));
        VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buf};
        addr = vkGetBufferDeviceAddress(dev, &bdai);
    };

    const VkDeviceSize N_BYTES = 1024ULL * 1024 * 1024; // 1 GiB
    VkBuffer in_buf, out_buf;
    VkDeviceMemory in_mem, out_mem;
    uint64_t in_addr, out_addr;
    alloc_buf(N_BYTES, in_buf, in_mem, in_addr);
    alloc_buf(4096, out_buf, out_mem, out_addr);

    auto spv = read_spv("shaders/bw_bench.spv");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size()*4, spv.data()};
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
    VkPipelineLayout pl;
    VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
    cpci.layout = pl;
    VkPipeline pipe;
    VK_CHECK(vkCreateComputePipelines(dev, 0, 1, &cpci, nullptr, &pipe));

    VkCommandPoolCreateInfo cmdpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfam};
    VkCommandPool cmdp;
    VK_CHECK(vkCreateCommandPool(dev, &cmdpci, nullptr, &cmdp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmdp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_TIMESTAMP, 2, 0};
    VkQueryPool qp;
    VK_CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qp));

    struct { uint64_t in_a, out_a; uint32_t n_vec4; } pc{in_addr, out_addr, (uint32_t)(N_BYTES / 16)};
    std::vector<double> gbps;
    for (int run = 0; run < 7; ++run) {
        VK_CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        vkCmdDispatch(cb, 4096, 1, 1);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
        VK_CHECK(vkQueueSubmit(queue, 1, &si, 0));
        VK_CHECK(vkQueueWaitIdle(queue));
        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double sec = (ts[1] - ts[0]) * (double)p2.properties.limits.timestampPeriod * 1e-9;
        double bw = (double)N_BYTES / sec / 1e9;
        if (run >= 2) gbps.push_back(bw); // skip 2 warmup runs
    }
    std::sort(gbps.begin(), gbps.end());
    std::printf("1 GiB read bandwidth (5 runs after warmup): median=%.1f GB/s (min=%.1f, max=%.1f)\n",
        gbps[2], gbps.front(), gbps.back());
    return 0;
}
