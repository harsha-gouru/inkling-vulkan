#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define MAX_BUFS  4096
#define MAX_PIPES 64
#define MAX_CMD_SLOTS 8
#define STAGING_SZ (64ULL * 1024ULL * 1024ULL)

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "VK_ERROR %d at %s:%d\n", (int)_r, __FILE__, __LINE__); return -1; } } while (0)

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
    void *mapped;
} BufObj;

typedef struct {
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkShaderModule sm;
    VkPipeline pipe;
    int n_bindings;
    uint32_t uniform_mask;
} PipeObj;

static VkInstance g_inst;
static VkPhysicalDevice g_pdev;
static VkDevice g_dev;
static uint32_t g_qfam = 0;
static VkQueue g_queue;
static VkCommandPool g_cpool;
static VkDescriptorPool g_dpool;
static VkFence g_fence;
static VkCommandBuffer g_cmd_once;
static VkCommandBuffer g_cmd_rec[MAX_CMD_SLOTS];
static VkPhysicalDeviceMemoryProperties g_mprops;

static BufObj g_bufs[MAX_BUFS];
static int g_nbufs = 0;
static PipeObj g_pipes[MAX_PIPES];
static int g_npipes = 0;
static int g_last_pipe[MAX_CMD_SLOTS];

// Reusable 64 MiB staging buffer for uploads
static VkBuffer g_stg_buf;
static VkDeviceMemory g_stg_mem;
static void *g_stg_ptr = NULL;

static uint32_t find_mem_type(uint32_t bits, VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < g_mprops.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && ((g_mprops.memoryTypes[i].propertyFlags & req) == req))
            return i;
    }
    fprintf(stderr, "No matching Vulkan memory type for flags 0x%x\n", req);
    return 0;
}

int vk_init(void) {
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.pApplicationName = "inkling_vk";
    ai.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &ai;
    VK_CHECK(vkCreateInstance(&ici, NULL, &g_inst));

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(g_inst, &ndev, NULL);
    if (ndev == 0) { fprintf(stderr, "No Vulkan devices\n"); return -1; }
    VkPhysicalDevice pdevs[16];
    if (ndev > 16) ndev = 16;
    vkEnumeratePhysicalDevices(g_inst, &ndev, pdevs);

    g_pdev = pdevs[0];
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pdevs[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g_pdev = pdevs[i];
            break;
        }
    }
    vkGetPhysicalDeviceMemoryProperties(g_pdev, &g_mprops);

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pdev, &nqf, NULL);
    VkQueueFamilyProperties qfs[16];
    if (nqf > 16) nqf = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pdev, &nqf, qfs);
    for (uint32_t i = 0; i < nqf; i++) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_qfam = i; break; }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = g_qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures feats = {0};
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &feats;
    VK_CHECK(vkCreateDevice(g_pdev, &dci, NULL, &g_dev));
    vkGetDeviceQueue(g_dev, g_qfam, 0, &g_queue);

    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.queueFamilyIndex = g_qfam;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(g_dev, &cpci, NULL, &g_cpool));

    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = g_cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(g_dev, &cbai, &g_cmd_once));
    for (int i = 0; i < MAX_CMD_SLOTS; i++) {
        VK_CHECK(vkAllocateCommandBuffers(g_dev, &cbai, &g_cmd_rec[i]));
        g_last_pipe[i] = -1;
    }

    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(vkCreateFence(g_dev, &fci, NULL, &g_fence));

    VkDescriptorPoolSize psizes[2] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 96000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16000 },
    };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 16000;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = psizes;
    VK_CHECK(vkCreateDescriptorPool(g_dev, &dpci, NULL, &g_dpool));

    // Create reusable 64 MiB staging buffer
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = STAGING_SZ;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(g_dev, &bci, NULL, &g_stg_buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, g_stg_buf, &mr);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(g_dev, &mai, NULL, &g_stg_mem));
    VK_CHECK(vkBindBufferMemory(g_dev, g_stg_buf, g_stg_mem, 0));
    VK_CHECK(vkMapMemory(g_dev, g_stg_mem, 0, STAGING_SZ, 0, &g_stg_ptr));

    g_nbufs = 0;
    g_npipes = 0;
    return 0;
}

int vk_create_buffer(uint64_t size_bytes, int host_visible, int is_uniform) {
    if (g_nbufs >= MAX_BUFS) return -1;
    uint64_t sz = (size_bytes + 15ULL) & ~15ULL;
    if (sz == 0) sz = 16;

    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = sz;
    bci.usage = (is_uniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
              | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    int id = g_nbufs++;
    BufObj *b = &g_bufs[id];
    memset(b, 0, sizeof(*b));
    b->size = sz;
    VK_CHECK(vkCreateBuffer(g_dev, &bci, NULL, &b->buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, b->buf, &mr);
    VkMemoryPropertyFlags flags = (host_visible == 2)
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        : (host_visible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_mem_type(mr.memoryTypeBits, flags);
    VK_CHECK(vkAllocateMemory(g_dev, &mai, NULL, &b->mem));
    VK_CHECK(vkBindBufferMemory(g_dev, b->buf, b->mem, 0));

    if (host_visible) {
        VK_CHECK(vkMapMemory(g_dev, b->mem, 0, sz, 0, &b->mapped));
    }
    return id;
}

int vk_upload_buffer(int buf_id, const void *src, uint64_t nbytes) {
    BufObj *b = &g_bufs[buf_id];
    if (b->mapped) {
        memcpy(b->mapped, src, nbytes);
        return 0;
    }
    const uint8_t *p = (const uint8_t *)src;
    uint64_t off = 0;
    while (off < nbytes) {
        uint64_t chunk = nbytes - off;
        if (chunk > STAGING_SZ) chunk = STAGING_SZ;
        memcpy(g_stg_ptr, p + off, chunk);

        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(g_cmd_once, &bi));
        VkBufferCopy cp = { 0, off, chunk };
        vkCmdCopyBuffer(g_cmd_once, g_stg_buf, b->buf, 1, &cp);
        VK_CHECK(vkEndCommandBuffer(g_cmd_once));

        VK_CHECK(vkResetFences(g_dev, 1, &g_fence));
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g_cmd_once;
        VK_CHECK(vkQueueSubmit(g_queue, 1, &si, g_fence));
        VK_CHECK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
        off += chunk;
    }
    return 0;
}

int vk_download_buffer(int buf_id, void *dst, uint64_t nbytes) {
    BufObj *b = &g_bufs[buf_id];
    if (b->mapped) { memcpy(dst, b->mapped, nbytes); return 0; }
    uint8_t *p = (uint8_t *)dst;
    uint64_t off = 0;
    while (off < nbytes) {
        uint64_t chunk = nbytes - off;
        if (chunk > STAGING_SZ) chunk = STAGING_SZ;
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(g_cmd_once, &bi));
        VkBufferCopy cp = { off, 0, chunk };
        vkCmdCopyBuffer(g_cmd_once, b->buf, g_stg_buf, 1, &cp);
        VK_CHECK(vkEndCommandBuffer(g_cmd_once));

        VK_CHECK(vkResetFences(g_dev, 1, &g_fence));
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g_cmd_once;
        VK_CHECK(vkQueueSubmit(g_queue, 1, &si, g_fence));
        VK_CHECK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
        memcpy(p + off, g_stg_ptr, chunk);
        off += chunk;
    }
    return 0;
}

int vk_upload_slice(int buf_id, uint64_t dst_off, const void *src, uint64_t nbytes) {
    BufObj *b = &g_bufs[buf_id];
    if (b->mapped) { memcpy((uint8_t*)b->mapped + dst_off, src, nbytes); return 0; }
    const uint8_t *p = (const uint8_t *)src;
    uint64_t off = 0;
    while (off < nbytes) {
        uint64_t chunk = nbytes - off;
        if (chunk > STAGING_SZ) chunk = STAGING_SZ;
        memcpy(g_stg_ptr, p + off, chunk);
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(g_cmd_once, &bi));
        VkBufferCopy cp = { 0, dst_off + off, chunk };
        vkCmdCopyBuffer(g_cmd_once, g_stg_buf, b->buf, 1, &cp);
        VK_CHECK(vkEndCommandBuffer(g_cmd_once));
        VK_CHECK(vkResetFences(g_dev, 1, &g_fence));
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1; si.pCommandBuffers = &g_cmd_once;
        VK_CHECK(vkQueueSubmit(g_queue, 1, &si, g_fence));
        VK_CHECK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
        off += chunk;
    }
    return 0;
}

int vk_download_slice(int buf_id, uint64_t src_off, void *dst, uint64_t nbytes) {
    BufObj *b = &g_bufs[buf_id];
    if (b->mapped) { memcpy(dst, (uint8_t*)b->mapped + src_off, nbytes); return 0; }
    uint8_t *p = (uint8_t *)dst;
    uint64_t off = 0;
    while (off < nbytes) {
        uint64_t chunk = nbytes - off;
        if (chunk > STAGING_SZ) chunk = STAGING_SZ;
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(g_cmd_once, &bi));
        VkBufferCopy cp = { src_off + off, 0, chunk };
        vkCmdCopyBuffer(g_cmd_once, b->buf, g_stg_buf, 1, &cp);
        VK_CHECK(vkEndCommandBuffer(g_cmd_once));
        VK_CHECK(vkResetFences(g_dev, 1, &g_fence));
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1; si.pCommandBuffers = &g_cmd_once;
        VK_CHECK(vkQueueSubmit(g_queue, 1, &si, g_fence));
        VK_CHECK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
        memcpy(p + off, g_stg_ptr, chunk);
        off += chunk;
    }
    return 0;
}

int vk_zero_slice(int buf_id, uint64_t offset, uint64_t size) {
    BufObj *b = &g_bufs[buf_id];
    if (b->mapped) { memset((uint8_t*)b->mapped + offset, 0, size); return 0; }
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(g_cmd_once, &bi));
    vkCmdFillBuffer(g_cmd_once, b->buf, offset, size, 0);
    VK_CHECK(vkEndCommandBuffer(g_cmd_once));
    VK_CHECK(vkResetFences(g_dev, 1, &g_fence));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd_once;
    VK_CHECK(vkQueueSubmit(g_queue, 1, &si, g_fence));
    VK_CHECK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
    return 0;
}

int vk_zero_buffer(int buf_id) {
    BufObj *b = &g_bufs[buf_id];
    if (b->mapped) { memset(b->mapped, 0, b->size); return 0; }
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(g_cmd_once, &bi));
    vkCmdFillBuffer(g_cmd_once, b->buf, 0, b->size, 0);
    VK_CHECK(vkEndCommandBuffer(g_cmd_once));
    VK_CHECK(vkResetFences(g_dev, 1, &g_fence));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd_once;
    VK_CHECK(vkQueueSubmit(g_queue, 1, &si, g_fence));
    VK_CHECK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
    return 0;
}

void *vk_get_mapped(int buf_id) {
    return g_bufs[buf_id].mapped;
}

int vk_create_pipeline(const uint32_t *spv, uint64_t spv_bytes, int n_bindings, uint32_t uniform_mask) {
    if (g_npipes >= MAX_PIPES) return -1;
    int id = g_npipes++;
    PipeObj *po = &g_pipes[id];
    memset(po, 0, sizeof(*po));
    po->n_bindings = n_bindings;
    po->uniform_mask = uniform_mask;

    VkDescriptorSetLayoutBinding binds[16];
    for (int i = 0; i < n_bindings; i++) {
        memset(&binds[i], 0, sizeof(binds[i]));
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = ((uniform_mask >> i) & 1u)
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = (uint32_t)n_bindings;
    dslci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(g_dev, &dslci, NULL, &po->dsl));

    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &po->dsl;
    VK_CHECK(vkCreatePipelineLayout(g_dev, &plci, NULL, &po->pl));

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = (size_t)spv_bytes;
    smci.pCode = spv;
    VK_CHECK(vkCreateShaderModule(g_dev, &smci, NULL, &po->sm));

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = po->sm;
    cpci.stage.pName = "main";
    cpci.layout = po->pl;
    VK_CHECK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpci, NULL, &po->pipe));
    return id;
}

int vk_record_begin_slot(int slot) {
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = 0;
    VK_CHECK(vkBeginCommandBuffer(g_cmd_rec[slot], &bi));
    g_last_pipe[slot] = -1;
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(g_cmd_rec[slot], VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    return 0;
}

int vk_record_dispatch_slot(int slot, int pipe_id, int n_bufs, const int *buf_ids, uint32_t wx, uint32_t wy, uint32_t wz, int barrier_after) {
    PipeObj *po = &g_pipes[pipe_id];
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = g_dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &po->dsl;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(g_dev, &dsai, &ds));

    VkDescriptorBufferInfo dbis[16];
    VkWriteDescriptorSet writes[16];
    for (int i = 0; i < n_bufs; i++) {
        dbis[i].buffer = g_bufs[buf_ids[i]].buf;
        dbis[i].offset = 0;
        dbis[i].range  = VK_WHOLE_SIZE;
        memset(&writes[i], 0, sizeof(writes[i]));
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = ((po->uniform_mask >> i) & 1u)
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(g_dev, (uint32_t)n_bufs, writes, 0, NULL);

    if (pipe_id != g_last_pipe[slot]) {
        vkCmdBindPipeline(g_cmd_rec[slot], VK_PIPELINE_BIND_POINT_COMPUTE, po->pipe);
        g_last_pipe[slot] = pipe_id;
    }
    vkCmdBindDescriptorSets(g_cmd_rec[slot], VK_PIPELINE_BIND_POINT_COMPUTE, po->pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(g_cmd_rec[slot], wx, wy, wz);

    if (barrier_after) {
        VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(g_cmd_rec[slot], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);
    }
    return 0;
}

int vk_record_end_slot(int slot) {
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_cmd_rec[slot], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    VK_CHECK(vkEndCommandBuffer(g_cmd_rec[slot]));
    return 0;
}

int vk_run_recorded_slot(int slot) {
    VK_CHECK(vkResetFences(g_dev, 1, &g_fence));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd_rec[slot];
    VK_CHECK(vkQueueSubmit(g_queue, 1, &si, g_fence));
    VK_CHECK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
    return 0;
}

int vk_record_begin(void) { return vk_record_begin_slot(0); }
int vk_record_dispatch(int pipe_id, int n_bufs, const int *buf_ids, uint32_t wx, uint32_t wy, uint32_t wz, int barrier_after) {
    return vk_record_dispatch_slot(0, pipe_id, n_bufs, buf_ids, wx, wy, wz, barrier_after);
}
int vk_record_end(void) { return vk_record_end_slot(0); }
int vk_run_recorded(void) { return vk_run_recorded_slot(0); }

void vk_shutdown(void) {
    if (!g_dev) return;
    vkDeviceWaitIdle(g_dev);
    for (int i = 0; i < g_npipes; i++) {
        vkDestroyPipeline(g_dev, g_pipes[i].pipe, NULL);
        vkDestroyShaderModule(g_dev, g_pipes[i].sm, NULL);
        vkDestroyPipelineLayout(g_dev, g_pipes[i].pl, NULL);
        vkDestroyDescriptorSetLayout(g_dev, g_pipes[i].dsl, NULL);
    }
    for (int i = 0; i < g_nbufs; i++) {
        if (g_bufs[i].mapped) vkUnmapMemory(g_dev, g_bufs[i].mem);
        vkDestroyBuffer(g_dev, g_bufs[i].buf, NULL);
        vkFreeMemory(g_dev, g_bufs[i].mem, NULL);
    }
    if (g_stg_ptr) vkUnmapMemory(g_dev, g_stg_mem);
    vkDestroyBuffer(g_dev, g_stg_buf, NULL);
    vkFreeMemory(g_dev, g_stg_mem, NULL);
    vkDestroyDescriptorPool(g_dev, g_dpool, NULL);
    vkDestroyFence(g_dev, g_fence, NULL);
    vkDestroyCommandPool(g_dev, g_cpool, NULL);
    vkDestroyDevice(g_dev, NULL);
    vkDestroyInstance(g_inst, NULL);
    g_dev = VK_NULL_HANDLE;
}
