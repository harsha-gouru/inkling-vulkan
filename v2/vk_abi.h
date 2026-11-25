#pragma once
// Minimal Vulkan 1.1-1.3 C ABI declarations for compute-only engine (links against libvulkan.so).
#include <cstdint>
#include <cstddef>

extern "C" {

typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;

#define VK_TRUE  1U
#define VK_FALSE 0U
#define VK_WHOLE_SIZE (~0ULL)
#define VK_MAKE_API_VERSION(variant, major, minor, patch) \
    ((((uint32_t)(variant)) << 29U) | (((uint32_t)(major)) << 22U) | (((uint32_t)(minor)) << 12U) | ((uint32_t)(patch)))

typedef struct VkInstance_T*             VkInstance;
typedef struct VkPhysicalDevice_T*       VkPhysicalDevice;
typedef struct VkDevice_T*               VkDevice;
typedef struct VkQueue_T*                VkQueue;
typedef struct VkCommandBuffer_T*        VkCommandBuffer;
typedef uint64_t VkDeviceMemory;
typedef uint64_t VkBuffer;
typedef uint64_t VkShaderModule;
typedef uint64_t VkPipelineLayout;
typedef uint64_t VkPipeline;
typedef uint64_t VkPipelineCache;
typedef uint64_t VkDescriptorSetLayout;
typedef uint64_t VkDescriptorPool;
typedef uint64_t VkDescriptorSet;
typedef uint64_t VkCommandPool;
typedef uint64_t VkFence;
typedef uint64_t VkQueryPool;
typedef void (*PFN_vkVoidFunction)(void);

typedef enum VkResult {
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
} VkResult;

typedef enum VkStructureType {
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
    VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 11,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES = 1000094000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES = 49,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES = 51,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES = 53,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO = 1000060000,
    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO = 1000244001,
} VkStructureType;

enum {
    VK_QUEUE_GRAPHICS_BIT = 0x00000001,
    VK_QUEUE_COMPUTE_BIT  = 0x00000002,
    VK_QUEUE_TRANSFER_BIT = 0x00000004,
};

enum {
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  = 0x00000001,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  = 0x00000002,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004,
    VK_MEMORY_PROPERTY_HOST_CACHED_BIT   = 0x00000008,
};

enum {
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT          = 0x00000001,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT          = 0x00000002,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT        = 0x00000020,
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x00020000,
};

enum {
    VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT = 0x00000002,
};

enum {
    VK_SHARING_MODE_EXCLUSIVE = 0,
};

enum {
    VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020,
};

enum {
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
};

enum {
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001,
};

enum {
    VK_PIPELINE_BIND_POINT_COMPUTE = 1,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800,
    VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000,
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x00002000,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT = 0x00010000,
    VK_ACCESS_SHADER_READ_BIT = 0x00000020,
    VK_ACCESS_SHADER_WRITE_BIT = 0x00000040,
    VK_ACCESS_TRANSFER_READ_BIT = 0x00000800,
    VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000,
};

enum {
    VK_QUERY_TYPE_TIMESTAMP = 2,
    VK_QUERY_RESULT_64_BIT = 0x00000001,
    VK_QUERY_RESULT_WAIT_BIT = 0x00000002,
};

typedef struct VkApplicationInfo {
    VkStructureType sType;
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkExtent3D { uint32_t width, height, depth; } VkExtent3D;

typedef struct VkQueueFamilyProperties {
    VkFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    VkExtent3D minImageTransferGranularity;
} VkQueueFamilyProperties;

typedef struct VkMemoryType { VkFlags propertyFlags; uint32_t heapIndex; } VkMemoryType;
typedef struct VkMemoryHeap { VkDeviceSize size; VkFlags flags; } VkMemoryHeap;

typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties;

// Full VkPhysicalDeviceLimits (exact Khronos layout up to 504 bytes)
typedef struct VkPhysicalDeviceLimits {
    uint32_t maxImageDimension1D;
    uint32_t maxImageDimension2D;
    uint32_t maxImageDimension3D;
    uint32_t maxImageDimensionCube;
    uint32_t maxImageArrayLayers;
    uint32_t maxTexelBufferElements;
    uint32_t maxUniformBufferRange;
    uint32_t maxStorageBufferRange;
    uint32_t maxPushConstantsSize;
    uint32_t maxMemoryAllocationCount;
    uint32_t maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity;
    VkDeviceSize sparseAddressSpaceSize;
    uint32_t maxBoundDescriptorSets;
    uint32_t maxPerStageDescriptorSamplers;
    uint32_t maxPerStageDescriptorUniformBuffers;
    uint32_t maxPerStageDescriptorStorageBuffers;
    uint32_t maxPerStageDescriptorSampledImages;
    uint32_t maxPerStageDescriptorStorageImages;
    uint32_t maxPerStageDescriptorInputAttachments;
    uint32_t maxPerStageResources;
    uint32_t maxDescriptorSetSamplers;
    uint32_t maxDescriptorSetUniformBuffers;
    uint32_t maxDescriptorSetUniformBuffersDynamic;
    uint32_t maxDescriptorSetStorageBuffers;
    uint32_t maxDescriptorSetStorageBuffersDynamic;
    uint32_t maxDescriptorSetSampledImages;
    uint32_t maxDescriptorSetStorageImages;
    uint32_t maxDescriptorSetInputAttachments;
    uint32_t maxVertexInputAttributes;
    uint32_t maxVertexInputBindings;
    uint32_t maxVertexInputAttributeOffset;
    uint32_t maxVertexInputBindingStride;
    uint32_t maxVertexOutputComponents;
    uint32_t maxTessellationGenerationLevel;
    uint32_t maxTessellationPatchSize;
    uint32_t maxTessellationControlPerVertexInputComponents;
    uint32_t maxTessellationControlPerVertexOutputComponents;
    uint32_t maxTessellationControlPerPatchOutputComponents;
    uint32_t maxTessellationControlTotalOutputComponents;
    uint32_t maxTessellationEvaluationInputComponents;
    uint32_t maxTessellationEvaluationOutputComponents;
    uint32_t maxGeometryShaderInvocations;
    uint32_t maxGeometryInputComponents;
    uint32_t maxGeometryOutputComponents;
    uint32_t maxGeometryOutputVertices;
    uint32_t maxGeometryTotalOutputComponents;
    uint32_t maxFragmentInputComponents;
    uint32_t maxFragmentOutputAttachments;
    uint32_t maxFragmentDualSrcAttachments;
    uint32_t maxFragmentCombinedOutputResources;
    uint32_t maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupCount[3];
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxComputeWorkGroupSize[3];
    uint32_t subPixelPrecisionBits;
    uint32_t subTexelPrecisionBits;
    uint32_t mipmapPrecisionBits;
    uint32_t maxDrawIndexedIndexValue;
    uint32_t maxDrawIndirectCount;
    float maxSamplerLodBias;
    float maxSamplerAnisotropy;
    uint32_t maxViewports;
    uint32_t maxViewportDimensions[2];
    float viewportBoundsRange[2];
    uint32_t viewportSubPixelBits;
    size_t minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment;
    VkDeviceSize minUniformBufferOffsetAlignment;
    VkDeviceSize minStorageBufferOffsetAlignment;
    int32_t minTexelOffset;
    uint32_t maxTexelOffset;
    int32_t minTexelGatherOffset;
    uint32_t maxTexelGatherOffset;
    float minInterpolationOffset;
    float maxInterpolationOffset;
    uint32_t subPixelInterpolationOffsetBits;
    uint32_t maxFramebufferWidth;
    uint32_t maxFramebufferHeight;
    uint32_t maxFramebufferLayers;
    VkFlags framebufferColorSampleCounts;
    VkFlags framebufferDepthSampleCounts;
    VkFlags framebufferStencilSampleCounts;
    VkFlags framebufferNoAttachmentsSampleCounts;
    uint32_t maxColorAttachments;
    VkFlags sampledImageColorSampleCounts;
    VkFlags sampledImageIntegerSampleCounts;
    VkFlags sampledImageDepthSampleCounts;
    VkFlags sampledImageStencilSampleCounts;
    VkFlags storageImageSampleCounts;
    uint32_t maxSampleMaskWords;
    VkBool32 timestampComputeAndGraphics;
    float timestampPeriod;
    uint32_t maxClipDistances;
    uint32_t maxCullDistances;
    uint32_t maxCombinedClipAndCullDistances;
    uint32_t discreteQueuePriorities;
    float pointSizeRange[2];
    float lineWidthRange[2];
    float pointSizeGranularity;
    float lineWidthGranularity;
    VkBool32 strictLines;
    VkBool32 standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment;
    VkDeviceSize optimalBufferCopyRowPitchAlignment;
    VkDeviceSize nonCoherentAtomSize;
} VkPhysicalDeviceLimits;

typedef struct VkPhysicalDeviceSparseProperties {
    VkBool32 residencyStandard2DBlockShape;
    VkBool32 residencyStandard2DMultisampleBlockShape;
    VkBool32 residencyStandard3DBlockShape;
    VkBool32 residencyAlignedMipSize;
    VkBool32 residencyNonResidentStrict;
} VkPhysicalDeviceSparseProperties;

typedef struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
} VkPhysicalDeviceProperties;

typedef struct VkPhysicalDeviceSubgroupProperties {
    VkStructureType sType;
    void* pNext;
    uint32_t subgroupSize;
    VkFlags supportedStages;
    VkFlags supportedOperations;
    VkBool32 quadOperationsInAllStages;
} VkPhysicalDeviceSubgroupProperties;

typedef struct VkPhysicalDeviceProperties2 {
    VkStructureType sType;
    void* pNext;
    VkPhysicalDeviceProperties properties;
} VkPhysicalDeviceProperties2;

typedef struct VkPhysicalDeviceVulkan11Features {
    VkStructureType sType;
    void* pNext;
    VkBool32 storageBuffer16BitAccess;
    VkBool32 uniformAndStorageBuffer16BitAccess;
    VkBool32 storagePushConstant16;
    VkBool32 storageInputOutput16;
    VkBool32 multiview;
    VkBool32 multiviewGeometryShader;
    VkBool32 multiviewTessellationShader;
    VkBool32 variablePointersStorageBuffer;
    VkBool32 variablePointers;
    VkBool32 protectedMemory;
    VkBool32 samplerYcbcrConversion;
    VkBool32 shaderDrawParameters;
} VkPhysicalDeviceVulkan11Features;

typedef struct VkPhysicalDeviceVulkan12Features {
    VkStructureType sType;
    void* pNext;
    VkBool32 samplerMirrorClampToEdge;
    VkBool32 drawIndirectCount;
    VkBool32 storageBuffer8BitAccess;
    VkBool32 uniformAndStorageBuffer8BitAccess;
    VkBool32 storagePushConstant8;
    VkBool32 shaderBufferInt64Atomics;
    VkBool32 shaderSharedInt64Atomics;
    VkBool32 shaderFloat16;
    VkBool32 shaderInt8;
    VkBool32 descriptorIndexing;
    VkBool32 shaderInputAttachmentArrayDynamicIndexing;
    VkBool32 shaderUniformTexelBufferArrayDynamicIndexing;
    VkBool32 shaderStorageTexelBufferArrayDynamicIndexing;
    VkBool32 shaderUniformBufferArrayNonUniformIndexing;
    VkBool32 shaderSampledImageArrayNonUniformIndexing;
    VkBool32 shaderStorageBufferArrayNonUniformIndexing;
    VkBool32 shaderStorageImageArrayNonUniformIndexing;
    VkBool32 shaderInputAttachmentArrayNonUniformIndexing;
    VkBool32 shaderUniformTexelBufferArrayNonUniformIndexing;
    VkBool32 shaderStorageTexelBufferArrayNonUniformIndexing;
    VkBool32 descriptorBindingUniformBufferUpdateAfterBind;
    VkBool32 descriptorBindingSampledImageUpdateAfterBind;
    VkBool32 descriptorBindingStorageImageUpdateAfterBind;
    VkBool32 descriptorBindingStorageBufferUpdateAfterBind;
    VkBool32 descriptorBindingUniformTexelBufferUpdateAfterBind;
    VkBool32 descriptorBindingStorageTexelBufferUpdateAfterBind;
    VkBool32 descriptorBindingUpdateUnusedWhilePending;
    VkBool32 descriptorBindingPartiallyBound;
    VkBool32 descriptorBindingVariableDescriptorCount;
    VkBool32 runtimeDescriptorArray;
    VkBool32 samplerFilterMinmax;
    VkBool32 scalarBlockLayout;
    VkBool32 imagelessFramebuffer;
    VkBool32 uniformBufferStandardLayout;
    VkBool32 shaderSubgroupExtendedTypes;
    VkBool32 separateDepthStencilLayouts;
    VkBool32 hostQueryReset;
    VkBool32 timelineSemaphore;
    VkBool32 bufferDeviceAddress;
    VkBool32 bufferDeviceAddressCaptureReplay;
    VkBool32 bufferDeviceAddressMultiDevice;
    VkBool32 vulkanMemoryModel;
    VkBool32 vulkanMemoryModelDeviceScope;
    VkBool32 vulkanMemoryModelAvailabilityVisibilityChains;
    VkBool32 shaderOutputViewportIndex;
    VkBool32 shaderOutputLayer;
    VkBool32 subgroupBroadcastDynamicId;
} VkPhysicalDeviceVulkan12Features;

typedef struct VkPhysicalDeviceFeatures {
    VkBool32 _pad[55]; // shaderInt64 is index 51, shaderInt16 is index 52
} VkPhysicalDeviceFeatures;

typedef struct VkPhysicalDeviceFeatures2 {
    VkStructureType sType;
    void* pNext;
    VkPhysicalDeviceFeatures features;
} VkPhysicalDeviceFeatures2;

typedef struct VkDeviceQueueCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkDeviceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures* pEnabledFeatures;
} VkDeviceCreateInfo;

typedef struct VkBufferCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkFlags usage;
    uint32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
} VkMemoryRequirements;

typedef struct VkMemoryAllocateFlagsInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t deviceMask;
} VkMemoryAllocateFlagsInfo;

typedef struct VkMemoryAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct VkBufferDeviceAddressInfo {
    VkStructureType sType;
    const void* pNext;
    VkBuffer buffer;
} VkBufferDeviceAddressInfo;

typedef struct VkShaderModuleCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    size_t codeSize;
    const uint32_t* pCode;
} VkShaderModuleCreateInfo;

typedef struct VkDescriptorSetLayoutBinding {
    uint32_t binding;
    uint32_t descriptorType;
    uint32_t descriptorCount;
    VkFlags stageFlags;
    const void* pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct VkPushConstantRange {
    VkFlags stageFlags;
    uint32_t offset;
    uint32_t size;
} VkPushConstantRange;

typedef struct VkPipelineLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount;
    const VkPushConstantRange* pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkFlags stage;
    VkShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct VkComputePipelineCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

typedef struct VkDescriptorPoolSize {
    uint32_t type;
    uint32_t descriptorCount;
} VkDescriptorPoolSize;

typedef struct VkDescriptorPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
} VkDescriptorPoolCreateInfo;

typedef struct VkDescriptorSetAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
} VkDescriptorSetAllocateInfo;

typedef struct VkDescriptorBufferInfo {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
} VkDescriptorBufferInfo;

typedef struct VkWriteDescriptorSet {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    uint32_t descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
} VkWriteDescriptorSet;

typedef struct VkCommandPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
} VkCommandPoolCreateInfo;

typedef struct VkCommandBufferAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkCommandPool commandPool;
    uint32_t level;
    uint32_t commandBufferCount;
} VkCommandBufferAllocateInfo;

typedef struct VkCommandBufferBeginInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const void* pInheritanceInfo;
} VkCommandBufferBeginInfo;

typedef struct VkMemoryBarrier {
    VkStructureType sType;
    const void* pNext;
    VkFlags srcAccessMask;
    VkFlags dstAccessMask;
} VkMemoryBarrier;

typedef struct VkBufferCopy {
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
} VkBufferCopy;

typedef struct VkSubmitInfo {
    VkStructureType sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const void* pWaitSemaphores;
    const VkFlags* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void* pSignalSemaphores;
} VkSubmitInfo;

typedef struct VkFenceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
} VkFenceCreateInfo;

typedef struct VkQueryPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queryType;
    uint32_t queryCount;
    VkFlags pipelineStatistics;
} VkQueryPoolCreateInfo;

typedef struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR {
    VkStructureType sType; // 1000506000
    void* pNext;
    VkBool32 cooperativeMatrix;
    VkBool32 cooperativeMatrixRobustBufferAccess;
} VkPhysicalDeviceCooperativeMatrixFeaturesKHR;

typedef struct VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType sType; // 1000141000
    void* pNext;
    VkBool32 shaderBFloat16Type;
    VkBool32 shaderBFloat16DotProduct;
    VkBool32 shaderBFloat16CooperativeMatrix;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;


VkResult vkCreateInstance(const VkInstanceCreateInfo*, const void*, VkInstance*);
void     vkDestroyInstance(VkInstance, const void*);
VkResult vkEnumeratePhysicalDevices(VkInstance, uint32_t*, VkPhysicalDevice*);
void     vkGetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties*);
void     vkGetPhysicalDeviceProperties2(VkPhysicalDevice, VkPhysicalDeviceProperties2*);
void     vkGetPhysicalDeviceFeatures2(VkPhysicalDevice, VkPhysicalDeviceFeatures2*);
void     vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
void     vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
VkResult vkCreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
void     vkDestroyDevice(VkDevice, const void*);
void     vkGetDeviceQueue(VkDevice, uint32_t, uint32_t, VkQueue*);
VkResult vkCreateBuffer(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
void     vkDestroyBuffer(VkDevice, VkBuffer, const void*);
void     vkGetBufferMemoryRequirements(VkDevice, VkBuffer, VkMemoryRequirements*);
VkResult vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
void     vkFreeMemory(VkDevice, VkDeviceMemory, const void*);
VkResult vkBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
VkResult vkMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
void     vkUnmapMemory(VkDevice, VkDeviceMemory);
uint64_t vkGetBufferDeviceAddress(VkDevice, const VkBufferDeviceAddressInfo*);
VkResult vkCreateShaderModule(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
void     vkDestroyShaderModule(VkDevice, VkShaderModule, const void*);
VkResult vkCreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*);
void     vkDestroyDescriptorSetLayout(VkDevice, VkDescriptorSetLayout, const void*);
VkResult vkCreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
void     vkDestroyPipelineLayout(VkDevice, VkPipelineLayout, const void*);
VkResult vkCreateComputePipelines(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const void*, VkPipeline*);
void     vkDestroyPipeline(VkDevice, VkPipeline, const void*);
VkResult vkCreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
void     vkDestroyDescriptorPool(VkDevice, VkDescriptorPool, const void*);
VkResult vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
void     vkUpdateDescriptorSets(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*);
VkResult vkCreateCommandPool(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
void     vkDestroyCommandPool(VkDevice, VkCommandPool, const void*);
VkResult vkAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
VkResult vkResetCommandBuffer(VkCommandBuffer, VkFlags);
VkResult vkBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*);
VkResult vkEndCommandBuffer(VkCommandBuffer);
void     vkCmdBindPipeline(VkCommandBuffer, uint32_t, VkPipeline);
void     vkCmdBindDescriptorSets(VkCommandBuffer, uint32_t, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
void     vkCmdPushConstants(VkCommandBuffer, VkPipelineLayout, VkFlags, uint32_t, uint32_t, const void*);
void     vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
void     vkCmdPipelineBarrier(VkCommandBuffer, VkFlags, VkFlags, VkFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const void*, uint32_t, const void*);
void     vkCmdCopyBuffer(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*);
void     vkCmdFillBuffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t);
void     vkCmdResetQueryPool(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t);
void     vkCmdWriteTimestamp(VkCommandBuffer, VkFlags, VkQueryPool, uint32_t);
VkResult vkCreateQueryPool(VkDevice, const VkQueryPoolCreateInfo*, const void*, VkQueryPool*);
void     vkDestroyQueryPool(VkDevice, VkQueryPool, const void*);
VkResult vkGetQueryPoolResults(VkDevice, VkQueryPool, uint32_t, uint32_t, size_t, void*, VkDeviceSize, VkFlags);
VkResult vkCreateFence(VkDevice, const VkFenceCreateInfo*, const void*, VkFence*);
void     vkDestroyFence(VkDevice, VkFence, const void*);
VkResult vkResetFences(VkDevice, uint32_t, const VkFence*);
VkResult vkWaitForFences(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t);
VkResult vkQueueSubmit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
VkResult vkQueueWaitIdle(VkQueue);

} // extern "C"
