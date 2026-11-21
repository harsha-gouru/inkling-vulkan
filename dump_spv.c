#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

static int g_idx = 0;
static PFN_vkGetInstanceProcAddr g_next_gipa = NULL;
static PFN_vkGetDeviceProcAddr g_next_gdpa = NULL;
static PFN_vkCreateShaderModule g_next_csm = NULL;

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateShaderModule(
    VkDevice dev, const VkShaderModuleCreateInfo* pCI, const VkAllocationCallbacks* pA, VkShaderModule* pM)
{
    const char *dir = getenv("DUMP_SPV_DIR");
    if (dir && pCI && pCI->pCode && pCI->codeSize > 0) {
        char path[512];
        snprintf(path, sizeof(path), "%s/shader_%02d.spv", dir, g_idx++);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(pCI->pCode, 1, pCI->codeSize, f); fclose(f); }
    }
    return g_next_csm(dev, pCI, pA, pM);
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetDeviceProcAddr(VkDevice dev, const char* name) {
    if (strcmp(name, "vkCreateShaderModule") == 0) return (PFN_vkVoidFunction)layer_CreateShaderModule;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)layer_GetDeviceProcAddr;
    return g_next_gdpa(dev, name);
}

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateDevice(
    VkPhysicalDevice pd, const VkDeviceCreateInfo* pCI, const VkAllocationCallbacks* pA, VkDevice* pDev)
{
    VkLayerDeviceCreateInfo *lci = (VkLayerDeviceCreateInfo*)pCI->pNext;
    while (lci && (lci->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO || lci->function != VK_LAYER_LINK_INFO))
        lci = (VkLayerDeviceCreateInfo*)lci->pNext;
    PFN_vkGetInstanceProcAddr gipa = lci->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    g_next_gdpa = lci->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    lci->u.pLayerInfo = lci->u.pLayerInfo->pNext;
    PFN_vkCreateDevice fn = (PFN_vkCreateDevice)gipa(VK_NULL_HANDLE, "vkCreateDevice");
    VkResult res = fn(pd, pCI, pA, pDev);
    if (res == VK_SUCCESS) {
        g_next_csm = (PFN_vkCreateShaderModule)g_next_gdpa(*pDev, "vkCreateShaderModule");
    }
    return res;
}

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateInstance(
    const VkInstanceCreateInfo* pCI, const VkAllocationCallbacks* pA, VkInstance* pInst)
{
    VkLayerInstanceCreateInfo *lci = (VkLayerInstanceCreateInfo*)pCI->pNext;
    while (lci && (lci->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO || lci->function != VK_LAYER_LINK_INFO))
        lci = (VkLayerInstanceCreateInfo*)lci->pNext;
    g_next_gipa = lci->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    lci->u.pLayerInfo = lci->u.pLayerInfo->pNext;
    PFN_vkCreateInstance fn = (PFN_vkCreateInstance)g_next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    return fn(pCI, pA, pInst);
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetInstanceProcAddr(VkInstance inst, const char* name) {
    if (strcmp(name, "vkCreateInstance") == 0) return (PFN_vkVoidFunction)layer_CreateInstance;
    if (strcmp(name, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)layer_CreateDevice;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)layer_GetDeviceProcAddr;
    if (strcmp(name, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)layer_GetInstanceProcAddr;
    return g_next_gipa(inst, name);
}

VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *p) {
    p->loaderLayerInterfaceVersion = 2;
    p->pfnGetInstanceProcAddr = layer_GetInstanceProcAddr;
    p->pfnGetDeviceProcAddr = layer_GetDeviceProcAddr;
    return VK_SUCCESS;
}
