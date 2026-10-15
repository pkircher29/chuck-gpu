/* C implementation of the CUDA Driver API, CUDA Runtime API, & NVML shim for Linux/POSIX
 * (compiled to libcuda.so.1, libcudart.so.12, libnvidia-ml.so.1)
 */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static void bridge_notify(const char* json_line) {
    const char* addr_env = getenv("CHUCK_GPU_BRIDGE_ADDR");
    char host[64] = "127.0.0.1";
    int port = 8771;
    if (addr_env && *addr_env) {
        const char* colon = strchr(addr_env, ':');
        if (colon) {
            size_t len = (size_t)(colon - addr_env);
            if (len < sizeof(host)) {
                memcpy(host, addr_env, len);
                host[len] = '\0';
            }
            port = atoi(colon + 1);
        }
    }
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return;
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, host, &sa.sin_addr);
    if (connect(sock, (struct sockaddr*)&sa, sizeof(sa)) == 0) {
        send(sock, json_line, strlen(json_line), 0);
        send(sock, "\n", 1, 0);
        char resp[256];
        recv(sock, resp, sizeof(resp) - 1, 0);
    }
    close(sock);
}

int cuInit(unsigned int flags) {
    if (flags != 0) return 1;
    bridge_notify("{\"op\":\"cuInit\"}");
    return 0;
}
int cuDriverGetVersion(int* v) { if (!v) return 1; *v = 12040; return 0; }
int cuDeviceGetCount(int* c) { if (!c) return 1; *c = 1; return 0; }
int cuDeviceGet(int* d, int ord) { if (!d || ord != 0) return 101; *d = 0; return 0; }
int cuDeviceGetName(char* name, int len, int dev) {
    if (!name || len <= 0 || dev != 0) return 1;
    const char* model = getenv("CHUCK_GPU_MODEL");
    char buf[128];
    snprintf(buf, sizeof(buf), "Modal NVIDIA %s (chuck-gpu)", (model && *model) ? model : "Tesla T4");
    strncpy(name, buf, (size_t)(len - 1));
    name[len - 1] = '\0';
    return 0;
}
int cuDeviceTotalMem_v2(size_t* bytes, int dev) {
    if (!bytes || dev != 0) return 1;
    *bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
    return 0;
}
int cuDeviceTotalMem(size_t* bytes, int dev) { return cuDeviceTotalMem_v2(bytes, dev); }
int cuDeviceComputeCapability(int* maj, int* min, int dev) {
    if (!maj || !min || dev != 0) return 1;
    *maj = 7;
    *min = 5;
    return 0;
}
int cuMemAlloc_v2(uint64_t* dptr, size_t bytesize) {
    if (!dptr || bytesize == 0) return 1;
    void* p = calloc(1, bytesize);
    if (!p) return 2;
    *dptr = (uint64_t)(uintptr_t)p;
    char msg[128];
    snprintf(msg, sizeof(msg), "{\"op\":\"cuMemAlloc\",\"dptr\":%llu,\"size\":%zu}",
             (unsigned long long)*dptr, bytesize);
    bridge_notify(msg);
    return 0;
}
int cuMemAlloc(uint64_t* dptr, size_t bytesize) { return cuMemAlloc_v2(dptr, bytesize); }
int cuMemcpyHtoD_v2(uint64_t dst, const void* src, size_t count) {
    if (!dst || !src) return 1;
    memcpy((void*)(uintptr_t)dst, src, count);
    return 0;
}
int cuMemcpyHtoD(uint64_t dst, const void* src, size_t count) { return cuMemcpyHtoD_v2(dst, src, count); }
int cuMemcpyDtoH_v2(void* dst, uint64_t src, size_t count) {
    if (!dst || !src) return 1;
    memcpy(dst, (const void*)(uintptr_t)src, count);
    return 0;
}
int cuMemcpyDtoH(void* dst, uint64_t src, size_t count) { return cuMemcpyDtoH_v2(dst, src, count); }
int cuMemFree_v2(uint64_t dptr) {
    if (dptr) free((void*)(uintptr_t)dptr);
    return 0;
}
int cuMemFree(uint64_t dptr) { return cuMemFree_v2(dptr); }
int cuCtxSynchronize(void) { bridge_notify("{\"op\":\"cuCtxSynchronize\"}"); return 0; }
int cudaGetDeviceCount(int* c) { return cuDeviceGetCount(c); }
int cudaGetDevice(int* d) { if (!d) return 1; *d = 0; return 0; }
int cudaSetDevice(int d) { return d == 0 ? 0 : 101; }
int cudaDriverGetVersion(int* v) { return cuDriverGetVersion(v); }
int cudaRuntimeGetVersion(int* v) { if (!v) return 1; *v = 12040; return 0; }
int cudaDeviceSynchronize(void) { return cuCtxSynchronize(); }
int nvmlInit_v2(void) { return 0; }
int nvmlShutdown(void) { return 0; }
int nvmlDeviceGetCount_v2(unsigned int* cnt) { if (!cnt) return 2; *cnt = 1; return 0; }
