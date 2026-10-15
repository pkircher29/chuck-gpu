//! Native CUDA Driver API (`nvcuda.dll` / `libcuda.so.1`), CUDA Runtime API (`cudart64_12.dll` / `libcudart.so`),
//! and NVML (`nvml.dll` / `libnvidia-ml.so.1`) virtual GPU driver for `chuck-gpu`.
//!
//! Exports standard NVIDIA CUDA Driver API (`cu*`), CUDA Runtime API (`cuda*`), and NVML (`nvml*`)
//! symbols so native applications, PyTorch/CUDA libraries, and `nvidia-smi` detect a virtual
//! Modal GPU and forward memory, PTX/CUBIN module loading, and kernel execution over TCP RPC
//! to the local `chuck-gpu` Modal driver daemon (`127.0.0.1:8771` by default, or `CHUCK_GPU_BRIDGE_ADDR`).

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void, CStr};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::ptr;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUdeviceptr = c_ulonglong;
pub type NvmlReturn = c_int;
pub type NvmlDevice = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;

pub const NVML_SUCCESS: NvmlReturn = 0;
pub const NVML_ERROR_INVALID_ARGUMENT: NvmlReturn = 2;

struct VirtualGpuConfig {
    name: String,
    vram_bytes: u64,
    cc_major: c_int,
    cc_minor: c_int,
    multiprocessors: c_int,
    clock_khz: c_int,
    mem_clock_khz: c_int,
    bus_width: c_int,
    driver_version: c_int,
    bridge_addr: String,
}

struct DriverState {
    initialized: bool,
    config: VirtualGpuConfig,
    next_handle: usize,
    allocations: HashMap<CUdeviceptr, Vec<u8>>,
    modules: HashMap<usize, Vec<u8>>,
    functions: HashMap<usize, (usize, String)>,
}

static STATE: OnceLock<Mutex<DriverState>> = OnceLock::new();

fn gpu_preset(model: &str) -> (String, u64, c_int, c_int, c_int) {
    match model.to_uppercase().as_str() {
        "L4" => ("Modal NVIDIA L4 (chuck-gpu)".into(), 24 * 1024 * 1024 * 1024, 8, 9, 58),
        "A10G" => ("Modal NVIDIA A10G (chuck-gpu)".into(), 24 * 1024 * 1024 * 1024, 8, 6, 80),
        "L40S" => ("Modal NVIDIA L40S (chuck-gpu)".into(), 48 * 1024 * 1024 * 1024, 8, 9, 142),
        "A100" => ("Modal NVIDIA A100-SXM4-40GB (chuck-gpu)".into(), 40 * 1024 * 1024 * 1024, 8, 0, 108),
        "A100-80GB" => ("Modal NVIDIA A100-SXM4-80GB (chuck-gpu)".into(), 80 * 1024 * 1024 * 1024, 8, 0, 108),
        "H100" => ("Modal NVIDIA H100 80GB HBM3 (chuck-gpu)".into(), 80 * 1024 * 1024 * 1024, 9, 0, 132),
        "H200" => ("Modal NVIDIA H200 141GB (chuck-gpu)".into(), 141 * 1024 * 1024 * 1024, 9, 0, 132),
        "B200" => ("Modal NVIDIA B200 (chuck-gpu)".into(), 192 * 1024 * 1024 * 1024, 10, 0, 160),
        _ => ("Modal NVIDIA Tesla T4 (chuck-gpu)".into(), 16 * 1024 * 1024 * 1024, 7, 5, 40),
    }
}

fn get_state() -> &'static Mutex<DriverState> {
    STATE.get_or_init(|| {
        let model = std::env::var("CHUCK_GPU_MODEL").unwrap_or_else(|_| "T4".into());
        let (name, vram_bytes, cc_major, cc_minor, multiprocessors) = gpu_preset(&model);
        let bridge_addr = std::env::var("CHUCK_GPU_BRIDGE_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8771".into());
        Mutex::new(DriverState {
            initialized: false,
            config: VirtualGpuConfig {
                name,
                vram_bytes,
                cc_major,
                cc_minor,
                multiprocessors,
                clock_khz: 1590000,
                mem_clock_khz: 5001000,
                bus_width: 256,
                driver_version: 12040,
                bridge_addr,
            },
            next_handle: 0x1000_0000,
            allocations: HashMap::new(),
            modules: HashMap::new(),
            functions: HashMap::new(),
        })
    })
}

fn bridge_rpc(addr: &str, payload: &str) -> Option<String> {
    let sock_addr = addr.parse().ok()?;
    let mut stream = TcpStream::connect_timeout(&sock_addr, Duration::from_millis(1500)).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    stream.write_all(payload.as_bytes()).ok()?;
    stream.write_all(b"\n").ok()?;
    stream.flush().ok()?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    Some(line)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

fn extract_json_str<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{}\":\"", key);
    let start = json.find(&pat)? + pat.len();
    let rest = &json[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

// ── CUDA Driver API (`cu*`) ──────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cuInit(flags: c_uint) -> CUresult {
    if flags != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut st = get_state().lock().unwrap();
    st.initialized = true;
    let _ = bridge_rpc(&st.config.bridge_addr, r#"{"op":"cuInit"}"#);
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDriverGetVersion(driver_version: *mut c_int) -> CUresult {
    if driver_version.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let st = get_state().lock().unwrap();
    unsafe {
        *driver_version = st.config.driver_version;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    if count.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        *count = 1;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    if device.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ordinal != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    unsafe {
        *device = 0;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    if name.is_null() || len <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let st = get_state().lock().unwrap();
    let bytes = st.config.name.as_bytes();
    let copy_len = bytes.len().min((len - 1) as usize);
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, name, copy_len);
        *name.add(copy_len) = 0;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    if bytes.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let st = get_state().lock().unwrap();
    unsafe {
        *bytes = st.config.vram_bytes as usize;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDeviceTotalMem(bytes: *mut usize, dev: CUdevice) -> CUresult {
    cuDeviceTotalMem_v2(bytes, dev)
}

#[no_mangle]
pub extern "C" fn cuDeviceComputeCapability(
    major: *mut c_int,
    minor: *mut c_int,
    dev: CUdevice,
) -> CUresult {
    if major.is_null() || minor.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let st = get_state().lock().unwrap();
    unsafe {
        *major = st.config.cc_major;
        *minor = st.config.cc_minor;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDeviceGetAttribute(
    pi: *mut c_int,
    attrib: c_int,
    dev: CUdevice,
) -> CUresult {
    if pi.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let st = get_state().lock().unwrap();
    let val = match attrib {
        1 => 1024, // MAX_THREADS_PER_BLOCK
        2 => 1024, // MAX_BLOCK_DIM_X
        3 => 1024, // MAX_BLOCK_DIM_Y
        4 => 64,   // MAX_BLOCK_DIM_Z
        5 => 2147483647, // MAX_GRID_DIM_X
        6 => 65535, // MAX_GRID_DIM_Y
        7 => 65535, // MAX_GRID_DIM_Z
        8 => 49152, // MAX_SHARED_MEMORY_PER_BLOCK
        10 => 32,   // WARP_SIZE
        13 => st.config.clock_khz, // CLOCK_RATE
        16 => st.config.multiprocessors, // MULTIPROCESSOR_COUNT
        36 => st.config.mem_clock_khz, // MEMORY_CLOCK_RATE
        37 => st.config.bus_width, // GLOBAL_MEMORY_BUS_WIDTH
        38 => 4194304, // L2_CACHE_SIZE
        75 => st.config.cc_major, // COMPUTE_CAPABILITY_MAJOR
        76 => st.config.cc_minor, // COMPUTE_CAPABILITY_MINOR
        _ => 1,
    };
    unsafe {
        *pi = val;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDeviceGetUuid(uuid: *mut u8, dev: CUdevice) -> CUresult {
    if uuid.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let fake_uuid: [u8; 16] = [
        0x43, 0x48, 0x55, 0x43, 0x4b, 0x47, 0x50, 0x55,
        0x4d, 0x4f, 0x44, 0x41, 0x4c, 0x00, 0x00, 0x01,
    ];
    unsafe {
        ptr::copy_nonoverlapping(fake_uuid.as_ptr(), uuid, 16);
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    unsafe {
        *pctx = 0x1 as CUcontext;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDevicePrimaryCtxRelease_v2(_dev: CUdevice) -> CUresult {
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult {
    cuDevicePrimaryCtxRelease_v2(dev)
}

#[no_mangle]
pub extern "C" fn cuCtxCreate_v2(
    pctx: *mut CUcontext,
    _flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    cuDevicePrimaryCtxRetain(pctx, dev)
}

#[no_mangle]
pub extern "C" fn cuCtxCreate(
    pctx: *mut CUcontext,
    flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    cuCtxCreate_v2(pctx, flags, dev)
}

#[no_mangle]
pub extern "C" fn cuCtxSetCurrent(_ctx: CUcontext) -> CUresult {
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    if pctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe {
        *pctx = 0x1 as CUcontext;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuCtxSynchronize() -> CUresult {
    let st = get_state().lock().unwrap();
    let _ = bridge_rpc(&st.config.bridge_addr, r#"{"op":"cuCtxSynchronize"}"#);
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuCtxDestroy_v2(_ctx: CUcontext) -> CUresult {
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuCtxDestroy(ctx: CUcontext) -> CUresult {
    cuCtxDestroy_v2(ctx)
}

#[no_mangle]
pub extern "C" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
    let st = get_state().lock().unwrap();
    let used: usize = st.allocations.values().map(|v| v.len()).sum();
    let tot = st.config.vram_bytes as usize;
    unsafe {
        if !free.is_null() {
            *free = tot.saturating_sub(used);
        }
        if !total.is_null() {
            *total = tot;
        }
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuMemGetInfo(free: *mut usize, total: *mut usize) -> CUresult {
    cuMemGetInfo_v2(free, total)
}

#[no_mangle]
pub extern "C" fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    if dptr.is_null() || bytesize == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut st = get_state().lock().unwrap();
    let handle = st.next_handle as CUdeviceptr;
    st.next_handle += bytesize.max(256);
    st.allocations.insert(handle, vec![0u8; bytesize]);
    let msg = format!(r#"{{"op":"cuMemAlloc","dptr":{},"size":{}}}"#, handle, bytesize);
    let _ = bridge_rpc(&st.config.bridge_addr, &msg);
    unsafe {
        *dptr = handle;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    cuMemAlloc_v2(dptr, bytesize)
}

#[no_mangle]
pub extern "C" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    let mut st = get_state().lock().unwrap();
    st.allocations.remove(&dptr);
    let msg = format!(r#"{{"op":"cuMemFree","dptr":{}}}"#, dptr);
    let _ = bridge_rpc(&st.config.bridge_addr, &msg);
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuMemFree(dptr: CUdeviceptr) -> CUresult {
    cuMemFree_v2(dptr)
}

#[no_mangle]
pub extern "C" fn cuMemcpyHtoD_v2(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult {
    if src_host.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut st = get_state().lock().unwrap();
    let slice = unsafe { std::slice::from_raw_parts(src_host as *const u8, byte_count) };
    if let Some(buf) = st.allocations.get_mut(&dst_device) {
        if byte_count <= buf.len() {
            buf[..byte_count].copy_from_slice(slice);
        }
    }
    let msg = format!(
        r#"{{"op":"cuMemcpyHtoD","dptr":{},"hex":"{}"}}"#,
        dst_device,
        hex_encode(slice)
    );
    let _ = bridge_rpc(&st.config.bridge_addr, &msg);
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuMemcpyHtoD(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult {
    cuMemcpyHtoD_v2(dst_device, src_host, byte_count)
}

#[no_mangle]
pub extern "C" fn cuMemcpyDtoH_v2(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    if dst_host.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut st = get_state().lock().unwrap();
    let msg = format!(
        r#"{{"op":"cuMemcpyDtoH","dptr":{},"size":{}}}"#,
        src_device, byte_count
    );
    if let Some(resp) = bridge_rpc(&st.config.bridge_addr, &msg) {
        if let Some(hex_str) = extract_json_str(&resp, "hex") {
            if let Some(decoded) = hex_decode(hex_str) {
                if let Some(buf) = st.allocations.get_mut(&src_device) {
                    let n = decoded.len().min(buf.len());
                    buf[..n].copy_from_slice(&decoded[..n]);
                }
            }
        }
    }
    if let Some(buf) = st.allocations.get(&src_device) {
        let copy_len = byte_count.min(buf.len());
        unsafe {
            ptr::copy_nonoverlapping(buf.as_ptr(), dst_host as *mut u8, copy_len);
        }
        return CUDA_SUCCESS;
    }
    CUDA_ERROR_INVALID_VALUE
}

#[no_mangle]
pub extern "C" fn cuMemcpyDtoH(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult {
    cuMemcpyDtoH_v2(dst_host, src_device, byte_count)
}

#[no_mangle]
pub extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    if module.is_null() || image.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut st = get_state().lock().unwrap();
    let cstr = unsafe { CStr::from_ptr(image as *const c_char) };
    let bytes = cstr.to_bytes().to_vec();
    let mod_id = st.next_handle;
    st.next_handle += 1;
    let msg = format!(
        r#"{{"op":"cuModuleLoadData","mod_id":{},"ptx_hex":"{}"}}"#,
        mod_id,
        hex_encode(&bytes)
    );
    let _ = bridge_rpc(&st.config.bridge_addr, &msg);
    st.modules.insert(mod_id, bytes);
    unsafe {
        *module = mod_id as CUmodule;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuModuleGetFunction(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    if hfunc.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let func_name = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
    let mut st = get_state().lock().unwrap();
    let fn_id = st.next_handle;
    st.next_handle += 1;
    st.functions.insert(fn_id, (hmod as usize, func_name));
    unsafe {
        *hfunc = fn_id as CUfunction;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuLaunchKernel(
    f: CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    _hstream: CUstream,
    _kernel_params: *mut *mut c_void,
    _extra: *mut *mut c_void,
) -> CUresult {
    let st = get_state().lock().unwrap();
    let fn_id = f as usize;
    let (mod_id, func_name) = st
        .functions
        .get(&fn_id)
        .cloned()
        .unwrap_or((0, "unknown".into()));
    let msg = format!(
        r#"{{"op":"cuLaunchKernel","mod_id":{},"func":"{}","grid":[{},{},{}],"block":[{},{},{}],"shared":{}}}"#,
        mod_id, func_name, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes
    );
    let _ = bridge_rpc(&st.config.bridge_addr, &msg);
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuGetErrorString(error: CUresult, pstr: *mut *const c_char) -> CUresult {
    if pstr.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let msg: &'static [u8] = match error {
        CUDA_SUCCESS => b"no error\0",
        CUDA_ERROR_INVALID_VALUE => b"invalid argument\0",
        CUDA_ERROR_OUT_OF_MEMORY => b"out of memory\0",
        CUDA_ERROR_NOT_INITIALIZED => b"initialization error\0",
        CUDA_ERROR_INVALID_DEVICE => b"invalid device ordinal\0",
        _ => b"CUDA driver error\0",
    };
    unsafe {
        *pstr = msg.as_ptr() as *const c_char;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuGetErrorName(error: CUresult, pstr: *mut *const c_char) -> CUresult {
    if pstr.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let msg: &'static [u8] = match error {
        CUDA_SUCCESS => b"CUDA_SUCCESS\0",
        CUDA_ERROR_INVALID_VALUE => b"CUDA_ERROR_INVALID_VALUE\0",
        CUDA_ERROR_OUT_OF_MEMORY => b"CUDA_ERROR_OUT_OF_MEMORY\0",
        CUDA_ERROR_NOT_INITIALIZED => b"CUDA_ERROR_NOT_INITIALIZED\0",
        CUDA_ERROR_INVALID_DEVICE => b"CUDA_ERROR_INVALID_DEVICE\0",
        _ => b"CUDA_ERROR_UNKNOWN\0",
    };
    unsafe {
        *pstr = msg.as_ptr() as *const c_char;
    }
    CUDA_SUCCESS
}

// ── CUDA Runtime API (`cuda*`) Exports ──────────────────────────────────────

#[no_mangle]
pub extern "C" fn cudaGetDeviceCount(count: *mut c_int) -> c_int {
    cuDeviceGetCount(count)
}

#[no_mangle]
pub extern "C" fn cudaGetDevice(device: *mut c_int) -> c_int {
    if device.is_null() {
        return 1;
    }
    unsafe {
        *device = 0;
    }
    0
}

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: c_int) -> c_int {
    if device != 0 {
        return 101;
    }
    0
}

#[no_mangle]
pub extern "C" fn cudaDriverGetVersion(driver_version: *mut c_int) -> c_int {
    cuDriverGetVersion(driver_version)
}

#[no_mangle]
pub extern "C" fn cudaRuntimeGetVersion(runtime_version: *mut c_int) -> c_int {
    if runtime_version.is_null() {
        return 1;
    }
    unsafe {
        *runtime_version = 12040;
    }
    0
}

#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> c_int {
    cuCtxSynchronize()
}

#[no_mangle]
pub extern "C" fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int {
    if dev_ptr.is_null() {
        return 1;
    }
    let mut dptr: CUdeviceptr = 0;
    let rc = cuMemAlloc_v2(&mut dptr, size);
    if rc == 0 {
        unsafe {
            *dev_ptr = dptr as usize as *mut c_void;
        }
    }
    rc
}

#[no_mangle]
pub extern "C" fn cudaFree(dev_ptr: *mut c_void) -> c_int {
    cuMemFree_v2(dev_ptr as usize as CUdeviceptr)
}

#[no_mangle]
pub extern "C" fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: c_int,
) -> c_int {
    // kind: 1 = HostToDevice, 2 = DeviceToHost, 3 = DeviceToDevice
    match kind {
        1 => cuMemcpyHtoD_v2(dst as usize as CUdeviceptr, src, count),
        2 => cuMemcpyDtoH_v2(dst, src as usize as CUdeviceptr, count),
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int {
    cuMemGetInfo_v2(free, total)
}

#[no_mangle]
pub extern "C" fn cudaGetLastError() -> c_int {
    0
}

#[no_mangle]
pub extern "C" fn cudaPeekAtLastError() -> c_int {
    0
}

#[no_mangle]
pub extern "C" fn cuGetProcAddress(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    _cuda_version: c_int,
    _flags: c_ulonglong,
) -> CUresult {
    if symbol.is_null() || pfn.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let name = unsafe { CStr::from_ptr(symbol) }.to_string_lossy();
    let base = name.split("_v").next().unwrap_or(&name);
    let ptr: *mut c_void = match base {
        "cuInit" => cuInit as *mut c_void,
        "cuDriverGetVersion" => cuDriverGetVersion as *mut c_void,
        "cuDeviceGetCount" => cuDeviceGetCount as *mut c_void,
        "cuDeviceGet" => cuDeviceGet as *mut c_void,
        "cuDeviceGetName" => cuDeviceGetName as *mut c_void,
        "cuDeviceTotalMem" => cuDeviceTotalMem_v2 as *mut c_void,
        "cuDeviceComputeCapability" => cuDeviceComputeCapability as *mut c_void,
        "cuDeviceGetAttribute" => cuDeviceGetAttribute as *mut c_void,
        "cuDeviceGetUuid" => cuDeviceGetUuid as *mut c_void,
        "cuDevicePrimaryCtxRetain" => cuDevicePrimaryCtxRetain as *mut c_void,
        "cuDevicePrimaryCtxRelease" => cuDevicePrimaryCtxRelease_v2 as *mut c_void,
        "cuCtxCreate" => cuCtxCreate_v2 as *mut c_void,
        "cuCtxSetCurrent" => cuCtxSetCurrent as *mut c_void,
        "cuCtxGetCurrent" => cuCtxGetCurrent as *mut c_void,
        "cuCtxSynchronize" => cuCtxSynchronize as *mut c_void,
        "cuCtxDestroy" => cuCtxDestroy_v2 as *mut c_void,
        "cuMemGetInfo" => cuMemGetInfo_v2 as *mut c_void,
        "cuMemAlloc" => cuMemAlloc_v2 as *mut c_void,
        "cuMemFree" => cuMemFree_v2 as *mut c_void,
        "cuMemcpyHtoD" => cuMemcpyHtoD_v2 as *mut c_void,
        "cuMemcpyDtoH" => cuMemcpyDtoH_v2 as *mut c_void,
        "cuModuleLoadData" => cuModuleLoadData as *mut c_void,
        "cuModuleGetFunction" => cuModuleGetFunction as *mut c_void,
        "cuLaunchKernel" => cuLaunchKernel as *mut c_void,
        "cuGetErrorString" => cuGetErrorString as *mut c_void,
        "cuGetErrorName" => cuGetErrorName as *mut c_void,
        _ => ptr::null_mut(),
    };
    if ptr.is_null() {
        return CUDA_ERROR_NOT_FOUND;
    }
    unsafe {
        *pfn = ptr;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: c_ulonglong,
    symbol_status: *mut c_int,
) -> CUresult {
    let rc = cuGetProcAddress(symbol, pfn, cuda_version, flags);
    if !symbol_status.is_null() {
        unsafe {
            *symbol_status = if rc == CUDA_SUCCESS { 1 } else { 0 };
        }
    }
    rc
}

// ── NVML Exports (`nvml.dll` / `libnvidia-ml.so.1`) ──────────────────────────

#[repr(C)]
pub struct NvmlMemory {
    pub total: c_ulonglong,
    pub free: c_ulonglong,
    pub used: c_ulonglong,
}

#[no_mangle]
pub extern "C" fn nvmlInit_v2() -> NvmlReturn {
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvmlInit() -> NvmlReturn {
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvmlShutdown() -> NvmlReturn {
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetCount_v2(device_count: *mut c_uint) -> NvmlReturn {
    if device_count.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    unsafe {
        *device_count = 1;
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetHandleByIndex_v2(
    index: c_uint,
    device: *mut NvmlDevice,
) -> NvmlReturn {
    if device.is_null() || index != 0 {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    unsafe {
        *device = 0x1 as NvmlDevice;
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetName(
    _device: NvmlDevice,
    name: *mut c_char,
    length: c_uint,
) -> NvmlReturn {
    if name.is_null() || length == 0 {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let st = get_state().lock().unwrap();
    let bytes = st.config.name.as_bytes();
    let copy_len = bytes.len().min((length - 1) as usize);
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, name, copy_len);
        *name.add(copy_len) = 0;
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetMemoryInfo(
    _device: NvmlDevice,
    memory: *mut NvmlMemory,
) -> NvmlReturn {
    if memory.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let st = get_state().lock().unwrap();
    let used = st.allocations.values().map(|v| v.len() as u64).sum::<u64>();
    let total = st.config.vram_bytes;
    unsafe {
        (*memory).total = total;
        (*memory).used = used;
        (*memory).free = total.saturating_sub(used);
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvmlSystemGetDriverVersion(version: *mut c_char, length: c_uint) -> NvmlReturn {
    if version.is_null() || length == 0 {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let ver = b"550.90.07\0";
    let copy_len = ver.len().min(length as usize);
    unsafe {
        ptr::copy_nonoverlapping(ver.as_ptr() as *const c_char, version, copy_len);
        *version.add(copy_len - 1) = 0;
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvmlSystemGetCudaDriverVersion_v2(cuda_driver_version: *mut c_int) -> NvmlReturn {
    if cuda_driver_version.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    unsafe {
        *cuda_driver_version = 12040;
    }
    NVML_SUCCESS
}
