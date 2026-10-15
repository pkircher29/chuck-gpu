use std::ffi::{c_void, CStr};
use nvcuda::*;

#[test]
fn test_virtual_cuda_driver_roundtrip() {
    unsafe {
        assert_eq!(cuInit(0), CUDA_SUCCESS);
        let mut ver = 0;
        assert_eq!(cuDriverGetVersion(&mut ver), CUDA_SUCCESS);
        assert_eq!(ver, 12040);

        let mut count = 0;
        assert_eq!(cuDeviceGetCount(&mut count), CUDA_SUCCESS);
        assert_eq!(count, 1);

        let mut dev = -1;
        assert_eq!(cuDeviceGet(&mut dev, 0), CUDA_SUCCESS);
        assert_eq!(dev, 0);

        let mut name_buf = [0i8; 128];
        assert_eq!(cuDeviceGetName(name_buf.as_mut_ptr(), 128, 0), CUDA_SUCCESS);
        let name = CStr::from_ptr(name_buf.as_ptr()).to_string_lossy();
        assert!(name.contains("Modal NVIDIA"));

        let mut total_mem = 0usize;
        assert_eq!(cuDeviceTotalMem_v2(&mut total_mem, 0), CUDA_SUCCESS);
        assert!(total_mem >= 16 * 1024 * 1024 * 1024);

        let mut major = 0;
        let mut minor = 0;
        assert_eq!(cuDeviceComputeCapability(&mut major, &mut minor, 0), CUDA_SUCCESS);
        assert!(major >= 7);

        let payload = b"modal-gpu-compute-tensor-12345";
        let mut dptr: CUdeviceptr = 0;
        assert_eq!(cuMemAlloc_v2(&mut dptr, payload.len()), CUDA_SUCCESS);
        assert!(dptr != 0);

        assert_eq!(
            cuMemcpyHtoD_v2(dptr, payload.as_ptr() as *const c_void, payload.len()),
            CUDA_SUCCESS
        );

        let mut out = vec![0u8; payload.len()];
        assert_eq!(
            cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, dptr, payload.len()),
            CUDA_SUCCESS
        );
        assert_eq!(&out[..], payload);

        assert_eq!(cuMemFree_v2(dptr), CUDA_SUCCESS);
    }
}
