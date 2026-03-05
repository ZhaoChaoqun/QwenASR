//! GPU weight conversion utilities — convert CPU weights to candle Tensors on Metal.

use candle_core::{DType, Device, Result, Tensor};

/// Convert an f32 slice to a candle Tensor on the given device.
///
/// Used for encoder weights (already loaded as Vec<f32>).
pub fn f32_to_tensor(data: &[f32], shape: &[usize], device: &Device) -> Result<Tensor> {
    Tensor::from_slice(data, shape, device)
}

/// Convert a raw BF16 pointer (mmap'd u16 values) to a candle Tensor on device.
///
/// Reads `n` u16 BF16 values, converts to f32, creates Tensor with F32 dtype,
/// then converts to F16 on device (Metal has native F16 but not BF16 support).
pub fn bf16_ptr_to_tensor(
    ptr: *const u16,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut f32_buf = vec![0.0f32; n];
    for i in 0..n {
        let bits = unsafe { *ptr.add(i) };
        f32_buf[i] = f32::from_bits((bits as u32) << 16);
    }
    let t = Tensor::from_slice(&f32_buf, shape, &Device::Cpu)?;
    t.to_dtype(DType::F16)?.to_device(device)
}

/// Convert INT8 quantized weight (data + per-channel scales) to F16 Tensor on device.
///
/// Dequantizes: f32[o,k] = int8[o,k] * scale[o], then converts to F16.
pub fn int8_to_f16_tensor(
    data: *const i8,
    scales: *const f32,
    out_dim: usize,
    in_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let n = out_dim * in_dim;
    let mut f32_buf = vec![0.0f32; n];
    for o in 0..out_dim {
        let scale = unsafe { *scales.add(o) };
        let row_offset = o * in_dim;
        for k in 0..in_dim {
            let val = unsafe { *data.add(row_offset + k) } as f32;
            f32_buf[row_offset + k] = val * scale;
        }
    }
    let t = Tensor::from_slice(&f32_buf, &[out_dim, in_dim], &Device::Cpu)?;
    t.to_dtype(DType::F16)?.to_device(device)
}
