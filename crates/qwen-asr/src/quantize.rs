/// INT8 per-channel symmetric quantization support.
///
/// File format V1 (`.qint8`):
/// ```text
/// [8 bytes]  magic: "QINT8\x01\x00\x00"
/// [4 bytes]  n_tensors: u32 LE
/// [4 bytes]  reserved: u32 (0)
///
/// -- Tensor table (n_tensors entries) --
/// Per tensor:
///   [2 bytes]  name_len: u16 LE
///   [name_len] name: UTF-8 bytes
///   [1 byte]   n_dims: u8
///   [n_dims*4] shape: [u32 LE; n_dims]
///   [8 bytes]  data_offset: u64 LE
///   [8 bytes]  scale_offset: u64 LE
/// ```
///
/// File format V2 (self-contained, includes encoder + embeddings):
/// ```text
/// [8 bytes]  magic: "QINT8\x02\x00\x00"
/// [4 bytes]  n_tensors: u32 LE
/// [4 bytes]  reserved: u32 (0)
///
/// -- Tensor table (n_tensors entries) --
/// Per tensor:
///   [2 bytes]  name_len: u16 LE
///   [name_len] name: UTF-8 bytes
///   [1 byte]   dtype: u8 (0=INT8, 1=F32, 2=BF16)
///   [1 byte]   n_dims: u8
///   [n_dims*4] shape: [u32 LE; n_dims]
///   [8 bytes]  data_offset: u64 LE
///   [8 bytes]  scale_offset: u64 LE (0 for F32/BF16)
/// ```

use std::collections::HashMap;
use std::os::unix::io::RawFd;

const MAGIC_V1: [u8; 8] = *b"QINT8\x01\x00\x00";
const MAGIC_V2: [u8; 8] = *b"QINT8\x02\x00\x00";

/// Data type tag for tensors in qint8 V2 files.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum QuantDtype {
    Int8 = 0,
    F32 = 1,
    BF16 = 2,
}

/// Metadata for a single tensor within a `.qint8` file.
#[derive(Clone, Debug)]
pub struct QuantTensorMeta {
    pub name: String,
    pub dtype: QuantDtype,
    pub shape: Vec<usize>,
    pub data_offset: usize,  // offset into data section
    pub scale_offset: usize, // offset into data section for f32 scales (INT8 only)
}

impl QuantTensorMeta {
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    /// Number of output channels (first dimension).
    pub fn out_dim(&self) -> usize {
        if self.shape.is_empty() { 0 } else { self.shape[0] }
    }

    /// Number of input features (product of remaining dimensions).
    pub fn in_dim(&self) -> usize {
        if self.shape.len() < 2 { 0 } else { self.shape[1..].iter().product() }
    }
}

/// A view into a quantized weight matrix (non-owning, points into mmap).
#[derive(Clone, Copy)]
pub struct QuantWeight {
    pub data: *const i8,     // INT8 weights, row-major [out_dim, in_dim]
    pub scales: *const f32,  // per-channel scale [out_dim]
    pub out_dim: usize,
    pub in_dim: usize,
}

unsafe impl Send for QuantWeight {}
unsafe impl Sync for QuantWeight {}

/// Owned quantized weight matrix (for fused weights like gate_up).
pub struct QuantWeightOwned {
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl QuantWeightOwned {
    /// Borrow as a non-owning `QuantWeight` view.
    pub fn as_ref(&self) -> QuantWeight {
        QuantWeight {
            data: self.data.as_ptr(),
            scales: self.scales.as_ptr(),
            out_dim: self.out_dim,
            in_dim: self.in_dim,
        }
    }
}

/// mmap-based reader for `.qint8` quantized model files.
pub struct QuantFile {
    _fd: RawFd,
    data: *mut u8,
    file_size: usize,
    data_section_offset: usize, // byte offset where data section begins in the file
    pub tensors: Vec<QuantTensorMeta>,
    tensor_map: HashMap<String, usize>,
}

unsafe impl Send for QuantFile {}
unsafe impl Sync for QuantFile {}

impl QuantFile {
    pub fn open(path: &str) -> Option<Self> {
        use libc::*;
        use std::ffi::CString;

        let c_path = CString::new(path).ok()?;
        let fd = unsafe { open(c_path.as_ptr(), O_RDONLY) };
        if fd < 0 {
            return None;
        }

        let mut stat_buf = unsafe { std::mem::zeroed::<stat>() };
        if unsafe { fstat(fd, &mut stat_buf) } < 0 {
            unsafe { close(fd); }
            return None;
        }

        let file_size = stat_buf.st_size as usize;
        if file_size < 16 {
            unsafe { close(fd); }
            return None;
        }

        let mmap_ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                file_size,
                PROT_READ,
                MAP_PRIVATE,
                fd,
                0,
            )
        };
        let raw_fd = fd;
        unsafe { close(fd); }

        if mmap_ptr == libc::MAP_FAILED {
            return None;
        }
        let data = mmap_ptr as *mut u8;

        // Verify magic (V1 or V2)
        let magic = unsafe { std::slice::from_raw_parts(data, 8) };
        let is_v2 = if magic == MAGIC_V2 {
            true
        } else if magic == MAGIC_V1 {
            false
        } else {
            unsafe { munmap(data as *mut _, file_size); }
            return None;
        };

        let n_tensors = unsafe { read_u32(data, 8) } as usize;
        // skip 4 bytes reserved
        let mut pos = 16usize;

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut tensor_map = HashMap::new();

        for _ in 0..n_tensors {
            if pos + 2 > file_size { break; }
            let name_len = unsafe { read_u16(data, pos) } as usize;
            pos += 2;

            if pos + name_len > file_size { break; }
            let name_bytes = unsafe { std::slice::from_raw_parts(data.add(pos), name_len) };
            let name = std::str::from_utf8(name_bytes).ok()?.to_string();
            pos += name_len;

            // V2: explicit dtype byte before n_dims
            let dtype = if is_v2 {
                if pos + 1 > file_size { break; }
                let d = unsafe { *data.add(pos) };
                pos += 1;
                match d {
                    0 => QuantDtype::Int8,
                    1 => QuantDtype::F32,
                    2 => QuantDtype::BF16,
                    _ => QuantDtype::Int8, // fallback
                }
            } else {
                // V1: inferred later from scale_offset
                QuantDtype::Int8 // placeholder, corrected below
            };

            if pos + 1 > file_size { break; }
            let n_dims = unsafe { *data.add(pos) } as usize;
            pos += 1;

            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                if pos + 4 > file_size { break; }
                shape.push(unsafe { read_u32(data, pos) } as usize);
                pos += 4;
            }

            if pos + 16 > file_size { break; }
            let data_offset = unsafe { read_u64(data, pos) } as usize;
            pos += 8;
            let scale_offset = unsafe { read_u64(data, pos) } as usize;
            pos += 8;

            // V1: infer dtype from scale_offset (0 = F32, non-zero = INT8)
            let dtype = if !is_v2 {
                if scale_offset == 0 { QuantDtype::F32 } else { QuantDtype::Int8 }
            } else {
                dtype
            };

            let idx = tensors.len();
            tensor_map.insert(name.clone(), idx);
            tensors.push(QuantTensorMeta {
                name,
                dtype,
                shape,
                data_offset,
                scale_offset,
            });
        }

        let data_section_offset = pos;

        Some(QuantFile {
            _fd: raw_fd,
            data,
            file_size,
            data_section_offset,
            tensors,
            tensor_map,
        })
    }

    pub fn find(&self, name: &str) -> Option<&QuantTensorMeta> {
        self.tensor_map.get(name).map(|&i| &self.tensors[i])
    }

    /// Get a `QuantWeight` view for a named tensor.
    pub fn get_quant_weight(&self, name: &str) -> Option<QuantWeight> {
        let meta = self.find(name)?;
        let data_ptr = unsafe {
            self.data.add(self.data_section_offset + meta.data_offset) as *const i8
        };
        let scale_ptr = unsafe {
            self.data.add(self.data_section_offset + meta.scale_offset) as *const f32
        };
        Some(QuantWeight {
            data: data_ptr,
            scales: scale_ptr,
            out_dim: meta.out_dim(),
            in_dim: meta.in_dim(),
        })
    }

    /// Get INT8 data as f32 Vec (dequantizes: f32[i,j] = scale[i] * int8[i,j]).
    /// For F32 tensors, returns raw f32 data.
    pub fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        let meta = self.find(name)?;
        match meta.dtype {
            QuantDtype::F32 => {
                let n = meta.numel();
                let ptr = unsafe {
                    self.data.add(self.data_section_offset + meta.data_offset) as *const f32
                };
                let mut out = vec![0.0f32; n];
                unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n); }
                Some(out)
            }
            QuantDtype::BF16 => {
                // Convert BF16 to F32
                self.get_bf16_as_f32(name)
            }
            QuantDtype::Int8 => {
                // Dequantize INT8
                let qw = self.get_quant_weight(name)?;
                let n = qw.out_dim * qw.in_dim;
                let mut out = vec![0.0f32; n];
                let data = unsafe { std::slice::from_raw_parts(qw.data, n) };
                let scales = unsafe { std::slice::from_raw_parts(qw.scales, qw.out_dim) };
                for o in 0..qw.out_dim {
                    let s = scales[o];
                    for k in 0..qw.in_dim {
                        out[o * qw.in_dim + k] = data[o * qw.in_dim + k] as f32 * s;
                    }
                }
                Some(out)
            }
        }
    }

    /// Get a direct pointer to BF16 data (mmap'd, zero-copy).
    /// Returns None if tensor is not BF16.
    pub fn get_bf16_direct(&self, name: &str) -> Option<*const u16> {
        let meta = self.find(name)?;
        if meta.dtype != QuantDtype::BF16 { return None; }
        Some(unsafe {
            self.data.add(self.data_section_offset + meta.data_offset) as *const u16
        })
    }

    /// Get BF16 data converted to f32 Vec.
    pub fn get_bf16_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        let meta = self.find(name)?;
        if meta.dtype != QuantDtype::BF16 { return None; }
        let n = meta.numel();
        let ptr = unsafe {
            self.data.add(self.data_section_offset + meta.data_offset) as *const u16
        };
        let src = unsafe { std::slice::from_raw_parts(ptr, n) };
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = bf16_to_f32(src[i]);
        }
        Some(out)
    }

    /// Check if a tensor exists in the file.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_map.contains_key(name)
    }

    /// Check if this is a V2 self-contained file (contains encoder weights).
    pub fn is_self_contained(&self) -> bool {
        self.has_tensor("thinker.audio_tower.layers.0.self_attn.q_proj.weight")
    }
}

impl Drop for QuantFile {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                libc::munmap(self.data as *mut _, self.file_size);
            }
        }
    }
}

// ========================================================================
// Writer (used by quantization tool)
// ========================================================================

/// Entry for writing a quantized tensor.
pub struct QuantWriteEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub int8_data: Vec<i8>,
    pub scales: Vec<f32>,
}

/// Entry for writing a raw f32 tensor (norms, biases).
pub struct F32WriteEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Entry for writing a raw BF16 tensor (encoder weights, embeddings).
pub struct BF16WriteEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<u16>, // raw BF16 values
}

/// Write a `.qint8` file from a collection of quantized and f32 tensors.
pub fn write_qint8_file(
    path: &str,
    quant_tensors: &[QuantWriteEntry],
    f32_tensors: &[F32WriteEntry],
) -> std::io::Result<()> {
    use std::io::Write;

    let n_tensors = quant_tensors.len() + f32_tensors.len();

    // First pass: compute data section layout
    let mut data_section = Vec::new();
    struct TensorLayout {
        name: String,
        shape: Vec<usize>,
        n_dims: u8,
        data_offset: usize,
        scale_offset: usize,
    }
    let mut layouts = Vec::with_capacity(n_tensors);

    // Pack quantized tensors: i8 data followed by f32 scales
    for entry in quant_tensors {
        let data_offset = data_section.len();
        // Write i8 data as bytes
        let i8_bytes = unsafe {
            std::slice::from_raw_parts(entry.int8_data.as_ptr() as *const u8, entry.int8_data.len())
        };
        data_section.extend_from_slice(i8_bytes);
        // Align to 4 bytes for scales
        while data_section.len() % 4 != 0 {
            data_section.push(0u8);
        }
        let scale_offset = data_section.len();
        let scale_bytes = unsafe {
            std::slice::from_raw_parts(entry.scales.as_ptr() as *const u8, entry.scales.len() * 4)
        };
        data_section.extend_from_slice(scale_bytes);

        layouts.push(TensorLayout {
            name: entry.name.clone(),
            shape: entry.shape.clone(),
            n_dims: entry.shape.len() as u8,
            data_offset,
            scale_offset,
        });
    }

    // Pack f32 tensors: stored as raw f32 at data_offset, scale_offset set to 0
    for entry in f32_tensors {
        // Align to 4 bytes
        while data_section.len() % 4 != 0 {
            data_section.push(0u8);
        }
        let data_offset = data_section.len();
        let f32_bytes = unsafe {
            std::slice::from_raw_parts(entry.data.as_ptr() as *const u8, entry.data.len() * 4)
        };
        data_section.extend_from_slice(f32_bytes);

        layouts.push(TensorLayout {
            name: entry.name.clone(),
            shape: entry.shape.clone(),
            n_dims: entry.shape.len() as u8,
            data_offset,
            scale_offset: 0,
        });
    }

    // Write file
    let mut file = std::fs::File::create(path)?;

    // Header
    file.write_all(&MAGIC_V1)?;
    file.write_all(&(n_tensors as u32).to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?; // reserved

    // Tensor table
    for layout in &layouts {
        let name_bytes = layout.name.as_bytes();
        file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
        file.write_all(name_bytes)?;
        file.write_all(&[layout.n_dims])?;
        for &dim in &layout.shape {
            file.write_all(&(dim as u32).to_le_bytes())?;
        }
        file.write_all(&(layout.data_offset as u64).to_le_bytes())?;
        file.write_all(&(layout.scale_offset as u64).to_le_bytes())?;
    }

    // Data section
    file.write_all(&data_section)?;

    Ok(())
}

/// Write a V2 `.qint8` file with INT8, F32 and BF16 tensors.
pub fn write_qint8_v2_file(
    path: &str,
    quant_tensors: &[QuantWriteEntry],
    f32_tensors: &[F32WriteEntry],
    bf16_tensors: &[BF16WriteEntry],
) -> std::io::Result<()> {
    use std::io::Write;

    let n_tensors = quant_tensors.len() + f32_tensors.len() + bf16_tensors.len();

    let mut data_section = Vec::new();
    struct TensorLayoutV2 {
        name: String,
        dtype: u8,
        shape: Vec<usize>,
        n_dims: u8,
        data_offset: usize,
        scale_offset: usize,
    }
    let mut layouts = Vec::with_capacity(n_tensors);

    // Pack INT8 quantized tensors
    for entry in quant_tensors {
        let data_offset = data_section.len();
        let i8_bytes = unsafe {
            std::slice::from_raw_parts(entry.int8_data.as_ptr() as *const u8, entry.int8_data.len())
        };
        data_section.extend_from_slice(i8_bytes);
        while data_section.len() % 4 != 0 { data_section.push(0u8); }
        let scale_offset = data_section.len();
        let scale_bytes = unsafe {
            std::slice::from_raw_parts(entry.scales.as_ptr() as *const u8, entry.scales.len() * 4)
        };
        data_section.extend_from_slice(scale_bytes);

        layouts.push(TensorLayoutV2 {
            name: entry.name.clone(),
            dtype: QuantDtype::Int8 as u8,
            shape: entry.shape.clone(),
            n_dims: entry.shape.len() as u8,
            data_offset,
            scale_offset,
        });
    }

    // Pack F32 tensors
    for entry in f32_tensors {
        while data_section.len() % 4 != 0 { data_section.push(0u8); }
        let data_offset = data_section.len();
        let f32_bytes = unsafe {
            std::slice::from_raw_parts(entry.data.as_ptr() as *const u8, entry.data.len() * 4)
        };
        data_section.extend_from_slice(f32_bytes);

        layouts.push(TensorLayoutV2 {
            name: entry.name.clone(),
            dtype: QuantDtype::F32 as u8,
            shape: entry.shape.clone(),
            n_dims: entry.shape.len() as u8,
            data_offset,
            scale_offset: 0,
        });
    }

    // Pack BF16 tensors
    for entry in bf16_tensors {
        while data_section.len() % 2 != 0 { data_section.push(0u8); }
        let data_offset = data_section.len();
        let bf16_bytes = unsafe {
            std::slice::from_raw_parts(entry.data.as_ptr() as *const u8, entry.data.len() * 2)
        };
        data_section.extend_from_slice(bf16_bytes);

        layouts.push(TensorLayoutV2 {
            name: entry.name.clone(),
            dtype: QuantDtype::BF16 as u8,
            shape: entry.shape.clone(),
            n_dims: entry.shape.len() as u8,
            data_offset,
            scale_offset: 0,
        });
    }

    // Write file
    let mut file = std::fs::File::create(path)?;

    // Header
    file.write_all(&MAGIC_V2)?;
    file.write_all(&(n_tensors as u32).to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?; // reserved

    // Tensor table (V2: includes dtype byte)
    for layout in &layouts {
        let name_bytes = layout.name.as_bytes();
        file.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
        file.write_all(name_bytes)?;
        file.write_all(&[layout.dtype])?; // dtype byte (V2)
        file.write_all(&[layout.n_dims])?;
        for &dim in &layout.shape {
            file.write_all(&(dim as u32).to_le_bytes())?;
        }
        file.write_all(&(layout.data_offset as u64).to_le_bytes())?;
        file.write_all(&(layout.scale_offset as u64).to_le_bytes())?;
    }

    // Data section
    file.write_all(&data_section)?;

    Ok(())
}

// ========================================================================
// Quantization helpers
// ========================================================================

/// Quantize a BF16 weight matrix to per-channel symmetric INT8.
/// `bf16_ptr` points to `[out_dim, in_dim]` BF16 values.
/// Returns (int8_data, scales).
pub fn quantize_bf16_to_int8(
    bf16_ptr: *const u16,
    out_dim: usize,
    in_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut q_data = vec![0i8; out_dim * in_dim];
    let mut scales = vec![0.0f32; out_dim];

    for o in 0..out_dim {
        // Find per-channel absmax
        let mut absmax = 0.0f32;
        for k in 0..in_dim {
            let val = bf16_to_f32(unsafe { *bf16_ptr.add(o * in_dim + k) });
            let abs = val.abs();
            if abs > absmax {
                absmax = abs;
            }
        }

        let scale = absmax / 127.0;
        scales[o] = scale;
        let inv_scale = if absmax > 0.0 { 127.0 / absmax } else { 0.0 };

        for k in 0..in_dim {
            let val = bf16_to_f32(unsafe { *bf16_ptr.add(o * in_dim + k) });
            let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
            q_data[o * in_dim + k] = q;
        }
    }

    (q_data, scales)
}

/// Quantize an f32 weight matrix to per-channel symmetric INT8.
pub fn quantize_f32_to_int8(
    f32_data: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut q_data = vec![0i8; out_dim * in_dim];
    let mut scales = vec![0.0f32; out_dim];

    for o in 0..out_dim {
        let row = &f32_data[o * in_dim..(o + 1) * in_dim];
        let mut absmax = 0.0f32;
        for &val in row {
            let abs = val.abs();
            if abs > absmax {
                absmax = abs;
            }
        }

        let scale = absmax / 127.0;
        scales[o] = scale;
        let inv_scale = if absmax > 0.0 { 127.0 / absmax } else { 0.0 };

        for k in 0..in_dim {
            let q = (row[k] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            q_data[o * in_dim + k] = q;
        }
    }

    (q_data, scales)
}

// ========================================================================
// Internal helpers
// ========================================================================

#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

#[inline]
unsafe fn read_u16(base: *const u8, offset: usize) -> u16 {
    let mut buf = [0u8; 2];
    std::ptr::copy_nonoverlapping(base.add(offset), buf.as_mut_ptr(), 2);
    u16::from_le_bytes(buf)
}

#[inline]
unsafe fn read_u32(base: *const u8, offset: usize) -> u32 {
    let mut buf = [0u8; 4];
    std::ptr::copy_nonoverlapping(base.add(offset), buf.as_mut_ptr(), 4);
    u32::from_le_bytes(buf)
}

#[inline]
unsafe fn read_u64(base: *const u8, offset: usize) -> u64 {
    let mut buf = [0u8; 8];
    std::ptr::copy_nonoverlapping(base.add(offset), buf.as_mut_ptr(), 8);
    u64::from_le_bytes(buf)
}
