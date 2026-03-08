//! Compute device abstraction for CPU vs Metal GPU dispatch.

#[cfg(feature = "metal")]
use candle_core::Device;

/// Unified device handle — either CPU or Metal GPU.
pub enum ComputeDevice {
    Cpu,
    #[cfg(feature = "metal")]
    Metal(Device),
}

impl ComputeDevice {
    /// Initialize the best available device (Metal GPU if available, else CPU).
    pub fn best() -> Self {
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(device) => {
                    if crate::kernels::verbose() >= 1 {
                        eprintln!("[metal] GPU device initialized");
                    }
                    return ComputeDevice::Metal(device);
                }
                Err(e) => {
                    if crate::kernels::verbose() >= 1 {
                        eprintln!("[metal] GPU init failed ({}), falling back to CPU", e);
                    }
                }
            }
        }
        ComputeDevice::Cpu
    }

    /// Returns true if this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        match self {
            ComputeDevice::Cpu => false,
            #[cfg(feature = "metal")]
            ComputeDevice::Metal(_) => true,
        }
    }

    /// Get the underlying candle Device reference (Metal only).
    #[cfg(feature = "metal")]
    pub fn candle_device(&self) -> Option<&Device> {
        match self {
            ComputeDevice::Metal(d) => Some(d),
            _ => None,
        }
    }
}
