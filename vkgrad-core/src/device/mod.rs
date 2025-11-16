use std::{any::Any, borrow::Cow, ffi::c_void, num::NonZeroUsize};

use smallvec::SmallVec;
use thiserror::Error;

use crate::{
    bits::{BITS_PER_BYTE, memcpy_bit},
    tensors::{
        dimensions::TensorDimension,
        dtype::TensorDataType,
        owned::Tensor,
        views::{TENSOR_DIM_SMALL_VEC_CAP, TensorMut, TensorMutTrait, TensorRef, TensorRefTrait},
    },
};

#[derive(Clone, Debug)]
pub struct TensorAllocateInfo {
    pub dtype: TensorDataType,
    pub dimensions: SmallVec<[TensorDimension; TENSOR_DIM_SMALL_VEC_CAP]>,
    pub alignment: NonZeroUsize, // in bits
}

impl TensorAllocateInfo {
    pub fn new(
        dtype: TensorDataType,
        dimensions: SmallVec<[TensorDimension; TENSOR_DIM_SMALL_VEC_CAP]>,
        alignment: NonZeroUsize,
    ) -> Self {
        Self {
            dtype,
            dimensions,
            alignment,
        }
    }

    pub fn new_row_major(dtype: TensorDataType, shape: &'_ [usize]) -> Self {
        let mut dimensions = SmallVec::<[TensorDimension; TENSOR_DIM_SMALL_VEC_CAP]>::new();
        let alignment = dtype.alignment();

        let mut last_stride = dtype.size_in_bits();
        for dim in shape.iter().rev() {
            dimensions.push(TensorDimension::new(*dim, last_stride as _));
            last_stride = (last_stride * dim)
                .checked_next_multiple_of(alignment.get())
                .unwrap();
        }
        dimensions.reverse();

        Self::new(dtype, dimensions, dtype.alignment())
    }
}

fn copy_tensor(
    dst: &mut [u8],
    src: &[u8],
    dst_bit_off: usize,
    src_bit_off: usize,
    dtype: TensorDataType,
    src_dims: &[TensorDimension],
    dst_dims: &[TensorDimension],
) {
    assert!(src_dims.len() == dst_dims.len());
    if TensorDimension::is_packed_homogenous_linear(dtype, src_dims.iter())
        && TensorDimension::is_packed_homogenous_linear(dtype, dst_dims.iter())
    {
        let size = src_dims.iter().map(|d| d.shape).product::<usize>() * dtype.size_in_bits();
        memcpy_bit(src, dst, src_bit_off, dst_bit_off, size);
        return;
    }
    match src_dims.split_first().zip(dst_dims.split_first()) {
        None => memcpy_bit(src, dst, src_bit_off, dst_bit_off, dtype.size_in_bits()),
        Some(((src_first_dim, src_rem_dims), (dst_first_dim, dst_rem_dims))) => {
            assert!(src_first_dim.shape == dst_first_dim.shape);
            // "row" in a figurative sense
            for row in 0..src_first_dim.shape {
                let row = row as isize;
                let dst_bit_off = dst_bit_off as isize + dst_first_dim.stride * row;
                let src_bit_off = src_bit_off as isize + src_first_dim.stride * row;
                assert!(src_bit_off >= 0 && dst_bit_off >= 0);
                copy_tensor(
                    dst,
                    src,
                    dst_bit_off as _,
                    src_bit_off as _,
                    dtype,
                    src_rem_dims,
                    dst_rem_dims,
                );
            }
        }
    }
}

fn assert_dim_eq(lhs: usize, rhs: usize) -> Result<(), DeviceError> {
    if lhs != rhs {
        Err(DeviceError::MismatchDimensions(lhs, rhs))
    } else {
        Ok(())
    }
}

pub trait Device {
    fn backend(&self) -> DeviceBackend;
    fn upcast(&self) -> &dyn Device;

    // internal: this is to make impl alloc_tensor a little bit easier
    fn alloc_tensor_memory(
        &self,
        _size_in_bytes: usize,
        _alloc_info: &TensorAllocateInfo,
    ) -> Result<(*mut c_void, Box<dyn Any>), DeviceError> {
        Err(DeviceError::UnsupportedFeature)
    }

    fn alloc_tensor(&self, alloc_info: TensorAllocateInfo) -> Result<Tensor<'_>, DeviceError> {
        let (min_offset, _, size_in_bits) =
            TensorDimension::calc_offsets_and_size(alloc_info.dtype, alloc_info.dimensions.iter());
        let size_in_bytes = size_in_bits.div_ceil(BITS_PER_BYTE);
        assert!(min_offset <= 0);
        let backend = self.backend();
        log::trace!(
            "Allocating {size_in_bytes} bytes ({size_in_bits} bits) for {alloc_info:?} on device {backend:?}"
        );
        let (memory, manager_ctx) = self.alloc_tensor_memory(size_in_bytes, &alloc_info)?;
        Ok(Tensor {
            view: TensorMut {
                device: self.upcast(),
                dimensions: alloc_info.dimensions.iter().copied().collect(),
                memory,
                memory_offset: -min_offset as _,
                dtype: alloc_info.dtype,
            },
            manager_ctx,
        })
    }

    fn copy(&self, src: &TensorRef, dst: &mut TensorMut) -> Result<(), DeviceError> {
        assert!(src.shape() == dst.shape());
        assert!(src.dtype == dst.dtype);

        let (src_data, src_bit_off) = src.read_back()?;
        // TODO: this read-back is not needed if aligned
        let (mut dst_data, dst_bit_off) = dst.read_back()?;

        copy_tensor(
            &mut dst_data,
            &src_data,
            dst_bit_off,
            src_bit_off,
            src.dtype,
            &src.dimensions,
            &dst.dimensions,
        );

        dst.write_to(&dst_data)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        _lhs: &TensorRef,
        _rhs: &TensorRef,
        _ans: &mut TensorMut,
        _alpha: f32,
        _beta: f32,
        _transpose_lhs: bool,
        _transpose_rhs: bool,
    ) -> Result<(), DeviceError> {
        Err(DeviceError::UnsupportedFeature)
    }

    /// # Safety
    /// ptr must be a valid tensor memory of this device, and write operations must not be out of bounds
    unsafe fn write_to(
        &self,
        _ptr: *mut c_void,
        _offset: usize, // in bytes
        _data: &[u8],
    ) -> Result<(), DeviceError> {
        Err(DeviceError::UnsupportedFeature)
    }

    /// # Safety
    /// ptr must be a valid tensor memory of this device, and read operations must not be out of bounds
    unsafe fn read_back(
        &self,
        _ptr: *const c_void,
        _offset: usize, // in bytes
        _size: usize,
    ) -> Result<Box<[u8]>, DeviceError> {
        Err(DeviceError::UnsupportedFeature)
    }
}

#[derive(Default)]
pub struct DeviceInfo {
    pub backend: Option<DeviceBackend>,
    pub physical_device_name: Option<Cow<'static, str>>,
    pub staging_buffer_size: Option<usize>,

    /// backend-specific info
    pub vk: vk::BackendSpecificInfo,
    pub cpu: cpu::BackendSpecificInfo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceBackend {
    Vulkan,
    Cpu,
}

#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("Unsupported feature")]
    UnsupportedFeature,
    #[error("Non-owning tensor")]
    NonOwningTensorError,
    #[error("Mismatch element type: {0:?} != {1:?}")]
    MismatchDtype(TensorDataType, TensorDataType),
    #[error("Mismatch dimensions: {0} != {1}")]
    MismatchDimensions(usize, usize),
    #[error("Vulkan backend error: {0}")]
    VulkanBackendError(#[from] vk::DeviceError),
    #[error("CPU backend error: {0}")]
    CpuBackendError(#[from] cpu::DeviceError),
}

pub mod cpu;
pub mod vk;

pub fn create_device(info: DeviceInfo) -> Result<Box<dyn Device>, DeviceError> {
    Ok(match &info.backend {
        None | Some(DeviceBackend::Vulkan) => Box::new(vk::Device::new(info)?),
        Some(DeviceBackend::Cpu) => Box::new(cpu::Device::new()),
    })
}
