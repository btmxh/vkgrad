use std::ffi::c_void;

use bytemuck::AnyBitPattern;
use ndarray::{Array, Dimension};
use smallvec::SmallVec;

use crate::{
    bits::BITS_PER_BYTE,
    device::{
        Device, DeviceError, TensorAllocateInfo,
        cpu::{self, TryIntoDim},
    },
    tensors::{
        dimensions::TensorDimension,
        dtype::{TensorDataType, TensorDataTypeTrait},
    },
};

// we mainly work with tensor with ndim <= 4, so here is a reasonable choice
pub const TENSOR_DIM_SMALL_VEC_CAP: usize = 4;

// generic tensor types based on dlpack
#[derive(Clone)]
pub struct TensorView<'a, M> {
    pub(crate) device: &'a dyn Device,
    pub(crate) dimensions: SmallVec<[TensorDimension; TENSOR_DIM_SMALL_VEC_CAP]>,
    pub(crate) memory: M,
    pub(crate) memory_offset: usize, // in bits, not bytes
    pub(crate) dtype: TensorDataType,
}

pub type TensorRef<'a> = TensorView<'a, *const c_void>;
pub type TensorMut<'a> = TensorView<'a, *mut c_void>;

pub trait TensorRefTrait {
    type Memory: Ptr;
    fn device(&self) -> &dyn Device;
    fn dimensions(&self) -> &[TensorDimension];
    fn memory(&self) -> Self::Memory;
    fn memory_offset(&self) -> usize;
    fn dtype(&self) -> TensorDataType;

    fn shape(&self) -> SmallVec<[usize; TENSOR_DIM_SMALL_VEC_CAP]> {
        self.dimensions().iter().map(|d| d.shape).collect()
    }

    fn strides(&self) -> SmallVec<[isize; TENSOR_DIM_SMALL_VEC_CAP]> {
        self.dimensions().iter().map(|d| d.stride).collect()
    }

    fn elem_strides(&self) -> Option<SmallVec<[isize; TENSOR_DIM_SMALL_VEC_CAP]>> {
        self.strides()
            .iter()
            .map(|s| match (*s, (self.dtype().size_in_bits() as isize)) {
                (a, b) if a % b == 0 => Some(a / b),
                _ => None,
            })
            .collect::<Option<SmallVec<[isize; TENSOR_DIM_SMALL_VEC_CAP]>>>()
    }

    fn calc_offsets_and_size(&self) -> (isize, isize, usize) {
        TensorDimension::calc_offsets_and_size(self.dtype(), self.dimensions().iter())
    }

    fn as_ref(&self) -> TensorRef<'_> {
        TensorRef {
            device: self.device(),
            dimensions: self.dimensions().iter().copied().collect(),
            memory: self.memory().to_ptr(),
            memory_offset: self.memory_offset(),
            dtype: self.dtype(),
        }
    }

    fn copy_to<T, M>(&self, dst: &mut T) -> Result<(), DeviceError>
    where
        T: TensorMutTrait<Memory = M> + ?Sized,
        M: MutPtr,
    {
        self.device().copy(&self.as_ref(), &mut dst.as_mut())
    }

    // returns:
    // - byte offset
    // - byte size
    // - bit offset (wrt the byte offset)
    fn byte_offset_and_sizes(&self) -> (usize, usize, usize) {
        let (start_offset_bits, _, size_bits) = self.calc_offsets_and_size();
        let size_bits = size_bits as isize;
        let start_offset_bits = self.memory_offset() as isize + start_offset_bits;
        let end_offset_bits = start_offset_bits + size_bits;
        assert!(start_offset_bits >= 0 && end_offset_bits >= 0);

        let start_offset_bits = start_offset_bits as usize;
        let end_offset_bits = end_offset_bits as usize;

        let offset_byte = start_offset_bits / BITS_PER_BYTE;
        let size_bytes = end_offset_bits.div_ceil(BITS_PER_BYTE) - offset_byte;
        let offset_bit = start_offset_bits % BITS_PER_BYTE;
        (offset_byte, size_bytes, offset_bit)
    }

    fn read_back(&self) -> Result<(Box<[u8]>, usize), DeviceError>
    where
        Self::Memory: Ptr,
    {
        let (byte_offset, byte_size, bit_offset) = self.byte_offset_and_sizes();
        // Safety: this is safe since the tensor should own its memory
        let data = unsafe {
            self.device()
                .read_back(self.memory().to_ptr(), byte_offset, byte_size)
        }?;

        Ok((data, bit_offset))
    }

    fn to_ndarray<A, D>(&self) -> Result<Array<A, D>, DeviceError>
    where
        A: AnyBitPattern + TensorDataTypeTrait + 'static,
        D: Dimension + 'static,
        [usize]: TryIntoDim<D>,
    {
        let mut cpu_tensor = cpu::Device.alloc_tensor(TensorAllocateInfo::new_row_major(
            self.dtype(),
            &self.shape(),
        ))?;
        self.copy_to(&mut cpu_tensor)?;
        cpu::Device.tensor_to_ndarray(cpu_tensor)
    }
}

pub trait TensorMutTrait: TensorRefTrait
where
    <Self as TensorRefTrait>::Memory: MutPtr,
{
    fn copy_from<T, M>(&mut self, src: &T) -> Result<(), DeviceError>
    where
        T: TensorRefTrait<Memory = M> + ?Sized,
        M: Ptr,
    {
        src.copy_to(self)
    }

    // NOTE: data must be properly aligned
    fn write_to(&self, data: &[u8]) -> Result<(), DeviceError> {
        let (byte_offset, byte_size, _) = self.byte_offset_and_sizes();
        assert!(data.len() <= byte_size);
        // Safety: this is safe since the tensor should own its memory
        unsafe {
            self.device()
                .write_to(self.memory().to_mut_ptr(), byte_offset, data)
        }?;

        Ok(())
    }

    fn as_mut(&mut self) -> TensorMut<'_> {
        TensorMut {
            device: self.device(),
            dimensions: self.dimensions().iter().copied().collect(),
            memory: self.memory().to_mut_ptr(),
            memory_offset: self.memory_offset(),
            dtype: self.dtype(),
        }
    }
}

impl<'a, M: Ptr> TensorRefTrait for TensorView<'a, M> {
    type Memory = M;

    fn device(&self) -> &dyn Device {
        self.device
    }

    fn dimensions(&self) -> &[TensorDimension] {
        &self.dimensions
    }

    fn memory(&self) -> Self::Memory {
        self.memory
    }

    fn memory_offset(&self) -> usize {
        self.memory_offset
    }

    fn dtype(&self) -> TensorDataType {
        self.dtype
    }
}

impl<'a, M: MutPtr> TensorMutTrait for TensorView<'a, M> {}

pub trait Ptr: Copy {
    fn to_ptr(&self) -> *const c_void;
}

impl Ptr for *const c_void {
    fn to_ptr(&self) -> *const c_void {
        *self
    }
}

impl Ptr for *mut c_void {
    fn to_ptr(&self) -> *const c_void {
        *self
    }
}

pub trait MutPtr: Ptr + Copy {
    fn to_mut_ptr(&self) -> *mut c_void;
}

impl MutPtr for *mut c_void {
    fn to_mut_ptr(&self) -> *mut c_void {
        *self
    }
}
