use std::{
    alloc::{Layout, LayoutError, alloc},
    any::Any,
    ffi::c_void,
    ptr::NonNull,
};

use bytemuck::{AnyBitPattern, BoxBytes, try_from_box_bytes};
use ndarray::{
    Array, ArrayView, ArrayViewMut, Dim, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn,
    ShapeBuilder, ShapeError,
};
use smallvec::SmallVec;
use thiserror::Error;

use crate::{
    bits::BITS_PER_BYTE,
    device::{DeviceBackend, TensorAllocateInfo},
    tensors::{
        dimensions::TensorDimension,
        dtype::TensorDataTypeTrait,
        owned::Tensor,
        views::{TENSOR_DIM_SMALL_VEC_CAP, TensorMut, TensorRef, TensorRefTrait},
    },
};

#[derive(Default, Debug)]
pub struct BackendSpecificInfo;

#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("Invalid type cast for tensor")]
    InvalidTypeCast,
    #[error("Ndarray does not support negative strides")]
    NegativeStrideError,
    #[error("Stride must be a multiple of elements")]
    NonElementMultipleStrideError,
    #[error("Shape error: {0}")]
    ShapeError(ShapeError),
    #[error("Invalid tensor layout: {0}")]
    InvalidLayoutError(LayoutError),
    #[error("Allocation error")]
    AllocationError,
}

pub enum TensorManagerContext {
    // TODO: make this nicer somehow (we can't use Array because generics)
    Ndarray(Box<dyn Any>),
    ByteContainer(BoxBytes),
}

#[derive(Debug, Default)]
pub struct Device;

fn alloc_safe(layout: Layout) -> Result<BoxBytes, DeviceError> {
    unsafe {
        match NonNull::new(alloc(layout)) {
            Some(ptr) => Ok(BoxBytes::from_raw_parts(ptr, layout)),
            None => Err(DeviceError::AllocationError),
        }
    }
}

impl super::Device for Device {
    fn backend(&self) -> super::DeviceBackend {
        super::DeviceBackend::Cpu
    }

    fn upcast(&self) -> &dyn super::Device {
        self
    }

    fn alloc_tensor_memory(
        &self,
        size_in_bytes: usize,
        alloc_info: &TensorAllocateInfo,
    ) -> Result<(*mut c_void, Box<dyn Any>), super::DeviceError> {
        let mut memory = alloc_safe(
            Layout::from_size_align(
                size_in_bytes,
                alloc_info.alignment.get().div_ceil(BITS_PER_BYTE),
            )
            .map_err(DeviceError::InvalidLayoutError)?,
        )?;
        Ok((
            memory.as_mut_ptr() as _,
            Box::new(TensorManagerContext::ByteContainer(memory)),
        ))
    }

    unsafe fn read_back(
        &self,
        ptr: *const c_void,
        offset: usize,
        size: usize,
    ) -> Result<Box<[u8]>, super::DeviceError> {
        unsafe {
            Ok(
                std::slice::from_raw_parts((ptr as *const u8).add(offset), size)
                    .to_vec()
                    .into_boxed_slice(),
            )
        }
    }

    unsafe fn write_to(
        &self,
        ptr: *mut c_void,
        offset: usize,
        data: &[u8],
    ) -> Result<(), super::DeviceError> {
        unsafe {
            std::slice::from_raw_parts_mut((ptr as *mut u8).add(offset), data.len())
                .copy_from_slice(data);
            Ok(())
        }
    }
}

fn slice_to_dim<const N: usize>(slice: &[usize]) -> Result<Dim<[usize; N]>, DeviceError>
where
    [usize; N]: IntoDim<Dim<[usize; N]>>,
{
    let array: [usize; N] = slice.try_into().map_err(|_| DeviceError::InvalidTypeCast)?;
    Ok(array.into_dim())
}

trait IntoDim<Dim> {
    fn into_dim(self) -> Dim;
}

impl<const N: usize> IntoDim<IxDyn> for [usize; N] {
    fn into_dim(self) -> IxDyn {
        IxDyn(&self)
    }
}

impl IntoDim<Ix0> for [usize; 0] {
    fn into_dim(self) -> Ix0 {
        Ix0()
    }
}

impl IntoDim<Ix1> for [usize; 1] {
    fn into_dim(self) -> Ix1 {
        Ix1(self[0])
    }
}

impl IntoDim<Ix2> for [usize; 2] {
    fn into_dim(self) -> Ix2 {
        Ix2(self[0], self[1])
    }
}

impl IntoDim<Ix3> for [usize; 3] {
    fn into_dim(self) -> Ix3 {
        Ix3(self[0], self[1], self[2])
    }
}

impl IntoDim<Ix4> for [usize; 4] {
    fn into_dim(self) -> Ix4 {
        Ix4(self[0], self[1], self[2], self[3])
    }
}

impl IntoDim<Ix5> for [usize; 5] {
    fn into_dim(self) -> Ix5 {
        Ix5(self[0], self[1], self[2], self[3], self[4])
    }
}

impl IntoDim<Ix6> for [usize; 6] {
    fn into_dim(self) -> Ix6 {
        Ix6(self[0], self[1], self[2], self[3], self[4], self[5])
    }
}

pub trait TryIntoDim<Dim> {
    fn try_into_dim(&self) -> Result<Dim, DeviceError>;
}

impl TryIntoDim<IxDyn> for [usize] {
    fn try_into_dim(&self) -> Result<IxDyn, DeviceError> {
        Ok(IxDyn(self))
    }
}

impl TryIntoDim<Ix0> for [usize] {
    fn try_into_dim(&self) -> Result<Ix0, DeviceError> {
        slice_to_dim::<0>(self)
    }
}

impl TryIntoDim<Ix1> for [usize] {
    fn try_into_dim(&self) -> Result<Ix1, DeviceError> {
        slice_to_dim::<1>(self)
    }
}

impl TryIntoDim<Ix2> for [usize] {
    fn try_into_dim(&self) -> Result<Ix2, DeviceError> {
        slice_to_dim::<2>(self)
    }
}

impl TryIntoDim<Ix3> for [usize] {
    fn try_into_dim(&self) -> Result<Ix3, DeviceError> {
        slice_to_dim::<3>(self)
    }
}

impl TryIntoDim<Ix4> for [usize] {
    fn try_into_dim(&self) -> Result<Ix4, DeviceError> {
        slice_to_dim::<4>(self)
    }
}

impl TryIntoDim<Ix5> for [usize] {
    fn try_into_dim(&self) -> Result<Ix5, DeviceError> {
        slice_to_dim::<5>(self)
    }
}

impl TryIntoDim<Ix6> for [usize] {
    fn try_into_dim(&self) -> Result<Ix6, DeviceError> {
        slice_to_dim::<6>(self)
    }
}

impl Device {
    pub fn new() -> Self {
        Self
    }

    pub fn tensor_from_ndarray<A, D>(&self, mut arr: Array<A, D>) -> Tensor
    where
        A: TensorDataTypeTrait + 'static,
        D: Dimension + 'static,
    {
        Tensor {
            view: self.tensor_from_ndarray_mut(arr.view_mut()),
            manager_ctx: Box::new(TensorManagerContext::Ndarray(Box::new(arr))),
        }
    }

    pub fn tensor_from_ndarray_ref<A, D>(&self, arr: ArrayView<A, D>) -> TensorRef
    where
        A: TensorDataTypeTrait + 'static,
        D: Dimension + 'static,
    {
        TensorRef {
            device: self,
            dimensions: arr
                .shape()
                .iter()
                .zip(ArrayView::<A, D>::strides(&arr).iter())
                .map(|(dim, stride)| {
                    TensorDimension::new(*dim, *stride * (A::data_type().size_in_bits() as isize))
                })
                .collect(),
            memory: arr.as_ptr() as *mut c_void,
            memory_offset: 0,
            dtype: A::data_type(),
        }
    }

    pub fn tensor_from_ndarray_mut<A, D>(&self, mut arr: ArrayViewMut<A, D>) -> TensorMut
    where
        A: TensorDataTypeTrait + 'static,
        D: Dimension + 'static,
    {
        TensorMut {
            device: self,
            dimensions: arr
                .shape()
                .iter()
                .zip(ArrayViewMut::<A, D>::strides(&arr).iter())
                .map(|(dim, stride)| {
                    TensorDimension::new(*dim, *stride * (A::data_type().size_in_bits() as isize))
                })
                .collect(),
            memory: arr.as_mut_ptr() as *mut c_void,
            memory_offset: 0,
            dtype: A::data_type(),
        }
    }

    pub fn tensor_to_ndarray<A, Dim>(
        &self,
        tensor: Tensor,
    ) -> Result<Array<A, Dim>, super::DeviceError>
    where
        A: TensorDataTypeTrait + AnyBitPattern,
        [usize]: TryIntoDim<Dim>,
        Dim: Dimension + 'static,
    {
        if tensor.view.device.backend() != DeviceBackend::Cpu {
            return Err(super::DeviceError::NonOwningTensorError);
        }

        if tensor.view.dtype != A::data_type() {
            return Err(DeviceError::InvalidTypeCast.into());
        }

        let Tensor { view, manager_ctx } = tensor;
        match manager_ctx.downcast::<TensorManagerContext>() {
            Ok(ctx) => match *ctx {
                TensorManagerContext::Ndarray(array) => array
                    .downcast::<Array<A, _>>()
                    .map(|a| *a)
                    .map_err(|_| DeviceError::InvalidTypeCast.into()),
                TensorManagerContext::ByteContainer(items) => {
                    let items = try_from_box_bytes::<[A]>(items)
                        .map_err(|(err, _)| err)
                        .unwrap()
                        // .map_err(|_| DeviceError::InvalidTypeCast)?
                        .into_vec();
                    let strides = TensorRefTrait::strides(&view).into_iter().map(|stride| -> Result<usize, DeviceError> {
                        let stride = usize::try_from(stride).map_err(|_| DeviceError::NegativeStrideError)?;
                        if stride % A::data_type().size_in_bits() == 0 {
                            Ok(stride / A::data_type().size_in_bits())
                        } else {
                            Err(DeviceError::NonElementMultipleStrideError)
                        }
                    }).collect::<Result<SmallVec<[usize; TENSOR_DIM_SMALL_VEC_CAP]>, DeviceError>>()?;
                    let shape: Dim = view.shape().as_ref().try_into_dim()?;
                    let strides: Dim = strides.as_ref().try_into_dim()?;
                    let shape = shape.strides(strides);
                    Ok(Array::from_shape_vec(shape, items).map_err(DeviceError::ShapeError)?)
                }
            },
            Err(_) => panic!("invalid CPU tensor manager context"),
        }
    }
}
