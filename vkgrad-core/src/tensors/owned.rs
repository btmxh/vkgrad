use std::any::Any;

use ndarray::{ArrayView, Dimension};

use crate::{
    device::{Device, DeviceError, TensorAllocateInfo, cpu},
    tensors::{
        dtype::TensorDataTypeTrait,
        views::{TensorMut, TensorMutTrait, TensorRefTrait},
    },
};

pub struct Tensor<'a> {
    pub view: TensorMut<'a>,
    pub manager_ctx: Box<dyn Any>,
}

impl<'a> TensorRefTrait for Tensor<'a> {
    type Memory = <TensorMut<'a> as TensorRefTrait>::Memory;

    fn device(&self) -> &dyn crate::device::Device {
        self.view.device()
    }

    fn dimensions(&self) -> &[super::dimensions::TensorDimension] {
        self.view.dimensions()
    }

    fn memory(&self) -> Self::Memory {
        self.view.memory()
    }

    fn memory_offset(&self) -> usize {
        self.view.memory_offset()
    }

    fn dtype(&self) -> super::dtype::TensorDataType {
        self.view.dtype()
    }
}

impl<'a> TensorMutTrait for Tensor<'a> {}

impl<'a> Tensor<'a> {
    pub fn from_ndarray<'b, A, D>(
        device: &'a dyn Device,
        view: ArrayView<'b, A, D>,
    ) -> Result<Self, DeviceError>
    where
        A: TensorDataTypeTrait + 'static,
        D: Dimension + 'static,
    {
        let view = cpu::Device.tensor_from_ndarray_ref(view);
        let mut tensor = device.alloc_tensor(TensorAllocateInfo::new_row_major(
            view.dtype(),
            &view.shape(),
        ))?;
        view.copy_to(&mut tensor)?;
        Ok(tensor)
    }
}
