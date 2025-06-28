use ndarray::{Array, Ix2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use vkgrad_core::{
    device::{DeviceInfo, TensorAllocateInfo, create_device},
    tensors::{
        owned::Tensor,
        views::{TensorMutTrait, TensorRefTrait},
    },
};

extern crate blas_src;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    #[allow(non_snake_case)]
    let M = 17usize;
    #[allow(non_snake_case)]
    let N = 21usize;
    #[allow(non_snake_case)]
    let K = 19usize;
    let dist = Uniform::new(-5.0f32, 5.0f32);
    let lhs = Array::random((M, K), dist);
    let rhs = Array::random((K, N), dist);
    let ans = lhs.dot(&rhs);

    let device = create_device(DeviceInfo::default())?;
    let device_lhs = Tensor::from_ndarray(&*device, lhs.view())?;
    let device_rhs = Tensor::from_ndarray(&*device, rhs.view())?;

    let lhs_readback = device_lhs.to_ndarray::<f32, Ix2>()?;
    assert!(lhs_readback.shape() == lhs.shape());
    assert!((lhs_readback - lhs).abs().iter().all(|elem| *elem < 1e-8));

    let mut device_ans = device.alloc_tensor(TensorAllocateInfo::new_row_major(
        device_lhs.dtype(),
        &[M, N],
    ))?;
    device.gemm(
        &device_lhs.as_ref(),
        &device_rhs.as_ref(),
        &mut device_ans.as_mut(),
    )?;

    let ans_readback = device_ans.to_ndarray::<f32, Ix2>()?;
    assert!(ans_readback.shape() == ans.shape());
    assert!((ans_readback - ans).abs().iter().all(|elem| *elem < 1e-8));

    Ok(())
}
