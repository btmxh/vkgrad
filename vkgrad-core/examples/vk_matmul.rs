use std::f32::consts::PI;

use approx::assert_abs_diff_eq;
use ndarray::{Array, Ix2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray_stats::QuantileExt;
use statrs::function::erf::erf_inv;
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
    let M = 1024usize;
    #[allow(non_snake_case)]
    let N = 1024usize;
    #[allow(non_snake_case)]
    let K = 1024usize;

    let a = 5.0f32;
    let dist = Uniform::new(-a, a);
    let lhs = Array::random((M, K), dist);
    let rhs = Array::random((K, N), dist);
    let ans = lhs.dot(&rhs);

    let device = create_device(DeviceInfo::default())?;
    let device_lhs = Tensor::from_ndarray(&*device, lhs.view())?;
    let device_rhs = Tensor::from_ndarray(&*device, rhs.view())?;

    let lhs_readback = device_lhs.to_ndarray::<f32, Ix2>()?;
    assert_abs_diff_eq!(lhs_readback, lhs, epsilon = f32::EPSILON);

    let mut device_ans = device.alloc_tensor(TensorAllocateInfo::new_row_major(
        device_lhs.dtype(),
        &[M, N],
    ))?;
    device.gemm(
        &device_lhs.as_ref(),
        &device_rhs.as_ref(),
        &mut device_ans.as_mut(),
        1.0,
        0.0,
        false,
        false,
    )?;

    let ans_readback = device_ans.to_ndarray::<f32, Ix2>()?;

    let err = (&ans - &ans_readback).abs();
    let emp_max_error = err.max().unwrap();
    log::debug!("Emperical max error: {emp_max_error}");

    // see docs/
    let alpha = 0.01f32;
    let max_error = (8.0 / 27.0 / PI).sqrt()
        * f32::EPSILON
        * a
        * a
        * (K as f32)
        * erf_inv((1.0 - alpha as f64).powf(1.0 / ((M * N) as f64))) as f32;
    log::debug!("Max error used for testing: {max_error}");
    assert_abs_diff_eq!(ans_readback, ans, epsilon = max_error);

    Ok(())
}
