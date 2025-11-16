pub(super) mod f32 {
    pub const BLOCK_M: usize = 64;
    pub const BLOCK_N: usize = 64;
    pub const BLOCK_K: usize = 8;
    pub const THREAD_M: usize = 8;

    // make sure the two TILE_SIZE definitions are consistent
    vulkano_shaders::shader! {
        ty: "compute",
        define: [("BLOCK_M", "64"), ("BLOCK_N", "64"), ("BLOCK_K", "8"), ("THREAD_M", "8")],
        path: "src/device/vk/gemm.comp",
    }

    impl_compute_pipeline_new!(new, load);
}

#[test]
fn test_gemm() {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, Ix2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_stats::QuantileExt;
    use statrs::function::erf::erf_inv;
    use std::f32::consts::PI;

    use crate::{
        device::{DeviceInfo, TensorAllocateInfo, create_device},
        tensors::{
            owned::Tensor,
            views::{TensorMutTrait, TensorRefTrait},
        },
    };

    extern crate blas_src;

    #[allow(non_snake_case)]
    let MNK = [
        (16, 16, 16),
        (64, 64, 64),
        (1024, 1024, 1024),
        (16, 64, 1024),
        (12, 34, 56),
    ];

    #[allow(non_snake_case)]
    fn test_gemm_mnk(M: usize, N: usize, K: usize) -> anyhow::Result<()> {
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

    #[allow(non_snake_case)]
    for (M, N, K) in MNK {
        test_gemm_mnk(M, N, K).expect("GEMM test failed");
    }
}
