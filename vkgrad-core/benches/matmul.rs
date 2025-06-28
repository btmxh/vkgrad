use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::{Array, Array2, ArrayView2};
use ndarray_rand::{
    RandomExt,
    rand_distr::{Distribution, Uniform, num_traits::Float, uniform::SampleUniform},
};
use std::hint::black_box;
use vkgrad_core::{
    device::{DeviceBackend, DeviceInfo, TensorAllocateInfo, create_device},
    tensors::{
        dtype::TensorDataTypeTrait,
        owned::Tensor,
        views::{TensorMut, TensorMutTrait, TensorRef, TensorRefTrait},
    },
};

extern crate blas_src;

fn ndarray_matmul<A: Float + 'static>(a: ArrayView2<'_, A>, b: ArrayView2<'_, A>) -> Array2<A> {
    a.dot(&b)
}

fn vulkan_matmul(lhs: &TensorRef<'_>, rhs: &TensorRef<'_>, ans: &mut TensorMut<'_>) {
    lhs.device().gemm(lhs, rhs, ans).unwrap();
}

fn matmul_bench<S>(
    c: &mut Criterion,
    #[allow(non_snake_case)] M: usize,
    #[allow(non_snake_case)] N: usize,
    #[allow(non_snake_case)] K: usize,
    elem_dist: impl Distribution<S>,
) where
    S: Clone + SampleUniform + Float + 'static + TensorDataTypeTrait,
{
    let lhs = Array::random((M, K), &elem_dist);
    let rhs = Array::random((K, N), &elem_dist);

    let mut group = c.benchmark_group("matmul");
    group.bench_function(
        BenchmarkId::new("ndarray_matmul", format!("{M}x{N}x{K}")),
        |b| b.iter(|| black_box(ndarray_matmul(black_box(lhs.view()), black_box(rhs.view())))),
    );

    let device = create_device(DeviceInfo {
        backend: Some(DeviceBackend::Vulkan),
        ..Default::default()
    })
    .unwrap();

    let device_lhs = Tensor::from_ndarray(&*device, lhs.view()).unwrap();
    let device_rhs = Tensor::from_ndarray(&*device, rhs.view()).unwrap();
    let mut device_ans = device
        .alloc_tensor(TensorAllocateInfo::new_row_major(S::data_type(), &[M, N]))
        .unwrap();

    group.bench_function(
        BenchmarkId::new("vulkan_matmul", format!("{M}x{N}x{K}")),
        |b| {
            b.iter(|| {
                vulkan_matmul(
                    black_box(&device_lhs.as_ref()),
                    black_box(&device_rhs.as_ref()),
                    black_box(&mut device_ans.as_mut()),
                )
            })
        },
    );
}

fn matmul_benches(c: &mut Criterion) {
    matmul_bench(c, 1024, 1024, 1024, Uniform::new(-5.0f32, 5.0f32));
    matmul_bench(c, 2048, 2048, 2048, Uniform::new(-5.0f32, 5.0f32));
    matmul_bench(c, 4096, 4096, 4096, Uniform::new(-5.0f32, 5.0f32));
}

criterion_group!(benches, matmul_benches);
criterion_main!(benches);
