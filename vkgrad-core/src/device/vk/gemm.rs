pub(super) mod f32 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/device/vk/gemm.comp",
    }

    impl_compute_pipeline_new!(new, load);

    pub const TILE_SIZE: usize = 16;
}
