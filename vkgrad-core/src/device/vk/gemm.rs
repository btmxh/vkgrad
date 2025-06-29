pub(super) mod f32 {
    pub const TILE_SIZE: usize = 16;

    vulkano_shaders::shader! {
        ty: "compute",
        define: [("TILE_SIZE", "16")],
        path: "src/device/vk/gemm.comp",
    }

    impl_compute_pipeline_new!(new, load);
}
