use std::{
    any::Any,
    collections::{HashMap, HashSet},
    ffi::c_void,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use thiserror::Error;
use vulkano::{
    DeviceSize, NonZeroDeviceSize, Version, VulkanLibrary,
    buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferExecError, CommandBufferUsage, CopyBufferInfo,
        PrimaryAutoCommandBuffer,
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    },
    descriptor_set::{
        DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet,
        allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
    },
    device::{
        DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDevice,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::{
        DeviceAlignment,
        allocator::{
            AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter,
            StandardMemoryAllocator,
        },
    },
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        layout::{IntoPipelineLayoutCreateInfoError, PipelineDescriptorSetLayoutCreateInfo},
    },
    shader::ShaderModule,
    sync::{self, GpuFuture, HostAccessError},
};

use crate::{
    bits::BITS_PER_BYTE,
    device::{DeviceInfo, TensorAllocateInfo, assert_dim_eq},
    tensors::{
        dtype::TensorDataType,
        views::{Ptr, TensorMut, TensorRef, TensorRefTrait},
    },
};

pub fn compute_pipeline(
    device: Arc<vulkano::device::Device>,
    module: Arc<ShaderModule>,
) -> Result<Arc<ComputePipeline>, DeviceError> {
    let stage = PipelineShaderStageCreateInfo::new(module.single_entry_point().unwrap());
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())?,
    )?;
    Ok(ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )?)
}

macro_rules! impl_compute_pipeline_new {
    ($fn_name:ident, $shader_loader:expr) => {
        pub fn $fn_name(
            device: std::sync::Arc<vulkano::device::Device>,
        ) -> Result<
            std::sync::Arc<vulkano::pipeline::ComputePipeline>,
            crate::device::vk::DeviceError,
        > {
            crate::device::vk::compute_pipeline(device.clone(), $shader_loader(device)?)
        }
    };
}

mod gemm;

#[derive(Debug)]
pub enum TensorAccelerator {
    Plain,      // pure compute shader
    KhrCoopMat, // VK_KHR_cooperative_matrix
    NvCoopMat2, // VK_NV_cooperative_matrix2
}

impl Default for TensorAccelerator {
    fn default() -> Self {
        Self::Plain
    }
}

impl TensorAccelerator {
    pub fn vk_ext_names(&self) -> &[&'static str] {
        match self {
            TensorAccelerator::Plain => &[],
            TensorAccelerator::KhrCoopMat => &["VK_KHR_cooperative_matrix"],
            TensorAccelerator::NvCoopMat2 => {
                &["VK_KHR_cooperative_matrix", "VK_NV_cooperative_matrix2"]
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct BackendSpecificInfo {
    tensor_accelerator: Option<TensorAccelerator>,
}

#[derive(Default)]
struct BufferManager {
    buffers: Mutex<HashMap<usize, Arc<Buffer>>>,
    counter: AtomicUsize,
}

struct BufferHandle {
    manager: Arc<BufferManager>,
    handle: usize,
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        self.manager.drop_buffer(self.handle);
    }
}

impl BufferManager {
    fn new_handle(self: Arc<Self>, counter: usize) -> BufferHandle {
        BufferHandle {
            manager: self,
            handle: counter,
        }
    }

    pub fn register(self: Arc<Self>, buffer: Arc<Buffer>) -> BufferHandle {
        let counter = self.counter.fetch_add(1, Ordering::Relaxed) + 1;
        self.buffers
            .lock()
            .expect("poison error")
            .insert(counter, buffer);
        self.new_handle(counter)
    }

    pub fn get(&self, handle: usize) -> Option<Arc<Buffer>> {
        self.buffers
            .lock()
            .expect("poison error")
            .get(&handle)
            .cloned()
    }

    pub fn drop_buffer(&self, handle: usize) {
        self.buffers.lock().expect("poison error").remove(&handle);
    }
}

#[allow(dead_code)]
pub struct TensorManagerContext(BufferHandle);

pub struct Device {
    _library: Arc<VulkanLibrary>,
    _instance: Arc<Instance>,
    _physical_device: Arc<PhysicalDevice>,
    queue_family_idx: u32,
    device: Arc<vulkano::device::Device>,
    queue: Arc<Queue>,
    mem_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
    buffer_manager: Arc<BufferManager>,
    staging_buffer: Mutex<Arc<Buffer>>,
    staging_buffer_size: NonZeroDeviceSize,

    // pipelines
    gemm_f32: Arc<ComputePipeline>,
}

#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("Vulkan library loading error: {0}")]
    LibraryLoadError(#[from] vulkano::LoadingError),
    #[error("Generic vulkan error: {0}")]
    GenericError(#[from] vulkano::VulkanError),
    #[error("Vulkan validation error: {0}")]
    ValidationError(#[from] Box<vulkano::ValidationError>),
    #[error("No physical devices available")]
    NoPhysicalDevice,
    #[error("Unable to allocate buffer: {0}")]
    AllocateBufferError(#[from] AllocateBufferError),
    #[error("Invalid alignment value: {0}")]
    InvalidAlignmentError(usize),
    #[error("Invalid stride value")]
    InvalidStrideError,
    #[error("Unable to execute command buffer: {0}")]
    ExecError(#[from] CommandBufferExecError),
    #[error("Unable to access buffer: {0}")]
    HostAccessError(#[from] HostAccessError),
    #[error("{0}")]
    IntoPipelineLayoutCreateInfoError(#[from] IntoPipelineLayoutCreateInfoError),
}

impl<T> From<vulkano::Validated<T>> for DeviceError
where
    T: Into<DeviceError>,
{
    fn from(value: vulkano::Validated<T>) -> Self {
        match value {
            vulkano::Validated::Error(err) => err.into(),
            vulkano::Validated::ValidationError(validation_error) => {
                Self::ValidationError(validation_error)
            }
        }
    }
}

impl super::Device for Device {
    fn backend(&self) -> super::DeviceBackend {
        super::DeviceBackend::Vulkan
    }

    fn upcast(&self) -> &dyn super::Device {
        self
    }

    fn alloc_tensor_memory(
        &self,
        size_in_bytes: usize,
        alloc_info: &TensorAllocateInfo,
    ) -> Result<(*mut c_void, Box<dyn Any>), super::DeviceError> {
        let alignment =
            NonZeroDeviceSize::new((alloc_info.alignment.get() / BITS_PER_BYTE) as DeviceSize)
                .unwrap();
        let buffer = Buffer::new(
            self.mem_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                ..Default::default()
            },
            DeviceLayout::new(
                NonZeroDeviceSize::new(size_in_bytes as _).unwrap_or(NonZeroDeviceSize::MIN),
                DeviceAlignment::try_from(alignment)
                    .map_err(|_| DeviceError::InvalidAlignmentError(alloc_info.alignment.get()))?,
            )
            .expect("should not panic here"),
        )
        .map_err(DeviceError::from)?;

        // TODO: maybe use buffer.device_address() as the handle?
        let buffer = self.buffer_manager.clone().register(buffer);
        Ok((buffer.handle as _, Box::new(TensorManagerContext(buffer))))
    }

    unsafe fn read_back(
        &self,
        ptr: *const c_void,
        offset: usize,
        size: usize,
    ) -> Result<Box<[u8]>, super::DeviceError> {
        let staging_buffer = self.staging_buffer.lock().expect("poison error");
        let mut mem_buffer = Vec::with_capacity(size);
        let buffer = self.get_buffer(ptr)?;
        log::trace!(
            "Downloading {} bytes from Vulkan buffer {:?} at offset {}",
            size,
            buffer,
            offset,
        );
        let staging_buffer_size = self.staging_buffer_size.get();
        for i in 0..size.div_ceil(staging_buffer_size as _) {
            let i = i as DeviceSize;
            let range = (i * staging_buffer_size)
                ..((i + 1) * staging_buffer_size).min((offset + size) as _);
            self.exec(|mut cmd| {
                cmd.copy_buffer(CopyBufferInfo::buffers(
                    Subbuffer::new(buffer.clone()).slice(range.clone()),
                    Subbuffer::new(staging_buffer.clone()),
                ))?;
                Ok(cmd.build()?)
            })?;
            let subbuffer = Subbuffer::new(staging_buffer.clone());
            let data = subbuffer.read().map_err(DeviceError::from)?;
            mem_buffer.extend_from_slice(&data[..range.count()]);
        }
        assert!(mem_buffer.len() == size);
        Ok(mem_buffer.into_boxed_slice())
    }

    unsafe fn write_to(
        &self,
        ptr: *mut c_void,
        offset: usize,
        data: &[u8],
    ) -> Result<(), super::DeviceError> {
        let staging_buffer = self.staging_buffer.lock().expect("poison error");
        let buffer = self.get_buffer(ptr)?;
        log::trace!(
            "Uploading {} bytes to Vulkan buffer {:?} at offset {}",
            data.len(),
            buffer,
            offset,
        );
        let staging_buffer_size = self.staging_buffer_size.get() as usize;
        for i in 0..data.len().div_ceil(staging_buffer_size as _) {
            let range = (i * staging_buffer_size)
                ..((i + 1) * staging_buffer_size).min((offset + data.len()) as _);
            {
                let staging_subbuffer = Subbuffer::new(staging_buffer.clone());
                let mut write = staging_subbuffer.write().map_err(DeviceError::from)?;
                write[..range.clone().count()].copy_from_slice(&data[range.clone()]);
            }
            self.exec(|mut cmd| {
                cmd.copy_buffer(CopyBufferInfo::buffers(
                    Subbuffer::new(staging_buffer.clone()),
                    Subbuffer::new(buffer.clone())
                        .slice(range.start as DeviceSize..range.end as DeviceSize),
                ))
                .map_err(DeviceError::from)?;
                Ok(cmd.build()?)
            })?;
        }
        Ok(())
    }

    fn gemm(
        &self,
        lhs: &TensorRef,
        rhs: &TensorRef,
        ans: &mut TensorMut,
        alpha: f32,
        beta: f32,
        transpose_lhs: bool,
        transpose_rhs: bool,
    ) -> Result<(), super::DeviceError> {
        if lhs.dtype != rhs.dtype {
            return Err(super::DeviceError::MismatchDtype(lhs.dtype, rhs.dtype));
        }

        match lhs.dtype {
            TensorDataType::Float(32) => {
                self.gemm_f32(lhs, rhs, ans, alpha, beta, transpose_lhs, transpose_rhs)
            }
            _ => Err(super::DeviceError::UnsupportedFeature),
        }
    }
}

impl Device {
    pub fn new(info: DeviceInfo) -> Result<Self, DeviceError> {
        let library = VulkanLibrary::new()?;
        log::debug!("Loaded Vulkan library version {}", library.api_version());

        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                application_name: Some("vkgrad-core".into()),
                application_version: Version::major_minor(0, 1),
                engine_name: Some("vkgrad-core".into()),
                engine_version: Version::major_minor(0, 1),
                max_api_version: Some(Version::V1_4),
                ..Default::default()
            },
        )?;
        log::debug!(
            "Created Vulkan instance with API version {}",
            instance.max_api_version()
        );

        let (physical_device, compute_queue_family_idx) = instance
            .enumerate_physical_devices()?
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.contains(QueueFlags::COMPUTE))
                    .map(|i| (p, i))
            })
            .min_by_key(|(p, _)| {
                if Some(p.properties().device_name.as_str()) == info.physical_device_name.as_deref()
                {
                    -1
                } else {
                    match p.properties().device_type {
                        vulkano::device::physical::PhysicalDeviceType::DiscreteGpu => 0,
                        vulkano::device::physical::PhysicalDeviceType::IntegratedGpu => 1,
                        vulkano::device::physical::PhysicalDeviceType::VirtualGpu => 2,
                        vulkano::device::physical::PhysicalDeviceType::Cpu => 3,
                        vulkano::device::physical::PhysicalDeviceType::Other => 4,
                        _ => 5,
                    }
                }
            })
            .ok_or(DeviceError::NoPhysicalDevice)?;
        log::debug!(
            "Using physical device: {}",
            physical_device.properties().device_name
        );

        let extensions: HashSet<_> = physical_device
            .extension_properties()
            .iter()
            .map(|e| e.extension_name.as_str())
            .collect();

        let tensor_accelerator = info.vk.tensor_accelerator.unwrap_or_else(|| {
            for opt in [
                // TODO: add support for NvCoopMat2, which is not currently supported by vulkano
                // TensorAccelerator::NvCoopMat2,
                TensorAccelerator::KhrCoopMat,
            ] {
                if opt.vk_ext_names().iter().all(|n| extensions.contains(n)) {
                    return opt;
                }
            }

            TensorAccelerator::Plain
        });

        log::debug!(
            "Using {:?} as tensor accelerator with Vulkan extensions {:?}",
            tensor_accelerator,
            tensor_accelerator.vk_ext_names()
        );

        let mut device_extensions = DeviceExtensions::default();

        if extensions.contains("VK_KHR_shader_float16_int8") {
            device_extensions.khr_shader_float16_int8 = true;
            log::debug!("Enabling support for f16 and i8 on GPU");
        }

        for extension in tensor_accelerator.vk_ext_names() {
            match *extension {
                "VK_KHR_cooperative_matrix" => device_extensions.khr_cooperative_matrix = true,
                _ => {
                    return Err(DeviceError::GenericError(
                        vulkano::VulkanError::ExtensionNotPresent,
                    ));
                }
            }
        }

        let (device, queues) = vulkano::device::Device::new(
            physical_device.clone(),
            vulkano::device::DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: compute_queue_family_idx as u32,
                    ..Default::default()
                }],
                enabled_features: DeviceFeatures {
                    shader_integer_dot_product: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;

        let mut queues: Vec<_> = queues.collect();
        let queue = queues.pop().expect("should have one queue");

        let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let staging_buffer_size =
            NonZeroDeviceSize::new(info.staging_buffer_size.unwrap_or(4usize << 20) as _)
                .expect("Invalid staging buffer size"); // 4 MiB by default
        let staging_buffer = Buffer::new(
            mem_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::new(staging_buffer_size, DeviceAlignment::MIN)
                .expect("should not error here"),
        )?;

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let gemm_f32 = gemm::f32::new(device.clone())?;

        Ok(Self {
            _library: library,
            _instance: instance,
            _physical_device: physical_device,
            queue_family_idx: compute_queue_family_idx as _,
            mem_allocator,
            device,
            queue,
            command_buffer_allocator,
            descriptor_set_allocator,
            staging_buffer: Mutex::new(staging_buffer),
            staging_buffer_size,
            buffer_manager: Arc::new(Default::default()),
            gemm_f32,
        })
    }

    fn get_buffer(&self, ptr: *const c_void) -> Result<Arc<Buffer>, super::DeviceError> {
        self.buffer_manager
            .get(ptr as _)
            .ok_or(super::DeviceError::NonOwningTensorError)
    }

    fn exec(
        &self,
        f: impl FnOnce(
            AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ) -> Result<Arc<PrimaryAutoCommandBuffer>, DeviceError>,
    ) -> Result<(), DeviceError> {
        let cmd = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue_family_idx,
            CommandBufferUsage::OneTimeSubmit,
        )?;
        let cmd = f(cmd)?;
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cmd)?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        Ok(())
    }

    fn get_buffer_info(
        &self,
        tensor: &impl TensorRefTrait,
    ) -> Result<DescriptorBufferInfo, super::DeviceError> {
        let buffer = self.get_buffer(tensor.memory().to_ptr())?;
        let buffer = Subbuffer::new(buffer);
        let (offset, size, bit_offset) = tensor.byte_offset_and_sizes();
        if bit_offset != 0 {
            Err(DeviceError::InvalidAlignmentError(BITS_PER_BYTE).into())
        } else {
            Ok(DescriptorBufferInfo {
                buffer,
                range: offset as _..(offset + size) as _,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_f32(
        &self,
        lhs: &TensorRef,
        rhs: &TensorRef,
        ans: &mut TensorMut,
        alpha: f32,
        beta: f32,
        transpose_lhs: bool,
        transpose_rhs: bool,
    ) -> Result<(), super::DeviceError> {
        let lshape = lhs.shape();
        let rshape = rhs.shape();
        assert_dim_eq(lshape.len(), 2)?;
        assert_dim_eq(rshape.len(), 2)?;
        assert_dim_eq(lshape[1], rshape[0])?;

        #[allow(non_snake_case)]
        let M = lshape[0];
        #[allow(non_snake_case)]
        let N = rshape[1];
        #[allow(non_snake_case)]
        let K = lshape[1];

        log::trace!("Performing f32 GEMM: ({M}, {K}) x ({K}, {N}) -> ({M}, {N})");

        let mut stride_lhs = lhs.elem_strides().ok_or(DeviceError::InvalidStrideError)?;
        let mut stride_rhs = rhs.elem_strides().ok_or(DeviceError::InvalidStrideError)?;
        let stride_ans = ans.elem_strides().ok_or(DeviceError::InvalidStrideError)?;
        if transpose_lhs {
            stride_lhs.swap(0, 1);
        }
        if transpose_rhs {
            stride_rhs.swap(0, 1);
        }
        log::trace!(
            "LHS strides: {:?}, RHS strides: {:?}, ANS strides: {:?}",
            stride_lhs,
            stride_rhs,
            stride_ans
        );

        let desc_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.gemm_f32.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer_with_range(0, self.get_buffer_info(lhs)?),
                WriteDescriptorSet::buffer_with_range(1, self.get_buffer_info(rhs)?),
                WriteDescriptorSet::buffer_with_range(2, self.get_buffer_info(ans)?),
            ],
            [],
        )
        .map_err(DeviceError::from)?;
        self.exec(|mut cmd| {
            cmd.bind_pipeline_compute(self.gemm_f32.clone())?
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.gemm_f32.layout().clone(),
                    0,
                    desc_set,
                )?
                .push_constants(
                    self.gemm_f32.layout().clone(),
                    0,
                    gemm::f32::PushConsts {
                        M: M as _,
                        N: N as _,
                        K: (K as i32).into(),
                        stride_A: [stride_lhs[0] as _, stride_lhs[1] as _],
                        stride_B: [stride_rhs[0] as _, stride_rhs[1] as _],
                        stride_C: [stride_ans[0] as _, stride_ans[1] as _],
                        alpha,
                        beta,
                    },
                )?;
            unsafe {
                cmd.dispatch([
                    M.div_ceil(gemm::f32::TILE_SIZE) as _,
                    N.div_ceil(gemm::f32::TILE_SIZE) as _,
                    1,
                ])?
            };
            Ok(cmd.build()?)
        })?;
        Ok(())
    }
}
