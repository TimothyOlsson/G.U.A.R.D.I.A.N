//! https://doc.rust-lang.org/beta/reference/procedural-macros.html
//! Could in theory make a macro that translates functions to wgpu
//! Thats a bit of work to do it, but would be very interesting
//! and could have a lot of usages for other types of work, meaning
//! it dynamically creates wgsl code from rust code

use std::collections::HashMap;
use std::borrow::Cow;
use std::time::Duration;

use anyhow::{anyhow, Result};
use tracing::info;
use wgpu::util::DeviceExt;

use super::wgsl_parsing;
use crate::GuardianSettings;

pub type Shape = [u32; 3];

// Common buffer usages combinations
#[derive(Debug, Copy, Clone)]
pub enum GpuBufferUsage {
    StageWrite,
    StageRead,
    Storage,
    Uniform,
}

impl Into<wgpu::BufferUsages> for GpuBufferUsage {
    fn into(self) -> wgpu::BufferUsages {
        match self {
            Self::StageWrite => {
              wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::MAP_WRITE
            },
            Self::StageRead => {
                wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_READ
            },
            Self::Storage => {
                wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
            }
            Self::Uniform => {
                wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
            }
        }
    }
}

#[allow(dead_code)]
pub struct GpuBuffer {
    name: String,
    buffer: wgpu::Buffer,
    usage: GpuBufferUsage,  // Can be taken from the .buffer as well
}

#[allow(dead_code)]
pub struct GpuConnection {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    limits: wgpu::Limits,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[allow(dead_code)]
pub struct GpuCompute {
    cs_module: wgpu::ShaderModule,
    compute_pipeline: wgpu::ComputePipeline,
    bind_groups: HashMap<u32, (wgpu::BindGroup, wgpu::BindGroupLayout)>,
    dispatch: Shape,
    pub timestamp: Timestamp,  // Start and end
}

pub struct Timestamp {
    query_set: wgpu::QuerySet,
    query_buffer: wgpu::Buffer,
    query_stage_buffer: wgpu::Buffer,
}

impl Timestamp {

    pub fn new(gpu_connection: &GpuConnection) -> Self {
        // Get timestamps
        let query_set = gpu_connection.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamps"),
            count: 2,  // start and stop
            ty: wgpu::QueryType::Timestamp,
        });
        let query_buffer = gpu_connection.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let query_stage_buffer = gpu_connection.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Stage Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            query_buffer,
            query_stage_buffer,
        }
    }

    pub fn set_timestamp(&self, encoder: &mut wgpu::CommandEncoder, timestamp_index: u32) {
        encoder.write_timestamp(&self.query_set, timestamp_index);  // End
    }

    pub fn resolve_timestamp(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.query_set,
            0..2,
            &self.query_buffer,
            0
        );
        encoder.copy_buffer_to_buffer(&self.query_buffer, 0, &self.query_stage_buffer, 0, 16);  // n_timestamps * u64::size
    }

    pub async fn get_timestamp(&self, gpu_connection: &GpuConnection) {
        let query_slice = self.query_stage_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        query_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        gpu_connection.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = receiver.recv_async().await {
            let timestamps: Vec<u64> = bytemuck::cast_slice(&query_slice.get_mapped_range()).to_vec();
            info!("{timestamps:?}");
            let timestamp_ns_start = (timestamps[0] as f64 * gpu_connection.queue.get_timestamp_period() as f64) as u64;
            let timestamp_ns_end = (timestamps[1] as f64 * gpu_connection.queue.get_timestamp_period() as f64) as u64;
            let dispatch_time_ns = timestamp_ns_end - timestamp_ns_start;
            info!("Elapsed: {:?}", Duration::from_nanos(dispatch_time_ns));
        }
    }
}

impl GpuConnection {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(anyhow!("Failed to find a proper GPU adapter!"))?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TIMESTAMP_QUERY,  // Empty if timestamp not needed
                    required_limits: limits.clone(),
                },
                None,
            )
            .await?;

        Ok(GpuConnection {
            instance,
            adapter,
            limits,
            device,
            queue,
        })
    }

    pub fn create_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    }

    pub fn compute(&self, to_compute: Vec<&GpuCompute>) {
        let mut encoder = self.create_encoder();
        for gpu_compute in to_compute {
            gpu_compute.timestamp.set_timestamp(&mut encoder, 0);
            let mut cpass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None
                }
            );
            cpass.set_pipeline(&gpu_compute.compute_pipeline);
            for (group, (bind_group, _)) in gpu_compute.bind_groups.iter() {
                cpass.set_bind_group(*group, bind_group, &[]);  // DynamicOffset not exposed
            }
            let [x, y, z] = gpu_compute.dispatch;
            cpass.dispatch_workgroups(x, y, z);
            drop(cpass);  // Free up the encoder, so we can set the end timestamp
            gpu_compute.timestamp.set_timestamp(&mut encoder, 1);
            gpu_compute.timestamp.resolve_timestamp(&mut encoder);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);  // This will not wait in wasm!
    }

    pub fn buffer2buffer(
        &self,
        offset_from: u64,
        offset_to: u64,
        gpu_buffer_from: &GpuBuffer,
        gpu_buffer_to: &GpuBuffer
    ) {
        let mut encoder = self.create_encoder();
        assert_eq!(gpu_buffer_from.buffer.size(), gpu_buffer_to.buffer.size());
        encoder.copy_buffer_to_buffer(
            &gpu_buffer_from.buffer,
            offset_from,
            &gpu_buffer_to.buffer,
            offset_to,
            gpu_buffer_from.buffer.size() as u64,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub async fn gpu2cpu(
        &self,
        gpu_buffer: &GpuBuffer
    ) -> Result<Vec<u8>> {
        let buffer_slice = gpu_buffer.buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait); // Not needed for wasm
        let finished = receiver.recv_async().await?;
        match finished {
            Ok(_) => {
                let data = buffer_slice.get_mapped_range();
                let result: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
                Ok(result)
            }
            Err(e) => Err(anyhow!(e)),
        }
    }

    pub fn get_info(&self) {
        info!("{:#?}", self.limits);
        info!("Explained:");
        info!("You can run at maximum of {} threads per worker", self.limits.max_compute_invocations_per_workgroup);
        info!("You can at maximum fit {} GB per buffer", self.limits.max_storage_buffer_binding_size / 1_000_000_000);
        info!("You can at maximum fit {} storage buffers per pipeline", self.limits.max_dynamic_storage_buffers_per_pipeline_layout);
        info!("With static buffers, you can at maximum fit {} GB per feature (with no binding optimizations)",
            (self.limits.max_storage_buffer_binding_size / 1_000_000_000) * 3000
        );
        info!("{:#?}", self.adapter.features());
        info!("Can run on webgpu: {:#?}", self.adapter.get_downlevel_capabilities().is_webgpu_compliant());
        info!("{:#?}", self.adapter.get_info());
    }
}

impl GpuCompute {
    pub fn new(
        gpu_connection: &GpuConnection,
        grouped_gpu_buffers: &HashMap<u32, Vec<&GpuBuffer>>,
        wgsl_shader: String,
        dispatch: Shape,
    ) -> Self {
        let cs_module = gpu_connection
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl_shader)),
            });
        let compute_pipeline =
            gpu_connection
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &cs_module,
                    entry_point: "main",
                });
        let mut bind_groups = HashMap::new();
        for (group, gpu_buffers) in grouped_gpu_buffers.into_iter() {
            let mut entries = vec![];
            for (binding, gpu_buffer) in gpu_buffers.into_iter().enumerate() {
                entries.push(
                    wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: gpu_buffer.buffer.as_entire_binding()
                    }
                );
            }
            let bind_group_layout = compute_pipeline.get_bind_group_layout(*group);
            let bind_group = gpu_connection
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &entries,
                });
            bind_groups.insert(*group, (bind_group, bind_group_layout));
        }
        let timestamp = Timestamp::new(gpu_connection);
        Self {
            cs_module,
            compute_pipeline,
            bind_groups,
            dispatch,
            timestamp
        }
    }
}

impl GpuBuffer {
    pub fn new(
        name: &str,
        gpu_connection: &GpuConnection,
        arr: &[u8],
        usage: GpuBufferUsage
    ) -> Self {
        let wgpu_usage = usage.clone().into();
        let buffer = gpu_connection
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: arr,
                usage: wgpu_usage,
            });
        Self {
            name: name.to_string(),
            buffer,
            usage
        }
    }
}


pub async fn test_webgpu() -> Result<()> {
    let guardian_settings = GuardianSettings::default();

    info!("Initiated test webgpu");
    let gpu_connection = GpuConnection::new().await?;
    gpu_connection.get_info();
    let stage_buffer = GpuBuffer::new("stage", &gpu_connection, bytemuck::cast_slice(&[0; 10_000]), GpuBufferUsage::StageRead);
    let buffer = GpuBuffer::new("terminals", &gpu_connection, bytemuck::cast_slice(&[0; 10_000]), GpuBufferUsage::Storage);
    info!("Creating compute");
    let gpu_compute = GpuCompute::new(
        &gpu_connection,
        &HashMap::from([
            (0, vec![&buffer]),
        ]),
        "hello".to_string(),
        [1, 1, 1],
    );
    info!("Computing");
    gpu_connection.compute(vec![&gpu_compute]);
    gpu_compute.timestamp.get_timestamp(&gpu_connection).await;
    info!("Copying to gpu buf -> gpu buf");
    gpu_connection.buffer2buffer(0, 0, &buffer, &stage_buffer);
    info!("Copying to gpu buf -> cpu");
    let result = gpu_connection.gpu2cpu(&stage_buffer).await?;
    info!("result = {:?}", result[0..4*20].to_vec());
    info!("result.len() = {:?}", result.len());
    let new_arr: Vec<f32> = bytemuck::cast_slice(&result[0..4*20]).to_vec();
    info!("new_arr = {new_arr:?}");
    Ok(())
}