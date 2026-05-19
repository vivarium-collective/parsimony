//! GPU acceleration for parsimony. Phase-4 in the design doc.
//!
//! v0: GPU port of `ClearanceGrid::update_for_placement` as a wgpu
//! compute shader. The grid is a `Vec<f32>` (f32::INFINITY = free,
//! 0 = occupied, positive = distance to nearest sphere surface);
//! each placement writes `min(current, |c−p| − r)` into every cell
//! within `r + max_required_radius` of the placement centre. That's
//! a stencil-style scatter — natural fit for compute shaders, since
//! many placements can be batched and many threads can update many
//! cells in parallel with atomic min ops on the buffer.
//!
//! Scope of this milestone:
//! - `GpuClearanceGrid` mirrors the CPU `ClearanceGrid` layout.
//! - `update_for_placements(slice)` uploads a placement batch and
//!   dispatches the compute kernel.
//! - `download()` reads the f32 grid back for verification.
//! - Cross-check property test: many random placements, GPU vs CPU
//!   results must agree to within FP noise.
//!
//! Out of scope here:
//! - Per-directive `valid_cells` filtering (next milestone).
//! - Placement loop / collision queries (after that).
//! - Mesh-ingredient proxy voxelisation (later still).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use nalgebra::Point3;
use thiserror::Error;
use wgpu::util::DeviceExt;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no compatible GPU adapter")]
    NoAdapter,
    #[error("device request failed: {0}")]
    Device(String),
}

/// Match the WGSL push-constant layout for one placement.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPlacement {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub radius: f32,
}

/// Uniform parameters shared by every dispatch.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridParams {
    origin: [f32; 3],
    cell_size: f32,
    dims: [u32; 3],
    range_max: f32, // r + max_required_radius (per-placement; we pass largest)
}

/// GPU mirror of `parsimony_core::clearance_grid::ClearanceGrid`.
/// Stores the clearance buffer on-device + the descriptor needed to
/// dispatch kernels against it.
pub struct GpuClearanceGrid {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pub origin: Point3<f32>,
    pub cell_size: f32,
    pub dims: [usize; 3],
    /// f32 storage buffer, one entry per cell. We use *unsigned*
    /// integer atomics in the shader (atomicMin on the bit-pattern
    /// of f32 happens to give correct results for non-negative
    /// values, and our grid only ever stores non-negative values:
    /// 0 = occupied, ∞ = free, anything else = positive distance).
    clearance: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    placements_buf: wgpu::Buffer,
    placements_capacity: usize,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuClearanceGrid {
    pub fn new(
        ctx: &GpuContext,
        origin: Point3<f32>,
        dims: [usize; 3],
        cell_size: f32,
    ) -> Self {
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();

        let n = dims[0] * dims[1] * dims[2];
        // Pre-fill the GPU buffer with the bit pattern of +∞.
        let inf_bits = f32::INFINITY.to_bits();
        let init: Vec<u32> = vec![inf_bits; n];
        let clearance = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("clearance grid"),
            contents: bytemuck::cast_slice(&init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid params"),
            size: std::mem::size_of::<GridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Start with capacity for 1024 placements per dispatch; we
        // reallocate on demand if batches grow.
        let placements_capacity = 1024;
        let placements_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("placement batch"),
            size: (placements_capacity * std::mem::size_of::<GpuPlacement>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clearance update kernel"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("clearance_update.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clearance bgl"),
            entries: &[
                // params: uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // placements: read-only storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // clearance: read-write storage (atomic<u32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clearance pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clearance pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("update_clearance"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            origin,
            cell_size,
            dims,
            clearance,
            params_buf,
            placements_buf,
            placements_capacity,
            pipeline,
            bind_group_layout,
        }
    }

    /// Dispatch the clearance-update kernel for one batch of
    /// placements. `max_required_radius` is the largest ingredient
    /// radius anyone might sample for (controls the kernel's per-
    /// placement write range, matching the CPU's `update_for_placement`).
    pub fn update_for_placements(
        &mut self,
        placements: &[GpuPlacement],
        max_required_radius: f32,
    ) {
        if placements.is_empty() {
            return;
        }

        if placements.len() > self.placements_capacity {
            // Grow the placement buffer.
            self.placements_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("placement batch"),
                size: (placements.len() * std::mem::size_of::<GpuPlacement>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.placements_capacity = placements.len();
        }

        self.queue.write_buffer(
            &self.placements_buf,
            0,
            bytemuck::cast_slice(placements),
        );

        let params = GridParams {
            origin: [self.origin.x, self.origin.y, self.origin.z],
            cell_size: self.cell_size,
            dims: [self.dims[0] as u32, self.dims[1] as u32, self.dims[2] as u32],
            range_max: max_required_radius,
        };
        self.queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::bytes_of(&params),
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clearance bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.placements_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.clearance.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("clearance update encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clearance update pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Workgroup size in the kernel is (4,4,4)=64 threads;
            // each thread handles one local cell. We dispatch one
            // workgroup per placement × per 4³ cell-block. The
            // kernel iterates the affected block from a per-placement
            // bbox computed inside the shader.
            pass.dispatch_workgroups(placements.len() as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Read the clearance buffer back to CPU memory. Synchronous;
    /// blocks until the GPU has flushed all pending work.
    pub fn download(&self) -> Vec<f32> {
        let n = self.dims[0] * self.dims[1] * self.dims[2];
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clearance readback"),
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.clearance, 0, &staging, 0,
            (n * std::mem::size_of::<f32>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        // wgpu 23: Device::poll(Maintain) returns a MaintainResult enum
        // (Ok/SubmissionQueueEmpty); we don't care which, just need to
        // have actually flushed.
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map result").expect("map read");
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

/// Bundles the wgpu device + queue. One per process is usually plenty.
pub struct GpuContext {
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    /// Pick a high-performance backend (Vulkan / Metal / DX12). Blocks
    /// on the async wgpu init via `pollster`.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("parsimony device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GpuError::Device(format!("{e:?}")))?;
        Ok(Self {
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

/// CPU reference implementation — the oracle the GPU kernel is
/// validated against. Same math as
/// `parsimony_core::clearance_grid::ClearanceGrid::update_for_placement`
/// but exposed as a standalone function so the GPU crate can call it
/// without depending on the (currently `pub(crate)`-scoped) module.
pub fn cpu_update(
    clearance: &mut [f32],
    dims: [usize; 3],
    origin: Point3<f32>,
    cell_size: f32,
    p: Point3<f32>,
    r: f32,
    max_required_radius: f32,
) {
    let range = r + max_required_radius;
    let inv_cs = 1.0 / cell_size;
    let lo = [
        ((p.x - range - origin.x) * inv_cs).floor() as i32,
        ((p.y - range - origin.y) * inv_cs).floor() as i32,
        ((p.z - range - origin.z) * inv_cs).floor() as i32,
    ];
    let hi = [
        ((p.x + range - origin.x) * inv_cs).floor() as i32,
        ((p.y + range - origin.y) * inv_cs).floor() as i32,
        ((p.z + range - origin.z) * inv_cs).floor() as i32,
    ];
    let r2_outer = range * range;
    let r2_inner = r * r;
    let stride_y = dims[0];
    let stride_z = dims[0] * dims[1];
    for cz in lo[2].max(0)..=hi[2].min(dims[2] as i32 - 1) {
        let wz = origin.z + (cz as f32 + 0.5) * cell_size;
        let dz = wz - p.z;
        let dz2 = dz * dz;
        let row_base_z = cz as usize * stride_z;
        for cy in lo[1].max(0)..=hi[1].min(dims[1] as i32 - 1) {
            let wy = origin.y + (cy as f32 + 0.5) * cell_size;
            let dy = wy - p.y;
            let dy2 = dy * dy;
            let row_base = row_base_z + cy as usize * stride_y;
            for cx in lo[0].max(0)..=hi[0].min(dims[0] as i32 - 1) {
                let wx = origin.x + (cx as f32 + 0.5) * cell_size;
                let dx = wx - p.x;
                let d2 = dx * dx + dy2 + dz2;
                if d2 > r2_outer {
                    continue;
                }
                let v = if d2 <= r2_inner { 0.0 } else { d2.sqrt() - r };
                let i = row_base + cx as usize;
                if v < clearance[i] {
                    clearance[i] = v;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// GPU and CPU produce the same clearance field for a batch of
    /// random placements. Allows a tiny FP tolerance because the GPU
    /// sqrt is implementation-defined.
    #[test]
    fn gpu_matches_cpu_oracle() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("skipping: no GPU available ({e})");
                return;
            }
        };
        let dims = [32usize, 32, 32];
        let cell_size = 2.0_f32;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let max_r = 10.0_f32;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        let n_placements = 64;
        let placements: Vec<GpuPlacement> = (0..n_placements)
            .map(|_| GpuPlacement {
                x: rng.gen_range(0.0..(dims[0] as f32 * cell_size)),
                y: rng.gen_range(0.0..(dims[1] as f32 * cell_size)),
                z: rng.gen_range(0.0..(dims[2] as f32 * cell_size)),
                radius: rng.gen_range(2.0..max_r),
            })
            .collect();

        // CPU reference.
        let mut cpu_grid = vec![f32::INFINITY; dims[0] * dims[1] * dims[2]];
        for p in &placements {
            cpu_update(
                &mut cpu_grid, dims, origin, cell_size,
                Point3::new(p.x, p.y, p.z), p.radius, max_r,
            );
        }

        // GPU.
        let mut gpu_grid = GpuClearanceGrid::new(&ctx, origin, dims, cell_size);
        gpu_grid.update_for_placements(&placements, max_r);
        let gpu = gpu_grid.download();

        // Compare.
        let n = cpu_grid.len();
        let mut max_diff = 0.0_f32;
        let mut mismatches = 0;
        for i in 0..n {
            let a = cpu_grid[i];
            let b = gpu[i];
            if a.is_infinite() && b.is_infinite() {
                continue;
            }
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
            if d > 1e-3 {
                mismatches += 1;
            }
        }
        assert!(
            mismatches == 0,
            "GPU and CPU clearance grids differ at {mismatches} cells (max diff {max_diff:.4})",
        );
    }
}
