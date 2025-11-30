use wgpu::util::DeviceExt;
use crate::particle::Particle;
use crate::material::Material;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParticle {
    position: [f32; 3],
    _pad1: f32,
    velocity: [f32; 3],
    _pad2: f32,
    acceleration: [f32; 3],
    _pad3: f32,
    radius: f32,
    mass: f32,
    id: u32,
    _pad4: u32,
}

impl GpuParticle {
    fn from_particle(p: &Particle) -> Self {
        Self {
            position: p.position.to_array(),
            _pad1: 0.0,
            velocity: p.velocity.to_array(),
            _pad2: 0.0,
            acceleration: p.acceleration.to_array(),
            _pad3: 0.0,
            radius: p.radius,
            mass: p.mass,
            id: p.id as u32,
            _pad4: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimulationParams {
    bounds_min: [f32; 4], // x, y, z, dt
    bounds_max: [f32; 4], // x, y, z, padding
    periodic: [u32; 4],   // x, y, z, particle_count
    p_props: [f32; 4],    // E, nu, rho, mu
    p_props2: [f32; 4],   // e, pad, pad, pad
    w_props: [f32; 4],    // E, nu, rho, mu
    w_props2: [f32; 4],   // e, pad, pad, pad
    models: [u32; 4],     // normal_id, tangential_id, pad, pad
}

pub struct GpuSimulation {
    device: wgpu::Device,
    queue: wgpu::Queue,
    particle_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    particle_count: usize,
}

impl GpuSimulation {
    pub async fn new(
        particles: &[Particle], 
        dt: f32, 
        bounds_min: glam::Vec3, 
        bounds_max: glam::Vec3, 
        periodic: [bool; 3], 
        p_mat: Material, 
        w_mat: Material,
        normal_model: crate::physics::NormalForceModel,
        tangential_model: crate::physics::TangentialForceModel
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.ok_or("Failed to find an appropriate adapter")?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await.map_err(|e| format!("Failed to create device: {}", e))?;

        let gpu_particles: Vec<GpuParticle> = particles.iter().map(GpuParticle::from_particle).collect();
        
        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&gpu_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let params = SimulationParams {
            bounds_min: [bounds_min.x, bounds_min.y, bounds_min.z, dt],
            bounds_max: [bounds_max.x, bounds_max.y, bounds_max.z, 0.0],
            periodic: [periodic[0] as u32, periodic[1] as u32, periodic[2] as u32, particles.len() as u32],
            p_props: [p_mat.youngs_modulus, p_mat.poissons_ratio, p_mat.density, p_mat.friction_coefficient],
            p_props2: [p_mat.restitution_coefficient, 0.0, 0.0, 0.0],
            w_props: [w_mat.youngs_modulus, w_mat.poissons_ratio, w_mat.density, w_mat.friction_coefficient],
            w_props2: [w_mat.restitution_coefficient, 0.0, 0.0, 0.0],
            models: [normal_model as u32, tangential_model as u32, 0, 0],
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/physics.wgsl"));

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            particle_buffer,
            _params_buffer: params_buffer,
            compute_pipeline,
            bind_group,
            particle_count: particles.len(),
        })
    }

    pub fn step(&self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = (self.particle_count as f32 / 64.0).ceil() as u32;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }
    
    pub fn read_particles(&self) -> Vec<Particle> {
        let size = (self.particle_count * std::mem::size_of::<GpuParticle>()) as wgpu::BufferAddress;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.particle_buffer, 0, &staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        
        self.device.poll(wgpu::Maintain::Wait);
        
        let gpu_particles: Vec<GpuParticle> = if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to read GPU buffer");
        };
        
        gpu_particles.into_iter().map(|gp| {
            Particle {
                id: gp.id as usize,
                position: glam::Vec3::from_array(gp.position),
                velocity: glam::Vec3::from_array(gp.velocity),
                acceleration: glam::Vec3::from_array(gp.acceleration),
                radius: gp.radius,
                mass: gp.mass,
                fixed: false, // GPU doesn't support fixed particles yet
                position_residual: glam::Vec3::ZERO,
                velocity_residual: glam::Vec3::ZERO,
                angular_velocity: glam::Vec3::ZERO,
                torque: glam::Vec3::ZERO,
                angular_velocity_residual: glam::Vec3::ZERO,
            }
        }).collect()
    }
}
