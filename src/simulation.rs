use glam::Vec3;
use rayon::prelude::*;
use crate::particle::Particle;
use crate::grid::Grid;
use crate::mesh::Mesh;
use crate::material::Material;
use crate::physics::{self};
use crate::contact::Contact;
use crate::gpu;
use dashmap::DashMap;
use mpi::traits::*;

pub struct Simulation {
    pub particles: Vec<Particle>,
    pub grid: Grid,
    pub dt: f32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub meshes: Vec<Mesh>,
    pub contacts: DashMap<(usize, usize), Contact>,
    pub material: Material, // Global material for now, or per particle?
    // User asked for "specify the young's modulus and density of walls and particles seperately"
    // So we need materials for particles and walls.
    // Let's assume all particles share a material, and walls share a material for now.
    pub particle_material: Material,
    pub wall_material: Material,
    pub periodic: [bool; 3],
    pub gpu_sim: Option<gpu::GpuSimulation>,
    pub normal_model: physics::NormalForceModel,
    pub tangential_model: physics::TangentialForceModel,
    pub rank: i32,
    pub world_size: i32,
    pub universe: Option<mpi::environment::Universe>,
    pub ghosts: Vec<Particle>,
}

impl Simulation {
    pub fn new(dt: f32, bounds_min: Vec3, bounds_max: Vec3, particle_count: usize) -> Self {
        let particles = Vec::with_capacity(particle_count);
        // Heuristic for cell size: 2 * max_radius. Let's assume max_radius = 0.1 for now.
        // Heuristic for cell size: 2 * max_radius. Let's assume max_radius = 0.1 for now.
        let grid = Grid::new(1.0, bounds_min, bounds_max); // Cell size will be updated
        
        // Default materials
        let particle_material = Material::new(1e7, 0.3, 2500.0, 0.5, 0.5);
        let wall_material = Material::new(1e9, 0.3, 7800.0, 0.5, 0.5);

        Self {
            particles,
            grid,
            dt,
            bounds_min,
            bounds_max,
            meshes: Vec::new(),
            contacts: DashMap::new(),
            material: particle_material, // Deprecated
            particle_material,
            wall_material,
            periodic: [false; 3],
            gpu_sim: None,
            normal_model: physics::NormalForceModel::Hertzian,
            tangential_model: physics::TangentialForceModel::Mindlin,
            rank: 0,
            world_size: 1,
            universe: None,
            ghosts: Vec::new(),
        }
    }

    pub fn add_particle(&mut self, position: Vec3, radius: f32, mass: f32, fixed: bool) {
        let id = self.particles.len();
        self.particles.push(Particle::new(id, position, radius, mass, fixed));
    }

    pub fn add_mesh(&mut self, mesh: Mesh) {
        self.meshes.push(mesh);
    }

    pub fn init_mpi(&mut self) {
        use std::io::Write;
        println!("Initializing MPI...");
        std::io::stdout().flush().unwrap();
        
        if let Some(universe) = mpi::initialize() {
            println!("MPI Initialized.");
            let world = universe.world();
            self.rank = world.rank();
            self.world_size = world.size();
            self.universe = Some(universe);
            
            // Domain Decomposition (1D Slab along X)
            let domain_width = self.bounds_max.x - self.bounds_min.x;
            let slab_width = domain_width / self.world_size as f32;
            
            let original_min_x = self.bounds_min.x;
            
            self.bounds_min.x = original_min_x + self.rank as f32 * slab_width;
            self.bounds_max.x = original_min_x + (self.rank + 1) as f32 * slab_width;
            
            println!("Rank {}: Bounds X [{}, {}]", self.rank, self.bounds_min.x, self.bounds_max.x);
        } else {
            println!("MPI Initialization failed (None returned).");
        }
    }

    pub fn enable_gpu(&mut self) -> Result<(), String> {
        // Initialize GPU simulation
        // This requires async, but we are in sync context.
        // We use pollster to block.
        let gpu_sim = pollster::block_on(gpu::GpuSimulation::new(
            &self.particles,
            self.dt,
            self.bounds_min,
            self.bounds_max,
            self.periodic,
            self.particle_material,
            self.wall_material,
            self.normal_model,
            self.tangential_model,
        ))?;
        self.gpu_sim = Some(gpu_sim);
        Ok(())
    }

    pub fn step(&mut self) {
        if let Some(gpu_sim) = &mut self.gpu_sim {
            gpu_sim.step();
            // Sync back particles for VTK or other logic?
            // Doing it every step is slow.
            // But Simulation struct owns particles.
            // If we want to write VTK, we need to sync.
            // For now, let's NOT sync every step, only when needed (e.g. before write_vtk).
            // But step() updates self.particles in CPU mode.
            // If we switch modes, we need to sync.
            // Let's assume if GPU is enabled, we run on GPU.
            // We should add a sync_from_gpu() method.
        } else {
            self.step_cpu();
        }
    }

    fn step_cpu(&mut self) {
        let dt = self.dt;

        // 0. Handle boundaries (Wrap or Remove)
        let bounds_min = self.bounds_min;
        let bounds_max = self.bounds_max;
        let bounds_size = bounds_max - bounds_min;
        let periodic = self.periodic;
        
        if periodic[0] || periodic[1] || periodic[2] {
            // Wrap particles
            self.particles.par_iter_mut().for_each(|p| {
                if periodic[0] {
                    if p.position.x < bounds_min.x { p.position.x += bounds_size.x; }
                    else if p.position.x >= bounds_max.x { p.position.x -= bounds_size.x; }
                }
                if periodic[1] {
                    if p.position.y < bounds_min.y { p.position.y += bounds_size.y; }
                    else if p.position.y >= bounds_max.y { p.position.y -= bounds_size.y; }
                }
                if periodic[2] {
                    if p.position.z < bounds_min.z { p.position.z += bounds_size.z; }
                    else if p.position.z >= bounds_max.z { p.position.z -= bounds_size.z; }
                }
            });
            
            // Remove out of bounds for non-periodic dimensions
             self.particles.retain(|p| {
                (periodic[0] || (p.position.x >= bounds_min.x - 1.0 && p.position.x <= bounds_max.x + 1.0)) &&
                (periodic[1] || (p.position.y >= bounds_min.y - 1.0)) && // Allow falling a bit below if not periodic y
                (periodic[2] || (p.position.z >= bounds_min.z - 1.0 && p.position.z <= bounds_max.z + 1.0))
            });
        } else {
             // Original removal logic
            self.particles.retain(|p| {
                p.position.y >= bounds_min.y - 1.0 &&
                p.position.x >= bounds_min.x - 1.0 &&
                p.position.x <= bounds_max.x + 1.0 &&
                p.position.z >= bounds_min.z - 1.0 &&
                p.position.z <= bounds_max.z + 1.0
            });
        }
        
        // Re-index
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.id = i;
        }

        // 0.5. Exchange Particles (MPI)
        self.exchange_particles();
        
        // Re-index after exchange
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.id = i;
        }
        
        // 0.6. Exchange Ghosts
        self.exchange_ghosts();

        // 1. First Half Update (Velocity Verlet)
        // v(t + 0.5dt) = v(t) + 0.5 * a(t) * dt
        // x(t + dt) = x(t) + v(t + 0.5dt) * dt
        self.particles.par_iter_mut().for_each(|p| {
            if !p.fixed {
                // Kahan summation for Velocity
                let dv = 0.5 * p.acceleration * dt;
                let y_v = dv - p.velocity_residual;
                let t_v = p.velocity + y_v;
                p.velocity_residual = (t_v - p.velocity) - y_v;
                p.velocity = t_v;

                // Kahan summation for Position
                let dx = p.velocity * dt;
                let y_x = dx - p.position_residual;
                let t_x = p.position + y_x;
                p.position_residual = (t_x - p.position) - y_x;
                p.position = t_x;
            }
        });

        // 2. Force Calculation
        // Reset forces (gravity)
        let g = Vec3::new(0.0, -9.81, 0.0);
        self.particles.par_iter_mut().for_each(|p| {
            p.acceleration = g;
        });

        // Broad-phase
        self.grid.clear();
        for p in &self.particles {
            self.grid.insert(p);
        }
        // Insert ghosts into grid (with their IDs, which might conflict? No, grid uses ID for lookup)
        // Wait, if ghost has ID 0, and local particle has ID 0, grid lookup will return index 0 of local particles!
        // Grid stores indices? No, grid stores `p.id`.
        // And `get_potential_collisions` returns `Vec<usize>` which are IDs.
        // And we use `particles[j]`.
        // This is a problem. Ghosts need to be in `particles` array or handled separately.
        // If I put ghosts in `particles` array, they will be integrated.
        // I should append ghosts to `particles` temporarily?
        // Or `grid` should store `(is_ghost, index)`.
        // Or I append ghosts to `self.particles` but mark them?
        // Simplest: Append ghosts to `self.particles` before force calc, remove them after.
        // But `particles` is `Vec<Particle>`.
        
        let local_count = self.particles.len();
        self.particles.extend(self.ghosts.iter().cloned());
        
        // Re-insert ghosts to grid
        for (i, p) in self.particles.iter_mut().enumerate().skip(local_count) {
            // We need unique IDs for ghosts in the grid context.
            p.id = i; 
            self.grid.insert(p);
        }

        // Narrow-phase & Solve
        let p_mat = self.particle_material;
        let w_mat = self.wall_material;
        
        use rayon::prelude::*;
        use crate::physics::*;
        use crate::material::*;
        
        let grid = &self.grid;
        let particles = &self.particles;
        let contacts = &self.contacts;
        let meshes = &self.meshes;
        
        let forces: Vec<Vec3> = self.particles.par_iter().enumerate().map(|(i, p)| {
            let mut f_total = Vec3::ZERO;
            
            // Neighbor search
            let neighbors = grid.get_potential_collisions(p, periodic);
            for &j in &neighbors {
                if i == j { continue; }
                let other = &particles[j];
                
                let mut dist_vec = p.position - other.position;
                
                // Minimum Image Convention
                if periodic[0] {
                    if dist_vec.x > bounds_size.x * 0.5 { dist_vec.x -= bounds_size.x; }
                    else if dist_vec.x < -bounds_size.x * 0.5 { dist_vec.x += bounds_size.x; }
                }
                if periodic[1] {
                    if dist_vec.y > bounds_size.y * 0.5 { dist_vec.y -= bounds_size.y; }
                    else if dist_vec.y < -bounds_size.y * 0.5 { dist_vec.y += bounds_size.y; }
                }
                if periodic[2] {
                    if dist_vec.z > bounds_size.z * 0.5 { dist_vec.z -= bounds_size.z; }
                    else if dist_vec.z < -bounds_size.z * 0.5 { dist_vec.z += bounds_size.z; }
                }
                
                let dist_sq = dist_vec.length_squared();
                let r_sum = p.radius + other.radius;
                
                if dist_sq < r_sum * r_sum {
                    let dist = dist_sq.sqrt();
                    let overlap = r_sum - dist;
                    let normal = if dist > 1e-6 { dist_vec / dist } else { Vec3::Y };
                    let rel_vel = p.velocity - other.velocity;
                    
                    // Material properties
                    let r_star = effective_radius(p.radius, other.radius);
                    let m_star = effective_mass(p.mass, other.mass);
                    let e_star = effective_youngs_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio, p_mat.youngs_modulus, p_mat.poissons_ratio);
                    let g_star = effective_shear_modulus(shear_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio), shear_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio));
                    
                    // Contact state
                    // Key must be sorted to be unique for pair
                    let key = if i < j { (i, j) } else { (j, i) };
                    // We need to access DashMap.
                    // Since we are in a read-only pass over particles (conceptually), we can write to DashMap.
                    // But we are inside a par_iter map.
                    
                    // Note: DashMap might deadlock if we try to upgrade/downgrade?
                    // entry() is safe.
                    
                    let mut contact = contacts.entry(key).or_insert(Contact::new());
                    contact.age += 1;
                    
                    // Normal Force
                    let f_normal = match self.normal_model {
                        NormalForceModel::Hertzian => hertzian_contact(overlap, normal, rel_vel, e_star, r_star, m_star, p_mat.restitution_coefficient),
                        NormalForceModel::LinearSpringDashpot => {
                            // Need kn, gn. Derive from material props?
                            // For now, use some defaults or derived values.
                            // Kn ~ E * R
                            let kn = e_star * r_star;
                            // Gn ~ sqrt(m * kn)
                            let gn = 0.5 * (m_star * kn).sqrt();
                            linear_spring_dashpot(overlap, normal, rel_vel, kn, gn)
                        },
                        NormalForceModel::Hysteretic => {
                             // Placeholder
                             let kn = e_star * r_star;
                             hysteretic_contact(overlap, normal, &mut contact, kn, kn * 1.5)
                        }
                    };
                    
                    let fn_mag = f_normal.length();
                    
                    // Tangential Force
                    let f_tangent = match self.tangential_model {
                        TangentialForceModel::Mindlin => mindlin_contact(fn_mag, overlap, rel_vel, normal, &mut contact, g_star, r_star, p_mat.friction_coefficient, dt),
                        TangentialForceModel::Coulomb => coulomb_friction(fn_mag, rel_vel, normal, p_mat.friction_coefficient),
                        TangentialForceModel::LinearSpringCoulomb => {
                             // Kt ~ G * R
                             let kt = g_star * r_star;
                             linear_spring_coulomb(fn_mag, rel_vel, normal, &mut contact, kt, p_mat.friction_coefficient, dt)
                        }
                    };
                    
                    f_total += f_normal + f_tangent;
                }
            }
            
            // Particle-Mesh (Walls)
            // If periodic, we might skip wall collisions on periodic boundaries?
            // Usually periodic implies no walls on those sides.
            // But we might have internal meshes.
            // Let's assume meshes are internal or user handles placement.
            // However, explicit wall checks (like floor) need to be conditional.
            
            for mesh in meshes {
                for triangle in &mesh.triangles {
                    if let Some((penetration, normal)) = triangle.intersect_sphere(p.position, p.radius) {
                        let rel_vel = p.velocity; // Wall is static
                        
                        // Wall material
                        let r_star = p.radius; // Wall radius is infinite
                        let m_star = p.mass; // Wall mass infinite
                        let e_star = effective_youngs_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio, w_mat.youngs_modulus, w_mat.poissons_ratio);
                        let g_star = effective_shear_modulus(shear_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio), shear_modulus(w_mat.youngs_modulus, w_mat.poissons_ratio));
                        
                        // Hertzian normal force
                        let f_normal = hertzian_contact(penetration, normal, rel_vel, e_star, r_star, m_star, p_mat.restitution_coefficient);
                        
                        // Simple Coulomb for wall
                        let fn_mag = f_normal.length();
                        let f_tangent = coulomb_friction(fn_mag, rel_vel, normal, p_mat.friction_coefficient);
                        
                        f_total += f_normal + f_tangent;
                    }
                }
            }
            
            f_total
        }).collect();
        
        // Apply forces (only to local particles)
    self.particles.par_iter_mut().take(local_count).enumerate().for_each(|(i, p)| {
        p.acceleration += forces[i] / p.mass;
    });
    
    // Remove ghosts
    self.particles.truncate(local_count);

    // 3. Second Half Update (Velocity Verlet)
        // v(t + dt) = v(t + 0.5dt) + 0.5 * a(t + dt) * dt
        self.particles.par_iter_mut().for_each(|p| {
            if !p.fixed {
                let dv = 0.5 * p.acceleration * dt;
                let y_v = dv - p.velocity_residual;
                let t_v = p.velocity + y_v;
                p.velocity_residual = (t_v - p.velocity) - y_v;
                p.velocity = t_v;
            }
        });
    }

    pub fn sync_particles(&mut self) {
        if let Some(gpu_sim) = &self.gpu_sim {
            self.particles = gpu_sim.read_particles();
        }
    }

    pub fn exchange_particles(&mut self) {
        if self.world_size <= 1 {
            return;
        }
        
        // Identify particles leaving the domain
        let mut left_particles = Vec::new();
        let mut right_particles = Vec::new();
        
        let min_x = self.bounds_min.x;
        let max_x = self.bounds_max.x;
        
        // We need to retain particles that are inside.
        // But retain doesn't let us move out easily.
        // So we partition.
        
        let mut i = 0;
        while i < self.particles.len() {
            let p = self.particles[i];
            if p.position.x < min_x {
                left_particles.push(crate::particle::MpiParticle::from(p));
                self.particles.swap_remove(i);
            } else if p.position.x >= max_x {
                right_particles.push(crate::particle::MpiParticle::from(p));
                self.particles.swap_remove(i);
            } else {
                i += 1;
            }
        }
        
        let world = if let Some(universe) = &self.universe {
            universe.world()
        } else {
            return;
        };
        
        // Send Left (Rank - 1)
        let left_rank = if self.rank > 0 { self.rank - 1 } else { -1 }; // Or periodic?
        // Send Right (Rank + 1)
        let right_rank = if self.rank < self.world_size - 1 { self.rank + 1 } else { -1 };
        
        // We need to send and receive.
        // Use non-blocking or paired send/recv.
        
        // Send to Left, Recv from Right
        if left_rank >= 0 {
            world.process_at_rank(left_rank).send(&left_particles[..]);
        }
        
        // Recv from Right (which sent to its Left, i.e., us)
        if right_rank >= 0 {
            let (msg, status) = world.process_at_rank(right_rank).receive_vec::<crate::particle::MpiParticle>();
            for p in msg {
                self.particles.push(crate::particle::Particle::from(p));
            }
        }
        
        // Send to Right, Recv from Left
        if right_rank >= 0 {
            world.process_at_rank(right_rank).send(&right_particles[..]);
        }
        
        if left_rank >= 0 {
            let (msg, status) = world.process_at_rank(left_rank).receive_vec::<crate::particle::MpiParticle>();
            for p in msg {
                self.particles.push(crate::particle::Particle::from(p));
            }
        }
    }

    pub fn exchange_ghosts(&mut self) {
        if self.world_size <= 1 {
            return;
        }
        
        self.ghosts.clear();
        
        let world = if let Some(universe) = &self.universe {
            universe.world()
        } else {
            return;
        };
        
        let ghost_width = 0.1; // Should be max(radius) + cutoff. Hardcoded for now.
        
        let min_x = self.bounds_min.x;
        let max_x = self.bounds_max.x;
        
        // Identify ghosts to send Left
        let send_left: Vec<crate::particle::MpiParticle> = self.particles.iter()
            .filter(|p| p.position.x < min_x + ghost_width)
            .map(|p| crate::particle::MpiParticle::from(*p))
            .collect();
            
        // Identify ghosts to send Right
        let send_right: Vec<crate::particle::MpiParticle> = self.particles.iter()
            .filter(|p| p.position.x > max_x - ghost_width)
            .map(|p| crate::particle::MpiParticle::from(*p))
            .collect();
            
        let left_rank = if self.rank > 0 { self.rank - 1 } else { -1 };
        let right_rank = if self.rank < self.world_size - 1 { self.rank + 1 } else { -1 };
        
        // Send to Left, Recv from Right
        if left_rank >= 0 {
            world.process_at_rank(left_rank).send(&send_left[..]);
        }
        
        if right_rank >= 0 {
            let (msg, _) = world.process_at_rank(right_rank).receive_vec::<crate::particle::MpiParticle>();
            for p in msg {
                self.ghosts.push(crate::particle::Particle::from(p));
            }
        }
        
        // Send to Right, Recv from Left
        if right_rank >= 0 {
            world.process_at_rank(right_rank).send(&send_right[..]);
        }
        
        if left_rank >= 0 {
            let (msg, _) = world.process_at_rank(left_rank).receive_vec::<crate::particle::MpiParticle>();
            for p in msg {
                self.ghosts.push(crate::particle::Particle::from(p));
            }
        }
    }
}
