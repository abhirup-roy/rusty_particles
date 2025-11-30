use glam::Vec3;
use crate::particle::Particle;
use crate::grid::Grid;
use crate::mesh::Mesh;
use crate::material::Material;
use crate::physics::{self};
use crate::contact::Contact;
use crate::gpu;
use dashmap::DashMap;
use mpi::traits::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// The core simulation engine.
///
/// Manages particles, the grid, forces, and time integration.
pub struct Simulation {
    /// The list of particles in the simulation.
    pub particles: Vec<Particle>,
    /// The spatial grid for broad-phase collision detection.
    pub grid: Grid,
    /// The time step (seconds).
    pub dt: f32,
    /// The minimum bounds of the simulation domain.
    pub bounds_min: Vec3,
    /// The maximum bounds of the simulation domain.
    pub bounds_max: Vec3,
    /// Static meshes (e.g., boundaries).
    pub meshes: Vec<Mesh>,
    /// Active contacts between particles.
    pub contacts: DashMap<(usize, usize), Contact>,
    /// Material properties for particles.
    pub particle_material: Material,
    /// Material properties for walls.
    pub wall_material: Material,
    /// Periodic boundary flags [x, y, z].
    pub periodic: [bool; 3],
    /// GPU simulation state (if enabled).
    pub gpu_sim: Option<gpu::GpuSimulation>,
    /// The normal force model.
    pub normal_model: physics::NormalForceModel,
    /// The tangential force model.
    pub tangential_model: physics::TangentialForceModel,
    /// MPI rank (for parallel execution).
    pub rank: i32,
    /// Total number of MPI processes.
    pub world_size: i32,
    /// MPI universe.
    pub universe: Option<mpi::environment::Universe>,
    /// Ghost particles for boundary communication.
    pub ghosts: Vec<Particle>,
}

impl Simulation {
    /// Creates a new simulation.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step (s).
    /// * `bounds_min` - Minimum domain coordinates.
    /// * `bounds_max` - Maximum domain coordinates.
    /// * `particle_count` - Initial particle capacity.
    pub fn new(dt: f32, bounds_min: Vec3, bounds_max: Vec3, particle_count: usize) -> Self {
        let particles = Vec::with_capacity(particle_count);
        // Heuristic for cell size: 2 * max_radius. Let's assume max_radius = 0.1 for now.
        // Heuristic for cell size: 2 * max_radius. Let's assume max_radius = 0.1 for now.
        let grid = Grid::new(1.0, bounds_min, bounds_max); // Cell size will be updated
        
        // Default materials
        let particle_material = Material::new(1e7, 0.3, 2500.0, 0.5, 0.5, 0.0);
        let wall_material = Material::new(1e9, 0.3, 7800.0, 0.5, 0.5, 0.0);

        Self {
            particles,
            grid,
            dt,
            bounds_min,
            bounds_max,
            meshes: Vec::new(),
            contacts: DashMap::new(),
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

    /// Advances the simulation by one time step.
    ///
    /// This method orchestrates the entire simulation loop:
    /// 1. Broad-phase collision detection (Grid).
    /// 2. Force calculation (Particle-Particle, Particle-Wall, Particle-Mesh).
    /// 3. Time integration (Symplectic Euler).
    /// 4. Boundary handling (Periodic/Reflective).
    /// 5. MPI communication (if parallel).
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

                // Rotational Integration (Angular Velocity)
                // I = 2/5 * m * r^2 (Solid Sphere)
                let inertia = 0.4 * p.mass * p.radius * p.radius;
                let angular_acc = p.torque / inertia;
                
                // Kahan summation for Angular Velocity
                let dw = 0.5 * angular_acc * dt;
                let y_w = dw - p.angular_velocity_residual;
                let t_w = p.angular_velocity + y_w;
                p.angular_velocity_residual = (t_w - p.angular_velocity) - y_w;
                p.angular_velocity = t_w;
            }
        });

        // 2. Force Calculation
        // Reset forces (gravity) and torque
        let g = Vec3::new(0.0, -9.81, 0.0);
        self.particles.par_iter_mut().for_each(|p| {
            p.acceleration = g;
            p.torque = Vec3::ZERO;
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
        
        let excessive_overlap = AtomicBool::new(false);
        let excessive_overlap_ref = &excessive_overlap;

        let forces: Vec<(Vec3, Vec3)> = self.particles.par_iter().enumerate().map(|(i, p)| {
            let mut f_total = Vec3::ZERO;
            let mut torque_total = Vec3::ZERO;
            
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
                
                // For JKR, we need to check for contact even if slightly separated (necking).
                // We add a margin.
                let margin = if matches!(self.normal_model, NormalForceModel::JKR | NormalForceModel::SimplifiedJKR) {
                    r_sum * 0.5 // Allow up to 50% radius separation? Maybe too much.
                    // Pull-off distance is usually small.
                } else {
                    0.0
                };
                
                let cutoff = r_sum + margin;
                
                if dist_sq < cutoff * cutoff {
                    let dist = dist_sq.sqrt();
                    let overlap = r_sum - dist;
                    
                    // Check for excessive overlap (> 20% of smaller radius)
                    // Avoid expensive check if flag is already set? 
                    // But we want to warn if ANY pair has excessive overlap.
                    // Relaxed ordering is fine.
                    if !excessive_overlap_ref.load(Ordering::Relaxed) {
                        let min_radius = p.radius.min(other.radius);
                        if overlap > 0.2 * min_radius {
                            excessive_overlap_ref.store(true, Ordering::Relaxed);
                        }
                    }

                    let normal = if dist > 1e-6 { dist_vec / dist } else { Vec3::Y };
                    
                    // Relative velocity at contact point
                    // v_rel = (v_a + w_a x r_a) - (v_b + w_b x r_b)
                    // Contact point is approx halfway along normal?
                    // Or r_a = -normal * radius_a, r_b = normal * radius_b
                    // normal points from B to A.
                    // r_a = -normal * p.radius (vector from center A to contact)
                    // r_b = normal * other.radius (vector from center B to contact)
                    
                    let r_a = -normal * p.radius;
                    let r_b = normal * other.radius;
                    
                    let vel_a = p.velocity + p.angular_velocity.cross(r_a);
                    let vel_b = other.velocity + other.angular_velocity.cross(r_b);
                    
                    let rel_vel = vel_a - vel_b;
                    
                    // Material properties
                    let r_star = effective_radius(p.radius, other.radius);
                    let m_star = effective_mass(p.mass, other.mass);
                    let e_star = effective_youngs_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio, p_mat.youngs_modulus, p_mat.poissons_ratio);
                    let g_star = effective_shear_modulus(shear_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio), shear_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio));
                    
                    // Contact state
                    let key = if i < j { (i, j) } else { (j, i) };
                    let mut contact = contacts.entry(key).or_insert(Contact::new());
                    contact.age += 1;
                    
                    // Normal Force
                    let f_normal = match self.normal_model {
                        NormalForceModel::Hertzian => hertzian_contact(overlap, normal, rel_vel, e_star, r_star, m_star, p_mat.restitution_coefficient),
                        NormalForceModel::LinearSpringDashpot => {
                            let kn = e_star * r_star;
                            let gn = 0.5 * (m_star * kn).sqrt();
                            linear_spring_dashpot(overlap, normal, rel_vel, kn, gn)
                        },
                        NormalForceModel::Hysteretic => {
                             let kn = e_star * r_star;
                             hysteretic_contact(overlap, normal, &mut contact, kn, kn * 1.5)
                        },
                        NormalForceModel::JKR => {
                            // Use effective surface energy?
                            // Usually gamma = sqrt(gamma1 * gamma2) or similar.
                            // Let's assume particle material gamma for P-P.
                            // Or average?
                            // JKR usually defines W = gamma1 + gamma2 - gamma12.
                            // For identical materials W = 2 * gamma.
                            // The formula uses gamma as "surface energy" or "work of adhesion"?
                            // User said "gamma: Surface energy (adhesion work)".
                            // If it's work of adhesion, we should combine.
                            // Let's assume the material property is surface energy per area.
                            // Work of adhesion W = 2 * gamma (for same material).
                            // The user's formula uses "gamma".
                            // "F_pull = -2/3 pi gamma R*".
                            // Standard JKR pull-off is -1.5 pi W R*.
                            // If user's gamma is W, then -1.5 pi gamma R*.
                            // User said -2/3 pi gamma R*.
                            // This matches DMT pull-off if gamma is W? No, DMT is -2 pi R W.
                            // Let's just pass the material's gamma directly as requested.
                            // If different materials, maybe average?
                            let gamma = (p_mat.surface_energy + p_mat.surface_energy) * 0.5;
                            compute_jkr_force(overlap, normal, rel_vel, e_star, r_star, gamma)
                        },
                        NormalForceModel::SimplifiedJKR => {
                            let gamma = (p_mat.surface_energy + p_mat.surface_energy) * 0.5;
                            compute_sjkr_force(overlap, normal, rel_vel, e_star, r_star, gamma)
                        }
                    };
                    
                    let fn_mag = f_normal.length();
                    
                    // Tangential Force
                    let f_tangent = match self.tangential_model {
                        TangentialForceModel::Mindlin => mindlin_contact(fn_mag, overlap, rel_vel, normal, &mut contact, g_star, r_star, p_mat.friction_coefficient, dt),
                        TangentialForceModel::Coulomb => coulomb_friction(fn_mag, rel_vel, normal, p_mat.friction_coefficient),
                        TangentialForceModel::LinearSpringCoulomb => {
                             let kt = g_star * r_star;
                             linear_spring_coulomb(fn_mag, rel_vel, normal, &mut contact, kt, p_mat.friction_coefficient, dt)
                        }
                    };
                    
                    let f_contact = f_normal + f_tangent;
                    f_total += f_contact;
                    
                    // Torque = r x F
                    // Force on A is f_contact.
                    // Torque on A = r_a x f_contact
                    torque_total += r_a.cross(f_contact);
                }
            }
            
            // Particle-Mesh (Walls)
            for mesh in meshes {
                for triangle in &mesh.triangles {
                    if let Some((penetration, normal)) = triangle.intersect_sphere(p.position, p.radius) {
                        // Check for excessive overlap with wall
                        if !excessive_overlap_ref.load(Ordering::Relaxed) {
                            if penetration > 0.2 * p.radius {
                                excessive_overlap_ref.store(true, Ordering::Relaxed);
                            }
                        }

                        // Wall is static (v=0, w=0)
                        // Contact point relative to particle center
                        let r_a = -normal * p.radius;
                        let vel_a = p.velocity + p.angular_velocity.cross(r_a);
                        let rel_vel = vel_a; // - 0
                        
                        // Wall material
                        let r_star = p.radius; 
                        let m_star = p.mass; 
                        let e_star = effective_youngs_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio, w_mat.youngs_modulus, w_mat.poissons_ratio);
                        let _g_star = effective_shear_modulus(shear_modulus(p_mat.youngs_modulus, p_mat.poissons_ratio), shear_modulus(w_mat.youngs_modulus, w_mat.poissons_ratio));
                        
                        // Normal force
                        let f_normal = match self.normal_model {
                            NormalForceModel::Hertzian => hertzian_contact(penetration, normal, rel_vel, e_star, r_star, m_star, p_mat.restitution_coefficient),
                            NormalForceModel::LinearSpringDashpot => {
                                let kn = e_star * r_star;
                                let gn = 0.5 * (m_star * kn).sqrt();
                                linear_spring_dashpot(penetration, normal, rel_vel, kn, gn)
                            },
                            NormalForceModel::Hysteretic => {
                                // Placeholder
                                let kn = e_star * r_star;
                                // We don't have contact state for walls easily yet.
                                // Fallback to linear
                                linear_spring_dashpot(penetration, normal, rel_vel, kn, 0.0)
                            },
                            NormalForceModel::JKR => {
                                let gamma = (p_mat.surface_energy + w_mat.surface_energy) * 0.5;
                                compute_jkr_force(penetration, normal, rel_vel, e_star, r_star, gamma)
                            },
                            NormalForceModel::SimplifiedJKR => {
                                let gamma = (p_mat.surface_energy + w_mat.surface_energy) * 0.5;
                                compute_sjkr_force(penetration, normal, rel_vel, e_star, r_star, gamma)
                            }
                        };
                        
                        // Simple Coulomb for wall
                        let fn_mag = f_normal.length();
                        let f_tangent = coulomb_friction(fn_mag, rel_vel, normal, p_mat.friction_coefficient);
                        
                        let f_contact = f_normal + f_tangent;
                        f_total += f_contact;
                        
                        // Torque
                        torque_total += r_a.cross(f_contact);
                    }
                }
            }
            
            (f_total, torque_total)
        }).collect();

        if excessive_overlap.load(Ordering::Relaxed) {
            eprintln!("Warning: Excessive particle overlap detected (>20% of radius). Simulation may be unstable.");
        }
        
        // Apply forces (only to local particles)
    self.particles.par_iter_mut().take(local_count).enumerate().for_each(|(i, p)| {
        p.acceleration += forces[i].0 / p.mass;
        p.torque += forces[i].1;
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

                // Rotational Integration (Second Half)
                let inertia = 0.4 * p.mass * p.radius * p.radius;
                let angular_acc = p.torque / inertia;
                
                let dw = 0.5 * angular_acc * dt;
                let y_w = dw - p.angular_velocity_residual;
                let t_w = p.angular_velocity + y_w;
                p.angular_velocity_residual = (t_w - p.angular_velocity) - y_w;
                p.angular_velocity = t_w;
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
            let (msg, _status) = world.process_at_rank(right_rank).receive_vec::<crate::particle::MpiParticle>();
            for p in msg {
                self.particles.push(crate::particle::Particle::from(p));
            }
        }
        
        // Send to Right, Recv from Left
        if right_rank >= 0 {
            world.process_at_rank(right_rank).send(&right_particles[..]);
        }
        
        if left_rank >= 0 {
            let (msg, _status) = world.process_at_rank(left_rank).receive_vec::<crate::particle::MpiParticle>();
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
